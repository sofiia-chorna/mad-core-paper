import hashlib
import json

import numpy as np
import torch
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from ase import io as ase_io
from metatrain.utils.io import load_model

from audit_overlap import composition_key, match_reference
from common import CHECKPOINTS, HERE, SETS, STORE

R2SCAN = f"{STORE}/mad-core/r2scan_random_test/results/test_1000_r2scan.xyz"
RELEASE = f"{STORE}/mad-core/release"
OUT = f"{HERE}/results_random_test/r2scan_eval.json"
BATCH = 16
BUCKETS = {"fff": "molecules", "ttt": "bulk", "surfaces": "surfaces"}


def read_tsv(path):
    with open(path) as handle:
        header = next(handle).rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def stats(values):
    errors = np.abs(np.asarray(values, dtype=np.float64))

    return {
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt((errors ** 2).mean())),
        "p8": float(((errors ** 8).mean()) ** 0.125),
        "sup": float(errors.max()),
    }


# energy errors per atom in meV and force errors over all cartesian components in meV per angstrom, as in the delta learning table
def metrics(energy_pred, energy_true, forces_pred, forces_true, natoms):
    energy = (np.asarray(energy_pred) - np.asarray(energy_true)) / np.asarray(natoms) * 1000
    forces = np.concatenate([((pred - true) * 1000).ravel() for pred, true in zip(forces_pred, forces_true)])

    return {
        "n_structures": int(len(energy)),
        "n_atoms": int(np.sum(natoms)),
        "energy_mev_per_atom": stats(energy),
        "forces_mev_per_angstrom": stats(forces),
    }


def load_predictor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CHECKPOINTS["phace-atomic"]).to(device)
    capabilities = model.capabilities()
    dtype = getattr(torch, capabilities.dtype)
    options = mta.ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs={
            "energy": mta.ModelOutput(unit="eV", per_atom=False),
            "non_conservative_forces": mta.ModelOutput(unit="eV/A", per_atom=True),
        },
        selected_atoms=None,
    )
    print(f"ecumetric on {device}, {capabilities.dtype}", flush=True)

    def predict(atoms_list):
        systems = mta.systems_to_torch(atoms_list)
        if not isinstance(systems, list):
            systems = [systems]

        vesin_metatomic.compute_requested_neighbors(systems, capabilities.length_unit, model)
        systems = [system.to(dtype=dtype, device=device) for system in systems]

        with torch.no_grad():
            output = model(systems, options, check_consistency=False)

        energy = output["energy"].block()
        energies = np.full(len(atoms_list), np.nan)
        energies[energy.samples.column("system").cpu().numpy()] = energy.values[:, 0].double().cpu().numpy()

        forces = output["non_conservative_forces"].block()
        system_index = forces.samples.column("system").cpu().numpy()
        atom_index = forces.samples.column("atom").cpu().numpy()
        values = forces.values[:, :, 0].double().cpu().numpy()
        per_frame = []
        for frame, atoms in enumerate(atoms_list):
            rows = np.full((len(atoms), 3), np.nan)
            mask = system_index == frame
            rows[atom_index[mask]] = values[mask]
            per_frame.append(rows)

        return energies, per_frame

    return predict


# stored release predictions of the same rows, with the atom order checked against the r2scan frame
def release_check(frames, locations, energies, forces):
    arrays = {}
    species_mismatch = 0
    position_deviation = 0.0
    energy_deviation = []
    force_deviation = 0.0

    for atoms, (group, row), energy, force in zip(frames, locations, energies, forces):
        folder = f"{RELEASE}/mad-next-coreset" if group == "mad-next-coreset" else f"{RELEASE}/collection/{group}"
        if folder not in arrays:
            arrays[folder] = {
                "na": np.load(f"{folder}/na.npy", mmap_mode="r"),
                "a": np.memmap(f"{folder}/a.bin", dtype=np.int32, mode="r"),
                "x": np.memmap(f"{folder}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3),
                "energy": np.memmap(f"{folder}/ecumetric_energy.bin", dtype=np.float32, mode="r"),
                "forces": np.memmap(f"{folder}/ecumetric_nc_forces.bin", dtype=np.float32, mode="r").reshape(-1, 3),
            }
        stored = arrays[folder]
        start, end = int(stored["na"][row]), int(stored["na"][row + 1])

        if end - start != len(atoms) or not np.array_equal(stored["a"][start:end], atoms.get_atomic_numbers()):
            species_mismatch += 1
            continue

        delta = atoms.get_positions() - np.asarray(stored["x"][start:end], dtype=np.float64)
        if atoms.pbc.any():
            cell = atoms.get_cell().array
            fractional = delta @ np.linalg.inv(cell)
            fractional -= np.round(fractional)
            delta = fractional @ cell
        position_deviation = max(position_deviation, float(np.abs(delta).max()))

        energy_deviation.append(abs(float(stored["energy"][row]) - energy) / len(atoms))
        force_deviation = max(force_deviation, float(np.abs(np.asarray(stored["forces"][start:end]) - force).max()))

    return {
        "compared": int(len(energy_deviation)),
        "species_or_count_mismatch": int(species_mismatch),
        "max_position_deviation_angstrom": position_deviation,
        "max_energy_deviation_ev_per_atom": float(np.max(energy_deviation)),
        "max_force_deviation_ev_per_angstrom": force_deviation,
    }


def overlap_check(frames):
    indexed = dict(enumerate(frames))
    wanted = {composition_key(atoms.get_atomic_numbers()) for atoms in frames}
    result = {}

    for name in ["space", "mad_next_tier"]:
        matches, reference = match_reference(name, indexed, wanted)
        splits = sorted({str(split) or "none" for split in np.unique(reference["split"])})
        result[name] = {"geometry_match": int(len(matches))}
        for split in splits:
            result[name][split] = int(sum(any(m["reference_split"] == split for m in hits) for hits in matches.values()))

    return result


def main():
    frames = ase_io.read(R2SCAN, index=":")
    release_ids = [atoms.info["configuration_id"] for atoms in frames]
    energy_true = np.array([float(atoms.info["atomization_energy"]) for atoms in frames])
    forces_true = [atoms.get_forces() for atoms in frames]
    natoms = np.array([len(atoms) for atoms in frames])
    print(f"{len(frames)} r2scan frames, {natoms.sum()} atoms", flush=True)

    # the r2scan file keys structures by release id, the random sample by its own id
    by_release = {row["release_configuration_id"]: row for row in read_tsv(f"{SETS}/release_ids.tsv")}
    selection = {row["configuration_id"]: row for row in read_tsv(f"{SETS}/selection.tsv")}
    rows = [selection[by_release[release_id]["configuration_id"]] for release_id in release_ids]
    buckets = np.array([BUCKETS[row["split_file"]] for row in rows])
    locations = [(row["dataset_group"], int(row["row_in_group"])) for row in rows]

    predict = load_predictor()
    energy_pred = np.full(len(frames), np.nan)
    forces_pred = [None] * len(frames)
    for start in range(0, len(frames), BATCH):
        energies, forces = predict(frames[start:start + BATCH])
        energy_pred[start:start + BATCH] = energies
        forces_pred[start:start + BATCH] = forces
    assert np.isfinite(energy_pred).all() and all(np.isfinite(force).all() for force in forces_pred)
    print("predictions done", flush=True)

    errors = {"all": metrics(energy_pred, energy_true, forces_pred, forces_true, natoms)}
    for bucket in BUCKETS.values():
        picks = np.where(buckets == bucket)[0]
        errors[bucket] = metrics(
            energy_pred[picks], energy_true[picks],
            [forces_pred[i] for i in picks], [forces_true[i] for i in picks], natoms[picks],
        )
    picks = np.where(buckets != "molecules")[0]
    errors["periodic"] = metrics(
        energy_pred[picks], energy_true[picks],
        [forces_pred[i] for i in picks], [forces_true[i] for i in picks], natoms[picks],
    )
    print(json.dumps(errors, indent=1), flush=True)

    release = release_check(frames, locations, energy_pred, forces_pred)
    print("release", release, flush=True)

    overlap = overlap_check(frames)
    print("overlap", overlap, flush=True)

    with open(CHECKPOINTS["phace-atomic"], "rb") as handle:
        checkpoint_sha256 = hashlib.sha256(handle.read()).hexdigest()

    result = {
        "labels": R2SCAN,
        "checkpoint": CHECKPOINTS["phace-atomic"],
        "checkpoint_sha256": checkpoint_sha256,
        "energy_target": "atomization_energy, r2scan isolated atom references of the mad record",
        "errors": errors,
        "release_predictions": release,
        "overlap": overlap,
        "per_structure": {
            "configuration_id": release_ids,
            "bucket": buckets.tolist(),
            "natoms": natoms.tolist(),
            "energy_true_ev": energy_true.tolist(),
            "energy_pred_ev": energy_pred.tolist(),
            "force_mae_ev_per_angstrom": [float(np.abs(p - t).mean()) for p, t in zip(forces_pred, forces_true)],
        },
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
