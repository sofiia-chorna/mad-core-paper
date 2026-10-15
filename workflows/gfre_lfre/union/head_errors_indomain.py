from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from ase import io as ase_io
import metatomic.torch as mta
import vesin.metatomic as vesin_metatomic
from metatrain.utils.io import load_model

ROOT = "${DEV_ROOT}"
ATOMIC_COMBINED = f"{ROOT}/phace-atomic/data/atomic_combined"
MODEL_PATH = f"{ROOT}/phace-atomic/pre/phace_atomic_ep92.pt"
BLOCKS = [
    ("mad15", 0, 216803, f"{ROOT}/phace-atomic/data/mad15/xyz/mad-1.5-r2scan-val.xyz", 180184),
    ("mad_next", 216803, 638443, f"{ROOT}/phace-atomic/data/mad_next/mad_next.xyz", 216803),
    ("mad_sol", 638443, 677454, f"{ROOT}/phace-atomic/data/mad_sol/mad_sol.xyz", 638443),
]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = f"{HERE}/results_union/head_errors_indomain_val.json"
BATCH = 16

HEADS = {
    "pet": ("mtt::pet", "pet.bin", 512),
    "mace": ("mtt::mace", "mace.bin", 144),
    "uma": ("mtt::uma", "uma.bin", 128),
    "dpa_omat": ("mtt::dpa_omat", "dpa_omat.bin", 240),
    "dpa_omol": ("mtt::dpa_omol", "dpa_omol.bin", 240),
}


def load_selected_frames():
    atom_offsets = np.load(f"{ATOMIC_COMBINED}/na.npy")
    val_ids = np.loadtxt(f"{ATOMIC_COMBINED}/val_indices_in_all.txt", dtype=np.int64)

    frames, global_ids = [], []
    for name, block_start, block_end, xyz, xyz_start in BLOCKS:
        selected = np.sort(val_ids[(val_ids >= block_start) & (val_ids < block_end)])
        if len(selected) == 0:
            continue
        assert selected[0] >= xyz_start, \
            f"{name}: val id {selected[0]} precedes xyz file start {xyz_start}"
        wanted = set((selected - xyz_start).tolist())
        last_local = int(selected[-1] - xyz_start)

        found = 0
        for local_id, atoms in enumerate(ase_io.iread(xyz, index=":")):
            if local_id in wanted:
                gid = xyz_start + local_id
                expected = int(atom_offsets[gid + 1] - atom_offsets[gid])
                assert len(atoms) == expected, \
                    f"{name}[{local_id}]: natoms {len(atoms)} != na diff {expected}"
                frames.append(atoms)
                global_ids.append(gid)
                found += 1
            if found == len(selected) and local_id >= last_local:
                break
        assert found == len(selected), f"{name}: found {found}/{len(selected)} val frames"
        print(f"{name}: {found} val frames", flush=True)

    order = np.argsort(np.array(global_ids))
    return [frames[i] for i in order], np.array(global_ids, dtype=np.int64)[order], atom_offsets


def main():
    frames, global_ids, atom_offsets = load_selected_frames()
    natoms = np.array([len(a) for a in frames], dtype=np.int64)
    n_union_atoms = int(atom_offsets[-1])
    print(f"selected {len(frames):,} structures, {int(natoms.sum()):,} atoms", flush=True)

    targets = {name: np.memmap(f"{ATOMIC_COMBINED}/{bin_name}", dtype="float32",
                               mode="r", shape=(n_union_atoms, dim))
               for name, (_, bin_name, dim) in HEADS.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH).to(device)
    length_unit = model.capabilities().length_unit
    nl_calculators = vesin_metatomic.neighbor_lists_for_model(length_unit, model)
    options = mta.ModelEvaluationOptions(
        length_unit=length_unit,
        outputs={output: mta.ModelOutput(per_atom=True)
                 for output, _, _ in HEADS.values()},
        selected_atoms=None)
    print(f"model loaded  device={device}", flush=True)

    def predict(atoms_list):
        systems = mta.systems_to_torch(atoms_list)
        if not isinstance(systems, list):
            systems = [systems]
        for calculator in nl_calculators:
            calculator.add_neighbor_list(systems)
        systems = [s.to(dtype=torch.float32, device=device) for s in systems]
        with torch.no_grad():
            out = model(systems, options, check_consistency=False)
        return {name: out[output].block(0).values.detach().cpu().numpy().astype(np.float64)
                for name, (output, _, _) in HEADS.items()}

    stats = {name: {"sq": 0.0, "abs": 0.0, "n_components": 0,
                    "atoms_in": 0, "atoms_out": 0}
             for name in HEADS}

    n_failed = 0
    t_start = time.time()
    for i in range(0, len(frames), BATCH):
        chunk = frames[i:i + BATCH]
        chunk_gids = global_ids[i:i + len(chunk)]

        def accumulate(preds, gids_batch, counts):
            pos = 0
            for gid, count in zip(gids_batch, counts):
                row_start = int(atom_offsets[gid])
                for name in HEADS:
                    target = np.asarray(targets[name][row_start:row_start + count],
                                        dtype=np.float64)
                    pred = preds[name][pos:pos + count]
                    in_domain = np.isfinite(target).all(axis=1)
                    diff = pred[in_domain] - target[in_domain]
                    stats[name]["sq"] += float((diff ** 2).sum())
                    stats[name]["abs"] += float(np.abs(diff).sum())
                    stats[name]["n_components"] += diff.size
                    stats[name]["atoms_in"] += int(in_domain.sum())
                    stats[name]["atoms_out"] += int((~in_domain).sum())
                pos += count

        counts = [len(a) for a in chunk]
        try:
            accumulate(predict(chunk), chunk_gids, counts)
        except Exception as batch_err:
            print(f"  batch at frame {i} failed ({type(batch_err).__name__}); per-structure retry", flush=True)
            for c, atoms in enumerate(chunk):
                try:
                    accumulate(predict([atoms]), chunk_gids[c:c + 1], [len(atoms)])
                except Exception as frame_err:
                    n_failed += 1
                    print(f"  gid {int(chunk_gids[c])} dropped "
                          f"({type(frame_err).__name__}: {frame_err})", flush=True)
        if (i // BATCH) % 200 == 0:
            print(f"  {i + len(chunk)}/{len(frames)} frames  ({time.time() - t_start:.0f}s)", flush=True)

    results = {}
    for name, s in stats.items():
        rmse = float(np.sqrt(s["sq"] / s["n_components"]))
        mae = float(s["abs"] / s["n_components"])
        fraction_in = s["atoms_in"] / (s["atoms_in"] + s["atoms_out"])
        results[name] = {"rmse_indomain": rmse, "mae_indomain": mae,
                         "atoms_indomain": s["atoms_in"], "atoms_outdomain": s["atoms_out"],
                         "indomain_fraction": fraction_in}
        print(f"{name:10s} in-domain RMSE={rmse:.5f} MAE={mae:.5f} "
              f"({fraction_in:.1%} of atoms in domain)", flush=True)

    results["n_failed_structures"] = n_failed
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved => {OUT_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
