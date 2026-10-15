import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
sys.path.insert(0, str(D / "scripts"))
from water import FLOAT64, BASELINE_OUTPUT, DELTA_OUTPUT, Potential, atoms_call, sha256  # noqa: E402

SNAPSHOTS = HERE / "snapshots"
LABELLED = HERE / "existing_labels"
OUT = HERE / "tables"

MODELS = {
    "baseline": (FLOAT64 / "baseline_inner.float64.pt", BASELINE_OUTPUT),
    "delta_lda": (FLOAT64 / "delta_lda.float64.pt", DELTA_OUTPUT),
    "delta_pbe": (FLOAT64 / "delta_pbe.float64.pt", DELTA_OUTPUT),
}


def evaluate(potentials, atoms):
    out = {}
    for name, p in potentials.items():
        energy, forces = atoms_call(p, atoms)
        out[name] = {"energy_eV": energy, "forces": forces}
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    potentials = {name: Potential(path, output, device) for name, (path, output) in MODELS.items()}
    record = {"device": device, "models": {name: {"path": str(path), "sha256": sha256(path)} for name, (path, _) in MODELS.items()}, "snapshots": {}, "labelled": {}}
    arrays = {}

    for path in sorted(SNAPSHOTS.glob("*.extxyz")):
        atoms = read(path, format="extxyz")
        values = evaluate(potentials, atoms)
        md = atoms.arrays["md_forces"]
        record["snapshots"][path.stem] = {name: v["energy_eV"] for name, v in values.items()}
        for name, v in values.items():
            arrays[f"snapshots/{path.stem}/{name}"] = v["forces"]
        arrays[f"snapshots/{path.stem}/md_forces"] = md
        arrays[f"snapshots/{path.stem}/symbols"] = np.array(atoms.get_chemical_symbols())

        # the forces the trajectory saw must equal baseline minus correction for the run's own potential
        condition = atoms.info["run"].rsplit("_seed", 1)[0]
        combined = values["baseline"]["forces"].copy()
        if condition != "baseline":
            combined -= values["delta_" + condition.split("_")[0]]["forces"]
        record["snapshots"][path.stem]["md_force_gap_eV_A"] = float(np.abs(combined - md).max())
        print(path.stem, {k: round(v, 4) for k, v in record["snapshots"][path.stem].items()}, flush=True)

    for functional in ("lda", "pbe"):
        for i, atoms in enumerate(read(LABELLED / f"water_{functional}.xyz", index=":", format="extxyz")):
            cid = atoms.info["configuration_id"]
            if functional == "lda":
                values = evaluate(potentials, atoms)
                record["labelled"][cid] = {"natoms": len(atoms), "model_energy_eV": {name: v["energy_eV"] for name, v in values.items()}}
                for name, v in values.items():
                    arrays[f"labelled/{cid}/{name}"] = v["forces"]
                arrays[f"labelled/{cid}/symbols"] = np.array(atoms.get_chemical_symbols())
            # ase moves energy, free_energy and forces of an extxyz frame into a calculator
            record["labelled"][cid][f"{functional}_free_energy_eV"] = float(atoms.calc.results["free_energy"])
            arrays[f"labelled/{cid}/{functional}_forces"] = atoms.calc.results["forces"]
            print(functional, cid, len(atoms), flush=True)

    np.savez(OUT / "model_predictions.npz", **arrays)
    (OUT / "model_predictions.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
