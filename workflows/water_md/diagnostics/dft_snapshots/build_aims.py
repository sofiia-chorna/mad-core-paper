import json
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
SNAPSHOTS = HERE / "snapshots"
AIMS = HERE / "aims"

CAMPAIGN_CONTROL = Path("${DEV_ROOT}/lda_pbe/common/_upstream/control/periodic_ttt")
STORE_CONTROL = Path("${STORE_ROOT}/mad-core/lda_pbe/campaign_history/common/_upstream/control/periodic_ttt")
R2SCAN_TEMPLATE = HERE.parents[3] / "dft" / "r2scan_random_test" / "bundle" / "control_templates" / "control_r2scan_periodic_ttt.in"
BASIS = Path("${DEV_ROOT}/lda_pbe/common/defaults_2020/tight")
R2SCAN_SPECIES = HERE.parents[3] / "dft" / "r2scan_random_test" / "bundle" / "species"

FUNCTIONALS = {
    "pw-lda": CAMPAIGN_CONTROL / "control_pw-lda.in",
    "pbe": CAMPAIGN_CONTROL / "control_pbe.in",
    "r2scan": R2SCAN_TEMPLATE,
}
SPECIES_FILES = {"H": "01_H_default", "O": "08_O_default"}


def control_text(functional):
    template = FUNCTIONALS[functional].read_text()
    species_dir = R2SCAN_SPECIES if functional == "r2scan" else BASIS
    blocks = [(species_dir / SPECIES_FILES[s]).read_text() for s in ("H", "O")]
    return template.rstrip("\n") + "\n\n" + "\n\n".join(blocks)


def geometry_text(atoms):
    lines = [f"lattice_vector {v[0]:.12f} {v[1]:.12f} {v[2]:.12f}" for v in atoms.cell.array]
    lines += [f"atom {p[0]:.12f} {p[1]:.12f} {p[2]:.12f} {s}" for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]
    return "\n".join(lines) + "\n"


def reread(path):
    cell, positions, species = [], [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts[0] == "lattice_vector":
            cell.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            positions.append([float(x) for x in parts[1:4]])
            species.append(parts[4])
    return np.array(cell), np.array(positions), species


def main():
    # the scratch control files must equal the store copies
    for name in ("control_pw-lda.in", "control_pbe.in"):
        assert (CAMPAIGN_CONTROL / name).read_bytes() == (STORE_CONTROL / name).read_bytes(), name
    for s, f in SPECIES_FILES.items():
        assert (BASIS / f).read_bytes() == (R2SCAN_SPECIES / f).read_bytes(), s

    record = {"snapshots": {}, "control_sources": {k: str(v) for k, v in FUNCTIONALS.items()}, "basis": str(BASIS)}
    for path in sorted(SNAPSHOTS.glob("*.extxyz")):
        atoms = read(path, format="extxyz")
        assert atoms.pbc.all()
        name = path.stem
        for functional in FUNCTIONALS:
            work = AIMS / name / functional
            work.mkdir(parents=True, exist_ok=True)
            (work / "control.in").write_text(control_text(functional))
            (work / "geometry.in").write_text(geometry_text(atoms))
            cell, positions, species = reread(work / "geometry.in")
            assert species == atoms.get_chemical_symbols()
            assert np.abs(positions - atoms.positions).max() < 1e-9 and np.abs(cell - atoms.cell.array).max() < 1e-9
        record["snapshots"][name] = {"natoms": len(atoms), "cell_A": float(atoms.cell.array[0, 0]), "species_order_o_h_h": all(atoms.get_chemical_symbols()[3 * k: 3 * k + 3] == ["O", "H", "H"] for k in range(len(atoms) // 3))}
        print(name, len(atoms), "atoms, inputs written for", list(FUNCTIONALS))
    (AIMS / "INPUTS.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
