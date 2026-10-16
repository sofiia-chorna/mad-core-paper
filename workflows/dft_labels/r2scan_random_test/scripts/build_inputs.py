import csv
import sys
from pathlib import Path

import ase.io
from ase.data import atomic_numbers

# writes one FHI-aims input folder per selected structure
selection = sys.argv[1]
pbe_xyz = sys.argv[2]
species_dir = Path(sys.argv[3])
templates = Path(sys.argv[4])
out = Path(sys.argv[5])

with open(selection) as fh:
    rows = {r["configuration_id"]: r for r in csv.DictReader(fh, delimiter="\t")}

frames = [a for a in ase.io.read(pbe_xyz, index=":") if a.info["configuration_id"] in rows]

for atoms in frames:
    row = rows[atoms.info["configuration_id"]]
    folder = out / "frame_{:04d}".format(int(row["sample_rank"]))
    folder.mkdir(parents=True)

    lines = []
    if atoms.pbc.any():
        for v in atoms.cell:
            lines.append("lattice_vector {:.12f} {:.12f} {:.12f}".format(*v))
    for symbol, p in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append("atom {:.12f} {:.12f} {:.12f} {}".format(p[0], p[1], p[2], symbol))
    (folder / "geometry.in").write_text("\n".join(lines) + "\n")

    control = (templates / "control_r2scan_{}.in".format(row["dft_bucket"])).read_text()
    symbols = sorted(set(atoms.get_chemical_symbols()), key=lambda s: atomic_numbers[s])
    species = [(species_dir / "{:02d}_{}_default".format(atomic_numbers[s], s)).read_text() for s in symbols]
    (folder / "control.in").write_text(control + "\n" + "\n".join(species))
