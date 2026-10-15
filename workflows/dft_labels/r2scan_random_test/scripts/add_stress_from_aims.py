import argparse
import re
import warnings
from pathlib import Path

import ase.io
import numpy as np

warnings.filterwarnings("ignore")

RANK_RE = re.compile(r"sample_rank=(\d+)")
ENERGY_RE = re.compile(r"(?<![A-Za-z_])energy=(-?[\d.]+(?:[eE][-+]?\d+)?)")
PROPERTIES_RE = re.compile(r"\sProperties=")

ENERGY_TOL_EV = 1e-6
FORCE_TOL_EV_A = 1e-6


def read_frames(path):
    with open(path) as handle:
        while True:
            head = handle.readline()
            if not head:
                return
            natoms = int(head.strip())
            comment = handle.readline()
            body = [handle.readline() for _ in range(natoms)]
            yield natoms, comment, body


def format_stress(stress):
    return " ".join(f"{value:.8f}" for value in stress.reshape(9))


def insert_stress(comment, stress_text):
    match = PROPERTIES_RE.search(comment)
    assert match, "no Properties= key in comment line"
    cut = match.start()
    return f'{comment[:cut]} stress="{stress_text}"{comment[cut:]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aims-dir", required=True, type=Path)
    parser.add_argument("--in-xyz", required=True, type=Path)
    parser.add_argument("--out-xyz", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    args.out_xyz.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_written = 0
    n_stress = 0

    with open(args.out_xyz, "w") as out:
        for natoms, comment, body in read_frames(args.in_xyz):
            rank = int(RANK_RE.search(comment).group(1))
            energy_xyz = float(ENERGY_RE.search(comment).group(1))
            periodic = 'pbc="T T T"' in comment

            aims_path = args.aims_dir / f"frame_{rank:04d}" / "aims.out"
            atoms = ase.io.read(aims_path, format="aims-output")

            assert len(atoms) == natoms, f"rank {rank}: atom count mismatch"

            energy_aims = atoms.get_potential_energy()
            d_energy = abs(energy_aims - energy_xyz)
            assert d_energy < ENERGY_TOL_EV, f"rank {rank}: energy mismatch {d_energy}"

            forces_xyz = np.array([[float(v) for v in line.split()[4:7]] for line in body])
            d_force = float(np.abs(atoms.get_forces() - forces_xyz).max())
            assert d_force < FORCE_TOL_EV_A, f"rank {rank}: force mismatch {d_force}"

            if periodic:
                stress = atoms.get_stress(voigt=False)
                comment_out = insert_stress(comment.rstrip("\n"), format_stress(stress)) + "\n"
                pressure = -float(np.trace(stress)) / 3.0
                n_stress += 1
            else:
                comment_out = comment
                pressure = float("nan")

            out.write(f"{natoms}\n")
            out.write(comment_out)
            out.writelines(body)
            n_written += 1

            rows.append((rank, natoms, "periodic_ttt" if periodic else "isolated_fff",
                         f"{d_energy:.3e}", f"{d_force:.3e}",
                         f"{pressure:.8f}" if periodic else ""))

    with open(args.report, "w") as handle:
        handle.write("sample_rank\tnatoms\tdft_bucket\tdenergy_vs_xyz_eV\tdforce_vs_xyz_eV_A\tpressure_eV_A3\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")

    print(f"frames written : {n_written}")
    print(f"stress added   : {n_stress}")
    print(f"out            : {args.out_xyz}")
    print(f"report         : {args.report}")


if __name__ == "__main__":
    main()
