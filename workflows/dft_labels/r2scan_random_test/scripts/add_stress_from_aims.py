import re
import sys

import ase.io

# copies the r2SCAN stress of the periodic frames from the aims output into the xyz
aims_dir = sys.argv[1]
in_xyz = sys.argv[2]
out_xyz = sys.argv[3]

RANK = re.compile(r"sample_rank=(\d+)")
ENERGY = re.compile(r"(?<![A-Za-z_])energy=(\S+)")


def frames(path):
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head:
                return
            natoms = int(head)
            comment = fh.readline()
            body = [fh.readline() for _ in range(natoms)]
            yield natoms, comment, body


with open(out_xyz, "w") as out:
    for natoms, comment, body in frames(in_xyz):

        if 'pbc="T T T"' in comment:
            rank = int(RANK.search(comment).group(1))
            atoms = ase.io.read(f"{aims_dir}/frame_{rank:04d}/aims.out", format="aims-output")

            assert len(atoms) == natoms
            assert abs(atoms.get_potential_energy() - float(ENERGY.search(comment).group(1))) < 1e-6

            stress = " ".join(f"{v:.8f}" for v in atoms.get_stress(voigt=False).reshape(9))
            cut = comment.index(" Properties=")
            comment = f'{comment[:cut]} stress="{stress}"{comment[cut:]}'

        out.write(f"{natoms}\n")
        out.write(comment)
        out.writelines(body)
