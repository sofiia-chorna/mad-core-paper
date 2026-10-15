from __future__ import annotations
import os

import numpy as np
from ase.io import iread

from ..records import make_record


def read(path):
    base = os.path.basename(path)
    for frame, atoms in enumerate(iread(path, index=":", format="extxyz")):
        cell = np.asarray(atoms.cell)            # (3,3); all-zero for the non-periodic monomers
        yield make_record(
            atoms.numbers,
            atoms.positions,
            cell,
            atoms.pbc,
            energy=atoms.get_potential_energy(),     # the `energy=` total energy (NOT atomization_energy)
            forces=atoms.get_forces(),
            stress=None,
            src_file=base,
            frame_idx=frame,
            source_id=None,
        )
