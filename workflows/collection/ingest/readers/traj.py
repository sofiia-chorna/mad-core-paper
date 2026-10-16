from __future__ import annotations
import os
import numpy as np
from ase.io import read as ase_read

from ..records import make_record


def _try(getter):
    try:
        return getter()
    except Exception:
        return None


def read(path):
    base = os.path.basename(path)
    images = ase_read(path, ":")
    for frame, a in enumerate(images):
        numbers = a.numbers
        positions = a.positions
        cell = np.asarray(a.cell.array, dtype=float)
        pbc = np.asarray(a.pbc, dtype=bool)
        # geometry-only in the filtered passed trajs carry E/F only if actually present
        energy = _try(a.get_potential_energy)
        forces = _try(a.get_forces)
        yield make_record(
            numbers, positions, cell, pbc,
            energy=energy, forces=forces, stress=None,
            src_file=base, frame_idx=frame, source_id=None,
        )
