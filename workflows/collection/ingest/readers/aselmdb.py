from __future__ import annotations
import os
import numpy as np
from ase.db import connect

from ..records import make_record


def _voigt_to_3x3(s):
    s = np.asarray(s, dtype=float)
    if s.shape == (3, 3):
        return s
    if s.size == 6:  # ASE Voigt order xx,yy,zz,yz,xz,xy
        xx, yy, zz, yz, xz, xy = s.ravel()
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return None


def read(path):
    base = os.path.basename(path)
    db = connect(path, readonly=True)
    for frame, row in enumerate(db.select()):
        a = row.toatoms()
        numbers = a.numbers
        positions = a.positions
        cell = np.asarray(a.cell.array, dtype=float)
        pbc = np.asarray(a.pbc, dtype=bool)
        energy = forces = stress = None
        if a.calc is not None:
            r = a.calc.results
            if "energy" in r:
                energy = float(r["energy"])
            if "forces" in r:
                forces = np.asarray(r["forces"], dtype=float)
            if "stress" in r:
                stress = _voigt_to_3x3(r["stress"])
        # row.unique_id is the per-row hash (genuinely unique + stable in the db)
        data = getattr(row, "data", None) or {}
        source_id = getattr(row, "unique_id", None)
        subset = data.get("data_id") or data.get("source")
        yield make_record(
            numbers, positions, cell, pbc,
            energy=energy, forces=forces, stress=stress,
            src_file=base, frame_idx=frame, source_id=source_id,
            extra_names=[f"subset:{subset}"] if subset else None,
        )
