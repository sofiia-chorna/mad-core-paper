from __future__ import annotations
import os
import numpy as np
from ase.data import atomic_numbers as ASE_Z

from ..records import make_record
from ._extxyz_common import iter_frames, parse_comment, parse_block_columns


def read(path):
    base = os.path.basename(path)
    with open(path, "rt") as f:
        for frame, (natoms, comment, block) in enumerate(iter_frames(f)):
            cell, pbc, _energy, props, entry_id = parse_comment(comment)
            if cell is None:
                cell = np.zeros((3, 3))
            if pbc is None:
                pbc = cell.any(axis=1)
            sp_s, sp_n = props["species"]
            po_s, _ = props["pos"]
            cols = parse_block_columns(block, natoms, [("species", sp_s, sp_n), ("pos", po_s, 3)])
            numbers = np.fromiter((ASE_Z[s] for s in cols["species"]), dtype=np.int64, count=natoms)
            yield make_record(
                numbers, cols["pos"], cell, pbc,
                energy=None, forces=None, stress=None,
                src_file=base, frame_idx=frame, source_id=entry_id,
            )
