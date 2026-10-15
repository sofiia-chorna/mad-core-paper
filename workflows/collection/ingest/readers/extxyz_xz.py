from __future__ import annotations
import lzma
import os
import numpy as np
from ase.data import atomic_numbers as ASE_Z

from ..records import make_record
from ._extxyz_common import iter_frames, parse_comment, parse_block_columns


def read(path):
    base = os.path.basename(path)
    with lzma.open(path, "rt") as f:
        for frame, (natoms, comment, block) in enumerate(iter_frames(f)):
            cell, pbc, energy, props, _ = parse_comment(comment)
            if cell is None:
                cell = np.zeros((3, 3))
            if pbc is None:
                pbc = cell.any(axis=1)
            sp_s, sp_n = props["species"]
            po_s, _ = props["pos"]
            specs = [("species", sp_s, sp_n), ("pos", po_s, 3)]
            have_f = "forces" in props
            if have_f:
                fo_s, _ = props["forces"]
                specs.append(("forces", fo_s, 3))
            cols = parse_block_columns(block, natoms, specs)
            numbers = np.fromiter((ASE_Z[s] for s in cols["species"]), dtype=np.int64, count=natoms)
            yield make_record(
                numbers, cols["pos"], cell, pbc,
                energy=energy, forces=cols["forces"] if have_f else None, stress=None,
                src_file=base, frame_idx=frame, source_id=None,
            )
