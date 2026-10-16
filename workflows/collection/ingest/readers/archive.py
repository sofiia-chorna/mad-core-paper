from __future__ import annotations
import os

from ase.io import read as ase_read
from ase.db import connect

from ..records import make_record

STRUCT_EXT = {".xyz", ".extxyz", ".cif", ".traj"}
VASP_PREFIX = ("POSCAR", "CONTCAR", "XDATCAR")


def _emit(atoms, src_file, idx):
    return make_record(
        atoms.numbers, atoms.get_positions(), atoms.cell.array, atoms.pbc,
        src_file=src_file, frame_idx=idx, source_id=f"{src_file}:{idx}",
    )


def read(path):
    # path is a directory holding the extracted archive members
    for dp, _, fns in os.walk(path):
        if "__MACOSX" in dp:
            continue
        for fn in sorted(fns):
            if fn.startswith("._"):  # macOS resource-fork junk, not a real structure file
                continue
            fp = os.path.join(dp, fn)
            rel = os.path.relpath(fp, path)
            ext = os.path.splitext(fn)[1].lower()
            try:
                if ext == ".db":
                    with connect(fp) as db:
                        for i, row in enumerate(db.select()):
                            yield _emit(row.toatoms(), rel, i)
                elif ext in STRUCT_EXT or fn.startswith(VASP_PREFIX):
                    frames = ase_read(fp, index=":")
                    for i, a in enumerate(frames):
                        yield _emit(a, rel, i)
            except Exception as e:
                print(f"[archive] skip {rel}: {type(e).__name__}: {str(e)[:80]}", flush=True)
