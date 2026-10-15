#!/usr/bin/env python3
"""
Convert an (ext)xyz file into a colabfit-schema parquet that the downstream
scripts (extract_ids.py, zbl_forces.py, filter_parquet_folder.py) read directly.

Output columns (one row per frame):
    configuration_id : string   (from --id-key info field; see below)
    nsites           : int32
    cell             : list<list<double>>   3x3 lattice (row vectors)
    positions        : list<list<double>>   N x 3 Cartesian (Angstrom)
    pbc              : list<bool>            length 3
    atomic_numbers   : string                JSON list, e.g. "[8, 1, 1]"
    energy           : double                (null if absent)
    atomic_forces    : list<list<double>>    N x 3 (null if absent)

configuration_id: taken from the frame's info under --id-key. With the default
(auto) it tries, in order: configuration_id, config_id, mat_id, id, name; if none
are present it falls back to "<stem>_<index>". Pass --id-key NAME to force one.
EXTRACTION REQUIRES this to match the ids you filter on -- check the printed
sample id before running extract_ids.py.

Streams frame-by-frame (one structure in memory) so multi-GB files are fine.

Usage:
    python xyz_to_parquet.py input.xyz output.parquet [--id-key configuration_id]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase.io import iread

_ID_FALLBACKS = ["configuration_id", "config_id", "mat_id", "id", "name"]

_SCHEMA = pa.schema([
    ("configuration_id", pa.string()),
    ("nsites", pa.int32()),
    ("cell", pa.list_(pa.list_(pa.float64()))),
    ("positions", pa.list_(pa.list_(pa.float64()))),
    ("pbc", pa.list_(pa.bool_())),
    ("atomic_numbers", pa.string()),
    ("energy", pa.float64()),
    ("atomic_forces", pa.list_(pa.list_(pa.float64()))),
])


def _pick_id(info, id_key, stem, idx):
    if id_key != "auto":
        return str(info[id_key]) if id_key in info else f"{stem}_{idx}"
    for k in _ID_FALLBACKS:
        if k in info:
            return str(info[k])
    return f"{stem}_{idx}"


def _energy(atoms):
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001
        return None


def _forces(atoms):
    # extxyz maps a column literally named 'forces' onto a SinglePointCalculator
    # results dict, NOT atoms.arrays. Check there first, then custom array names.
    res = getattr(atoms.calc, "results", {}) if atoms.calc is not None else {}
    if "forces" in res:
        return np.asarray(res["forces"], float).tolist()
    for key in ("forces", "force", "vasp_force", "REF_forces"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], float).tolist()
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Input (ext)xyz file.")
    ap.add_argument("output", help="Output parquet path.")
    ap.add_argument("--id-key", default="auto",
                    help="info field to use as configuration_id (default: auto).")
    ap.add_argument("--batch", type=int, default=2000,
                    help="Frames per parquet row group. Default 2000.")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    writer = pq.ParquetWriter(args.output, _SCHEMA, compression="zstd")

    buf = {c: [] for c in _SCHEMA.names}
    n = 0
    first_id = None

    def flush():
        if not buf["configuration_id"]:
            return
        writer.write_table(pa.table(buf, schema=_SCHEMA))
        for c in buf:
            buf[c].clear()

    for i, atoms in enumerate(iread(args.input)):
        cid = _pick_id(atoms.info, args.id_key, stem, i)
        if first_id is None:
            first_id = cid
        buf["configuration_id"].append(cid)
        buf["nsites"].append(len(atoms))
        buf["cell"].append(np.asarray(atoms.cell.array, float).tolist())
        buf["positions"].append(atoms.get_positions().tolist())
        buf["pbc"].append([bool(x) for x in atoms.pbc])
        buf["atomic_numbers"].append(json.dumps([int(z) for z in atoms.numbers]))
        buf["energy"].append(_energy(atoms))
        buf["atomic_forces"].append(_forces(atoms))
        n += 1
        if n % args.batch == 0:
            flush()
            print(f"  {n} frames...", flush=True)

    flush()
    writer.close()
    if n == 0:
        sys.exit("No frames read from input.")
    print(f"Wrote {n} structures -> {args.output}")
    print(f"  id-key={args.id_key} | sample configuration_id = {first_id!r}")


if __name__ == "__main__":
    main()

