"""Shard helpers to featurize the heavy MOF datasets data-parallel across many GPUs.

Subcommands:
  split        norm parquet -> row shards  <out_dir>/<ds>@sNNNN.parquet
  convert-cifs strided slice of a dir's *.cif -> structure shard (parallel convert)
  filter       drop >1000-atom / ghost-atom (Z=0) / degenerate-cell rows
  concat       concat a glob of parquet shards -> one parquet
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase.io import read as ase_read

from inference.nc_ingest.convert import get_reader
from inference.nc_ingest.ids import mint_ids
from inference.nc_ingest.normalize import ShardWriter
from inference.nc_ingest.records import make_record


def keep_mask(t):
    n = t.num_rows
    keep = np.asarray(t.column("nsites")) <= 1000

    an = t.column("atomic_numbers").combine_chunks()
    flat = np.asarray(an.values)
    off = np.asarray(an.offsets)
    nonempty = np.diff(off) > 0
    row_min = np.zeros(n, dtype=np.int64)
    if nonempty.any():
        row_min[nonempty] = np.minimum.reduceat(flat, off[:-1][nonempty])
    keep &= row_min > 0

    cells = np.asarray(t.column("cell").to_pylist(), dtype=float).reshape(n, 3, 3)
    periodic = np.asarray(t.column("pbc").to_pylist()).any(axis=1)
    keep &= ~(periodic & (np.abs(np.linalg.det(cells)) < 1e-6))
    return keep


def cmd_split(a):
    t = pq.read_table(a.norm)
    n = t.num_rows
    os.makedirs(a.out_dir, exist_ok=True)
    k = 0
    for s in range(0, n, a.rows):
        sh = t.slice(s, min(a.rows, n - s))
        pq.write_table(sh, f"{a.out_dir}/{a.ds}@s{k:04d}.parquet")
        k += 1
    print(f"split {a.ds}: {n} rows -> {k} shards")


def cmd_convert_cifs(a):
    read, dataset_id = get_reader(a.dataset)
    files = sorted(glob.glob(os.path.join(a.extract, "*.cif")))
    mine = files[a.idx::a.num]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    w = ShardWriter(a.out, dataset_id, chunk_rows=20000)
    for fp in mine:
        rel = os.path.basename(fp)
        try:
            frames = ase_read(fp, index=":")
        except Exception:
            continue
        for i, at in enumerate(frames):
            rec = make_record(at.numbers, at.get_positions(), at.cell.array, at.pbc,
                              src_file=rel, frame_idx=i, source_id=f"{rel}:{i}")
            ids = mint_ids(rec["numbers"], rec["positions"], rec["cell"], rec["pbc"],
                          dataset=dataset_id, src_file=rec["src_file"],
                          frame_idx=rec["frame_idx"], energy=rec["energy"],
                          source_id=rec["source_id"])
            names = [f"{dataset_id}__{rel}__frame{i}", rec["source_id"]]
            w.add(rec, ids, names)
    written = w.close()
    print(f"convert-cifs {dataset_id} shard {a.idx}/{a.num}: {len(mine)} files -> {written} rows")


def cmd_filter(a):
    t = pq.read_table(a.inp)
    t = t.filter(pa.array(keep_mask(t)))
    pq.write_table(t, a.out)
    print(t.num_rows)


def cmd_concat(a):
    paths = sorted(glob.glob(a.glob))
    tables = [pq.read_table(p) for p in paths]
    out = pa.concat_tables(tables) if tables else None
    if out is None:
        raise SystemExit(f"no shards matched {a.glob}")
    pq.write_table(out, a.out)
    print(f"concat {len(paths)} shards -> {a.out} ({out.num_rows} rows)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("split"); p.set_defaults(fn=cmd_split)
    p.add_argument("--norm", required=True)
    p.add_argument("--ds", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--rows", type=int, default=25000)

    p = sub.add_parser("convert-cifs"); p.set_defaults(fn=cmd_convert_cifs)
    p.add_argument("--extract", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, required=True)
    p.add_argument("--idx", type=int, required=True)

    p = sub.add_parser("filter"); p.set_defaults(fn=cmd_filter)
    p.add_argument("inp")
    p.add_argument("out")

    p = sub.add_parser("concat"); p.set_defaults(fn=cmd_concat)
    p.add_argument("glob")
    p.add_argument("out")

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
