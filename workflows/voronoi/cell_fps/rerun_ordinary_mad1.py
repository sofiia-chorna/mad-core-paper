#!/usr/bin/env python3
"""Targeted within-cell FPS rerun for the ordinary (non-mega) cells affected by
the Massive_Atomic_Diversity_MAD removal.

Identical machinery to run_cell_fps.py (imported, not copied); differences:
  - keeps only rows whose cell is in mad1_removal/mad1_rerun_cells.npy
  - drops rows with dataset_group == Massive_Atomic_Diversity_MAD
  - writes to mad1_removal/fps_rerun/<bucket>.parquet, never touching fps/
Merge into fps/b*.parquet is a separate reviewed step."""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from coresets.cuda.extension import ensure_compiled, get_ext
from run_cell_fps import (OUT_SCHEMA, SCALER, load_bucket, process_rows, verify)

OUT_ROOT = ("${DEV_ROOT}/phace-atomic/voronoi/out/"
            "cell_fps")
RERUN_CELLS = np.load(f"{OUT_ROOT}/mad1_removal/mad1_rerun_cells.npy")
GONE = "Massive_Atomic_Diversity_MAD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.bucket_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    done_marker = f"{args.out_dir}/{name}.done"
    if os.path.exists(done_marker):
        print(f"{name}: already done")
        return

    b = int(name[1:])
    my_cells = np.array([c for c in RERUN_CELLS if c % 128 == b], dtype=np.int64)
    if len(my_cells) == 0:
        with open(done_marker, "w") as fh:
            fh.write("no cells\n")
        print(f"{name}: no rerun cells")
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    ensure_compiled()
    ext = get_ext()

    scaler = np.load(SCALER)
    mean = scaler["mean"].astype(np.float32)
    inv_scale = np.float32(1.0 / float(scaler["scale"]))

    t0 = time.time()
    table, cells, d24, cid, features = load_bucket(args.bucket_dir)
    grp = np.asarray(table.column("dataset_group").to_pandas(), dtype=object)
    keep = np.isin(cells, my_cells) & (grp != GONE)
    idx = np.nonzero(keep)[0]
    table = table.take(idx)
    cells, d24, cid, features = cells[idx], d24[idx], cid[idx], features[idx]
    assert not np.isnan(features).any()
    t_load = time.time() - t0
    print(f"{name}: {len(cells):,} rerun rows in {len(my_cells)} cells "
          f"(loaded {t_load:.0f}s)", flush=True)

    timings = []
    t0 = time.time()
    rank_out, eps_out, unique_cells, boundaries, order, cell_center_pos = process_rows(
        device, ext, cells, cid, features, mean, inv_scale, timings)
    t_fps = time.time() - t0
    problems = verify(d24, rank_out, eps_out, unique_cells, boundaries, order,
                      cell_center_pos)

    final = np.lexsort((rank_out, cells))
    out_table = pa.table({
        "cell": pa.array(cells[final]),
        "cell_rank": pa.array(rank_out[final]),
        "eps_sq": pa.array(eps_out[final]),
        "configuration_id": table.column("configuration_id").take(final),
        "dataset_group": table.column("dataset_group").take(final),
        "dataset_id": table.column("dataset_id").take(final),
    }, schema=OUT_SCHEMA)
    pq.write_table(out_table, f"{args.out_dir}/{name}.parquet")
    summary = {"bucket": name, "n_rows": len(cells), "n_cells": len(unique_cells),
               "load_seconds": round(t_load, 1), "fps_seconds": round(t_fps, 1),
               "problems": problems, "large_cell_timings": timings}
    with open(f"{args.out_dir}/{name}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(done_marker, "w") as fh:
        fh.write("done\n")
    print(f"{name}: DONE rows {len(cells):,} cells {len(unique_cells):,} "
          f"fps {t_fps:.0f}s problems {problems}", flush=True)


if __name__ == "__main__":
    main()
