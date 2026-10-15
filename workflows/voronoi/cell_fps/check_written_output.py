import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def check_file(path, check_ids):
    columns = ["cell", "cell_rank", "eps_sq"] + (["configuration_id"] if check_ids else [])
    table = pq.read_table(path, columns=columns)
    cells = table.column("cell").to_numpy()
    ranks = table.column("cell_rank").to_numpy()
    eps = table.column("eps_sq").to_numpy()

    problems = []
    if not np.all(np.diff(cells) >= 0):
        problems.append("rows not grouped by cell")

    boundaries = np.concatenate([[0], np.where(np.diff(cells) != 0)[0] + 1, [len(cells)]])
    eps_breaks = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        n = hi - lo
        cell_ranks = ranks[lo:hi]
        if cell_ranks[0] != 0 or not np.array_equal(cell_ranks, np.arange(n)):
            problems.append(f"cell {cells[lo]}: ranks are not 0..{n - 1} in order")
            continue
        cell_eps = eps[lo:hi]
        if n > 2:
            rises = np.where(np.diff(cell_eps[1:])
                             > 1e-5 * np.abs(cell_eps[1:-1]) + 1e-7)[0]
            eps_breaks += len(rises)

    duplicate_ids = 0
    if check_ids:
        cid = np.asarray(table.column("configuration_id").to_pandas(), dtype=object)
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if hi - lo > 1 and len(np.unique(cid[lo:hi])) != hi - lo:
                duplicate_ids += 1

    return {"rows": len(cells), "cells": len(boundaries) - 1,
            "eps_rises": eps_breaks, "cells_with_duplicate_ids": duplicate_ids,
            "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps-dir", required=True)
    parser.add_argument("--pattern", default="cell_*.parquet")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    total_rows = total_cells = total_rises = total_dupes = 0
    failures = []
    for path in sorted(glob.glob(f"{args.fps_dir}/{args.pattern}")):
        result = check_file(path, args.check_ids)
        total_rows += result["rows"]
        total_cells += result["cells"]
        total_rises += result["eps_rises"]
        total_dupes += result["cells_with_duplicate_ids"]
        if result["problems"]:
            failures.append((os.path.basename(path), result["problems"]))
        print(f"{os.path.basename(path):<28} rows {result['rows']:>12,} "
              f"cells {result['cells']:>9,} eps_rises {result['eps_rises']:>3} "
              f"dup_id_cells {result['cells_with_duplicate_ids']}", flush=True)

    print(f"\nTOTAL rows {total_rows:,} cells {total_cells:,} "
          f"eps_rises {total_rises} cells_with_duplicate_ids {total_dupes}")
    if failures:
        print(f"STRUCTURAL FAILURES in {len(failures)} files:")
        for name, problems in failures[:10]:
            print(f"  {name}: {problems[:3]}")
    else:
        print("structure OK: every cell's ranks are exactly 0..n-1 in row order")


if __name__ == "__main__":
    main()
