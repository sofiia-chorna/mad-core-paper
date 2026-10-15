"""
Verify that two filtered-dataset directories contain matching rows.

Walks both `folder_A` and `folder_B` looking for parallel dataset subdirs
(e.g. colabfit_*/co_passed/). For each matched dataset, samples N random
configuration IDs that exist in both, fetches the corresponding row from
each side, and reports which columns differ. Helpful for verifying that:

  - convert_fractional_to_cartesian.py only changed `positions` and left
    everything else byte-identical (expected: positions differ, all other
    columns identical),
  - a re-run of run_filter_all.py reproduces an earlier run (expected:
    all columns identical),
  - two pipeline variants agree where they should.

Usage:
    # Spot-check all colabfit_* datasets, 15 ids each
    python check_dataset_consistency.py filtered_A/ filtered_B/

    # Custom N samples, alternate id column, pattern, subdir
    python check_dataset_consistency.py filtered/ filtered_cart/ \\
        --n 25 --id-column extended_id \\
        --pattern 'colabfit_PWMLFF_*' --subdir co_passed

    # Compare RAW input (co/) against FILTERED output (co_passed/).
    # The filter never modifies row data, so every kept row must be
    # byte-identical to its source in the raw input.
    python check_dataset_consistency.py filtered/ colabfit_data/ \\
        --subdir-a co_passed --subdir-b co
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _row_to_dict(table, row_idx: int) -> dict:
    """Extract one row from a pyarrow Table as a dict of native Python values."""
    return {col: table.column(col)[row_idx].as_py() for col in table.schema.names}


def _values_equal(va, vb, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    """
    Compare two row values robustly. Handles:
      - None
      - scalars (int, float, str, bool)
      - flat lists of numbers (e.g. positions, cell, forces)
      - nested lists (e.g. (N,3) positions when stored nested)
    Uses absolute+relative tolerance for floats; exact match for everything else.
    """
    if va is None or vb is None:
        return va is None and vb is None

    # Try numeric array path first
    try:
        a = np.asarray(va)
        b = np.asarray(vb)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "iuf" and b.dtype.kind in "iuf":
            return np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        if a.dtype.kind == "b" and b.dtype.kind == "b":
            return bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        pass

    # Fallback: exact equality (strings, bools, mixed lists)
    return va == vb


def _load_id_to_position(dataset_dir: Path, id_col: str):
    """
    Build a map: id_value -> (parquet_path, row_offset_within_file).
    Reads only the id column from each shard for speed.
    """
    parquets = sorted(p for p in dataset_dir.glob("*.parquet") if p.is_file())
    if not parquets:
        return {}, []
    id_index: dict = {}
    for pq_path in parquets:
        try:
            tbl = pq.read_table(pq_path, columns=[id_col])
        except Exception as e:
            print(f"  WARN cannot read {pq_path.name}: {e}", flush=True)
            continue
        ids = tbl.column(id_col).to_pylist()
        for k, v in enumerate(ids):
            if v in id_index:
                continue   # silently keep first occurrence on duplicates
            id_index[v] = (pq_path, k)
    return id_index, parquets


def check_one_dataset(
    dataset_a: Path, dataset_b: Path, id_col: str, n_samples: int,
    rng: random.Random, ignore_columns: set,
) -> dict:
    """Spot-check a single matched dataset directory pair."""
    idx_a, _ = _load_id_to_position(dataset_a, id_col)
    idx_b, _ = _load_id_to_position(dataset_b, id_col)

    if not idx_a or not idx_b:
        return {"status": "empty", "name": dataset_a.parent.name,
                "n_a": len(idx_a), "n_b": len(idx_b)}

    common = set(idx_a.keys()) & set(idx_b.keys())
    only_a = set(idx_a.keys()) - set(idx_b.keys())
    only_b = set(idx_b.keys()) - set(idx_a.keys())
    print(common, only_a, only_b)
    if not common:
        return {"status": "no_overlap", "name": dataset_a.parent.name,
                "n_a": len(idx_a), "n_b": len(idx_b)}

    k = min(n_samples, len(common))
    sampled = rng.sample(sorted(common), k=k)

    # Cache: parquet_path -> read table (skip re-reading the same shard)
    cache: dict = {}
    def get_row(side_idx, id_val):
        path, row = side_idx[id_val]
        if path not in cache:
            cache[path] = pq.read_table(path)
        return _row_to_dict(cache[path], row)

    differ_by_col: dict = {}   # col_name -> count of rows where this col differed
    all_match = True
    mismatch_examples: list = []

    for id_val in sampled:
        row_a = get_row(idx_a, id_val)
        row_b = get_row(idx_b, id_val)

        # Schema mismatch -> note it once
        cols_a = set(row_a.keys()) - ignore_columns
        cols_b = set(row_b.keys()) - ignore_columns
        if cols_a != cols_b:
            differ_by_col.setdefault("_schema", 0)
            differ_by_col["_schema"] += 1
            all_match = False
            mismatch_examples.append({
                "id": id_val,
                "only_in_a": sorted(cols_a - cols_b),
                "only_in_b": sorted(cols_b - cols_a),
            })
            continue

        cols_to_compare = cols_a   # equal sets at this point
        row_diffs = []
        for c in cols_to_compare:
            if not _values_equal(row_a[c], row_b[c]):
                differ_by_col[c] = differ_by_col.get(c, 0) + 1
                row_diffs.append(c)
                all_match = False
        if row_diffs:
            mismatch_examples.append({"id": id_val, "differs_in": row_diffs})

    return {
        "status": "ok",
        "name": dataset_a.parent.name,
        "n_a": len(idx_a),
        "n_b": len(idx_b),
        "n_common": len(common),
        "n_sampled": k,
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "all_match": all_match,
        "differ_by_col": differ_by_col,
        "examples": mismatch_examples[:3],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder_a", help="First top-level directory (e.g. filtered/).")
    ap.add_argument("folder_b", help="Second top-level directory (e.g. filtered_cart/).")
    ap.add_argument("--pattern", default="*",
                    help="Glob over dataset subdirectories (default '*'). "
                         "E.g. 'colabfit_PWMLFF_*' restricts to PWMLFF datasets.")
    ap.add_argument("--subdir", default=None,
                    help="Subdirectory inside each dataset to compare on BOTH "
                         "sides (default 'co_passed'). Use '.' if parquets "
                         "are at the dataset root. Equivalent to setting "
                         "--subdir-a and --subdir-b to the same value.")
    ap.add_argument("--subdir-a", default=None,
                    help="Override subdir for folder_a only (e.g. 'co' when "
                         "comparing raw input vs filtered output).")
    ap.add_argument("--subdir-b", default=None,
                    help="Override subdir for folder_b only (e.g. 'co_passed').")
    ap.add_argument("--id-column", default="configuration_id",
                    help="Column to use as the unique row id (default "
                         "'configuration_id'). Use 'extended_id' or another "
                         "unique string column if your schema differs.")
    ap.add_argument("--n", type=int, default=15,
                    help="Random ids to sample per dataset (default 15).")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (default 0 -> reproducible sampling).")
    ap.add_argument("--ignore-columns", nargs="*", default=[],
                    help="Columns to skip in the comparison. Useful for the "
                         "fractional->Cartesian check: --ignore-columns positions")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    if not folder_a.is_dir() or not folder_b.is_dir():
        print(f"both inputs must be directories", flush=True)
        sys.exit(1)

    # Resolve per-side subdirs from three CLI flags:
    #   --subdir X            -> both sides use X
    #   --subdir-a X --subdir-b Y -> sides use X and Y respectively
    #   (neither set)         -> default 'co_passed' on both
    subdir_default = "co_passed"
    subdir_a = args.subdir_a or args.subdir or subdir_default
    subdir_b = args.subdir_b or args.subdir or subdir_default
    if subdir_a != subdir_b:
        print(f"Comparing folder_a/<ds>/{subdir_a}/ vs "
              f"folder_b/<ds>/{subdir_b}/", flush=True)

    ignore_columns = set(args.ignore_columns)

    # Walk folder_a for matching dataset subdirs, find paired subdirs in B.
    datasets_a = sorted(d for d in folder_a.glob(args.pattern) if d.is_dir())
    if not datasets_a:
        print(f"no matches for pattern {args.pattern!r} in {folder_a}",
              flush=True)
        sys.exit(1)

    print(f"Found {len(datasets_a)} datasets in {folder_a} matching "
          f"{args.pattern!r}", flush=True)
    if ignore_columns:
        print(f"Ignoring columns: {sorted(ignore_columns)}", flush=True)
    print()

    n_ok = n_mismatch = n_missing = n_empty = 0
    all_results: list = []

    for ds_a in datasets_a:
        ds_b = folder_b / ds_a.name
        if not ds_b.is_dir():
            print(f"[MISSING B] {ds_a.name}", flush=True)
            n_missing += 1
            continue

        sub_a = ds_a / subdir_a if subdir_a != "." else ds_a
        sub_b = ds_b / subdir_b if subdir_b != "." else ds_b
        if not sub_a.is_dir() or not sub_b.is_dir():
            missing_side = []
            if not sub_a.is_dir():
                missing_side.append(f"a:{subdir_a}")
            if not sub_b.is_dir():
                missing_side.append(f"b:{subdir_b}")
            print(f"[NO SUBDIR] {ds_a.name}  ({', '.join(missing_side)})",
                  flush=True)
            n_missing += 1
            continue

        r = check_one_dataset(sub_a, sub_b, args.id_column, args.n, rng,
                              ignore_columns)
        all_results.append(r)

        if r["status"] == "empty":
            print(f"[EMPTY]    {r['name']}  (a={r['n_a']} b={r['n_b']})",
                  flush=True)
            n_empty += 1
        elif r["status"] == "no_overlap":
            print(f"[NO MATCH] {r['name']}  (a={r['n_a']} b={r['n_b']} "
                  f"no shared ids)", flush=True)
            n_mismatch += 1
        elif r["all_match"]:
            print(f"[OK]       {r['name']}  "
                  f"({r['n_sampled']}/{r['n_common']} sampled, all identical)",
                  flush=True)
            n_ok += 1
        else:
            print(f"[DIFFER]   {r['name']}  "
                  f"({r['n_sampled']} sampled)", flush=True)
            for col, n in sorted(r["differ_by_col"].items(),
                                  key=lambda x: -x[1]):
                print(f"             differs in {col!r}: {n}/{r['n_sampled']} rows",
                      flush=True)
            for ex in r["examples"]:
                print(f"             example id={ex['id']!r}: "
                      f"{ex.get('differs_in', ex)}", flush=True)
            n_mismatch += 1

    # Summary
    print(f"\n=== Summary ===")
    print(f"  datasets checked: {len(all_results)}")
    print(f"  identical:        {n_ok}")
    print(f"  differ:           {n_mismatch}")
    print(f"  missing/empty:    {n_missing + n_empty}")
    if n_mismatch:
        sys.exit(2)


if __name__ == "__main__":
    main()
