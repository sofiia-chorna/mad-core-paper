"""
Convert fractional positions to Cartesian in already-filtered ColabFit parquet
outputs.

For datasets that were processed with `--positions-format fractional`, the
distance-check used the converted positions but the WRITTEN output parquets
preserved the original fractional positions. This script rewrites those
parquets so positions are in Cartesian (Angstrom), making them consistent
with the rest of your filtered pipeline.

Use on the per-dataset output directories produced by run_filter_all.py with
--positions-format fractional. Both co_passed/ and co_rejected/ should be
converted.

Usage:
    # Single dataset directory
    python convert_fractional_to_cartesian.py \\
        filtered_fractional/colabfit_PWMLFF_feature_comparison_NPJ2023/co_passed/

    # All datasets matching a pattern (in place)
    python convert_fractional_to_cartesian.py \\
        filtered_fractional/colabfit_PWMLFF_*/co_passed/ \\
        filtered_fractional/colabfit_PWMLFF_*/co_rejected/

    # With safety: write to a sibling co_passed_cart/ dir instead of in place
    python convert_fractional_to_cartesian.py \\
        filtered_fractional/colabfit_PWMLFF_*/co_passed/ \\
        --out-suffix _cart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _to_3n_flat(positions, n_atoms: int) -> np.ndarray:
    """Coerce flat (3N,) or nested (N,3) to a flat (3N,) numpy array."""
    arr = np.asarray(positions, dtype=np.float64).reshape(-1)
    if arr.size != 3 * n_atoms:
        raise ValueError(
            f"positions has {arr.size} values but atomic_numbers has "
            f"{n_atoms} atoms (expected {3 * n_atoms})"
        )
    return arr


def _to_3x3_matrix(cell) -> np.ndarray:
    arr = np.asarray(cell, dtype=np.float64).reshape(-1)
    if arr.size != 9:
        raise ValueError(f"cell has {arr.size} values, expected 9")
    return arr.reshape(3, 3)


def _looks_fractional(positions, cell, n_atoms: int) -> tuple[bool, str]:
    """
    Heuristic: are positions in [0, 1] range while cell is much larger?
    Returns (looks_fractional, reason).
    """
    p = _to_3n_flat(positions, n_atoms).reshape(-1, 3)
    c = _to_3x3_matrix(cell)
    p_max = np.abs(p).max()
    cell_extent = np.linalg.norm(c, axis=1).max()
    if p_max <= 1.05 and cell_extent > 1.5:
        return True, f"positions in [-1,1] (max={p_max:.3f}), cell ~{cell_extent:.1f} A"
    return False, f"positions max={p_max:.3f} A, cell ~{cell_extent:.1f} A"


def convert_parquet_file(
    in_path: Path,
    out_path: Path,
    cols: dict | None = None,
    sanity_check: bool = True,
) -> dict:
    """
    Read a parquet file, convert positions from fractional to Cartesian,
    write to out_path. Preserves all other columns and the schema.
    """
    if cols is None:
        cols = {
            "positions": "positions",
            "cell": "cell",
            "numbers": "atomic_numbers",
        }

    table = pq.read_table(in_path)
    schema = table.schema
    pos_col_name = cols["positions"]
    cell_col_name = cols["cell"]
    num_col_name = cols["numbers"]

    if pos_col_name not in schema.names:
        raise KeyError(f"{in_path}: no '{pos_col_name}' column "
                       f"(have: {schema.names})")
    if cell_col_name not in schema.names:
        raise KeyError(f"{in_path}: no '{cell_col_name}' column")
    if num_col_name not in schema.names:
        raise KeyError(f"{in_path}: no '{num_col_name}' column")

    n_rows = table.num_rows
    if n_rows == 0:
        # Empty parquet: just copy through.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)
        return {"in": str(in_path), "out": str(out_path), "rows": 0,
                "converted": 0, "skipped": 0}

    # Work with pyarrow Python lists (avoid pandas, which doesn't handle
    # ragged list-typed columns well -- each row may have a different
    # natoms, so the positions column is a list of variable-length lists).
    pos_pylist  = table.column(pos_col_name).to_pylist()
    cell_pylist = table.column(cell_col_name).to_pylist()
    num_pylist  = table.column(num_col_name).to_pylist()

    # Sanity check on first row
    sample_pos  = pos_pylist[0]
    sample_cell = cell_pylist[0]
    sample_n    = len(num_pylist[0])
    if sanity_check:
        looks_frac, reason = _looks_fractional(sample_pos, sample_cell, sample_n)
        if not looks_frac:
            print(f"  WARN {in_path.name}: sample row doesn't look fractional "
                  f"({reason}). Skipping this file -- inspect manually.",
                  flush=True)
            return {"in": str(in_path), "out": None, "rows": n_rows,
                    "converted": 0, "skipped": n_rows,
                    "reason": "not_fractional"}

    # Detect flat (3N,) vs nested (N, 3) layout from the first row's shape
    sample_arr = np.asarray(sample_pos)
    is_flat = sample_arr.ndim == 1

    # Convert row-by-row
    new_positions = []
    for k in range(n_rows):
        natoms = len(num_pylist[k])
        p_flat = _to_3n_flat(pos_pylist[k], natoms)
        c = _to_3x3_matrix(cell_pylist[k])
        p_frac = p_flat.reshape(-1, 3)
        p_cart = p_frac @ c
        if is_flat:
            new_positions.append(p_cart.reshape(-1).tolist())
        else:
            new_positions.append(p_cart.tolist())

    # Replace just the positions column on the pyarrow Table, preserving
    # every other column and the original schema (column order, types,
    # metadata). Use the original field's type so the list-of-double or
    # large-list-of-double encoding is preserved exactly.
    pos_field = schema.field(pos_col_name)
    new_pos_array = pa.array(new_positions, type=pos_field.type)
    col_idx = schema.get_field_index(pos_col_name)
    out_table = table.set_column(col_idx, pos_field, new_pos_array)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_path, compression="snappy")
    return {"in": str(in_path), "out": str(out_path), "rows": n_rows,
            "converted": n_rows, "skipped": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Parquet files OR directories containing co_*.parquet")
    ap.add_argument("--out-suffix", default="",
                    help="If empty (default), overwrite files in place. "
                         "Otherwise write to a sibling directory with this "
                         "suffix (e.g. '_cart' -> co_passed_cart/).")
    ap.add_argument("--no-sanity-check", action="store_true",
                    help="Skip the heuristic that warns when positions don't "
                         "look fractional. Use if you know they ARE fractional "
                         "but the heuristic is too strict.")
    ap.add_argument("--col-positions", default="positions")
    ap.add_argument("--col-cell", default="cell")
    ap.add_argument("--col-numbers", default="atomic_numbers")
    args = ap.parse_args()

    cols = {"positions": args.col_positions,
            "cell":      args.col_cell,
            "numbers":   args.col_numbers}

    # Expand: each input may be a file or a directory
    files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.parquet")))
        elif p.is_file() and p.suffix == ".parquet":
            files.append(p)
        else:
            print(f"WARN: {inp} is neither a parquet file nor a directory; "
                  f"skipping", flush=True)

    if not files:
        print("No parquet files found.", flush=True)
        sys.exit(1)

    print(f"Found {len(files)} parquet files to convert", flush=True)
    if args.out_suffix:
        print(f"  output mode: sibling dir with suffix {args.out_suffix!r}",
              flush=True)
    else:
        print(f"  output mode: IN-PLACE (overwriting inputs)", flush=True)

    total_rows = 0
    total_converted = 0
    total_skipped = 0
    failed_files = 0
    for k, f in enumerate(files, start=1):
        # Determine output path
        if args.out_suffix:
            # e.g. .../colabfit_X/co_passed/co_0.parquet  ->
            #      .../colabfit_X/co_passed_cart/co_0.parquet
            new_parent = f.parent.with_name(f.parent.name + args.out_suffix)
            out_p = new_parent / f.name
        else:
            out_p = f

        print(f"[{k}/{len(files)}] {f}", flush=True)
        try:
            r = convert_parquet_file(f, out_p, cols=cols,
                                       sanity_check=not args.no_sanity_check)
            print(f"  -> rows={r['rows']} converted={r['converted']} "
                  f"skipped={r['skipped']}"
                  + (f" reason={r.get('reason')}" if r.get('reason') else ""),
                  flush=True)
            total_rows += r["rows"]
            total_converted += r["converted"]
            total_skipped += r["skipped"]
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            failed_files += 1

    print(f"\n=== Summary ===")
    print(f"  files processed:  {len(files)}")
    print(f"  files failed:     {failed_files}")
    print(f"  rows total:       {total_rows:,}")
    print(f"  rows converted:   {total_converted:,}")
    print(f"  rows skipped:     {total_skipped:,}")


if __name__ == "__main__":
    main()
