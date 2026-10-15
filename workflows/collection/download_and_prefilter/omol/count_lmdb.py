"""
Count structures across a set of FAIRChem-style ASE LMDB (.aselmdb) files.

Each .aselmdb file is opened via ase.db.connect() and queried for its row
count. Counts are gathered in parallel across files (I/O-bound, threads
are fine). Outputs a per-file count if requested and a grand total.

Usage:
    # Count all .aselmdb files under one directory
    python count_lmdb.py omol/train/

    # Multiple inputs (dirs, files, globs all accepted)
    python count_lmdb.py omol/train/ omol/val/ omol/test/

    # Show per-file count alongside the total
    python count_lmdb.py omol/train/ --per-file

    # Recurse into subdirs
    python count_lmdb.py omol/ --recursive

    # Adjust worker count (default: min(16, #files))
    python count_lmdb.py omol/train/ --workers 32
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ase.db import connect


def _discover(inputs: list[str], recursive: bool) -> list[Path]:
    """Expand a mix of files / dirs / globs to a deduplicated sorted file list."""
    found: list[Path] = []
    seen: set[Path] = set()
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            it = p.rglob("*.aselmdb") if recursive else p.glob("*.aselmdb")
            for f in it:
                if f.is_file() and f not in seen:
                    found.append(f)
                    seen.add(f)
        elif p.is_file() and p.suffix == ".aselmdb":
            if p not in seen:
                found.append(p)
                seen.add(p)
        else:
            # Try glob expansion from cwd
            for f in sorted(Path(".").glob(inp)):
                if f.is_file() and f.suffix == ".aselmdb" and f not in seen:
                    found.append(f)
                    seen.add(f)
    return sorted(found)


def _count_one(path: Path) -> tuple[Path, int, str | None]:
    """Open one LMDB and return (path, count, error). Doesn't raise."""
    try:
        db = connect(str(path))
        # ase.db.connect doesn't expose a uniform .count() for every backend,
        # but len(db) works on the ASE LMDB backend used by FAIRChem.
        n = len(db)
        return path, int(n), None
    except Exception as e:
        return path, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help=".aselmdb files, directories, or globs.")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="Recurse into subdirectories when given a directory.")
    ap.add_argument("--per-file", action="store_true",
                    help="Print one line per file with its row count.")
    ap.add_argument("--workers", type=int, default=None,
                    help="Concurrent file readers (default: min(16, #files)).")
    ap.add_argument("--sort", choices=["name", "count", "size"], default="name",
                    help="How to sort the per-file listing (default: name).")
    args = ap.parse_args()

    files = _discover(args.inputs, recursive=args.recursive)
    if not files:
        print("No .aselmdb files found.", file=sys.stderr)
        sys.exit(1)

    n_workers = args.workers or min(16, len(files))
    print(f"Counting structures in {len(files):,} .aselmdb files "
          f"using {n_workers} workers...", flush=True)

    t0 = time.time()
    results: list[tuple[Path, int, str | None]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_count_one, f): f for f in files}
        for fut in as_completed(futs):
            results.append(fut.result())

    # Sort for display
    if args.sort == "count":
        results.sort(key=lambda r: -r[1])
    elif args.sort == "size":
        results.sort(key=lambda r: -r[0].stat().st_size if r[0].exists() else 0)
    else:
        results.sort(key=lambda r: str(r[0]))

    total = 0
    n_errors = 0
    if args.per_file:
        print()
        print(f"{'rows':>12s}  {'size':>8s}  file")
        for path, count, err in results:
            if err:
                print(f"{'ERR':>12s}  {'-':>8s}  {path}  ({err})")
                n_errors += 1
                continue
            sz = path.stat().st_size / 1e9
            print(f"{count:>12,d}  {sz:>6.2f} GB  {path}")
            total += count
    else:
        for _, count, err in results:
            if err is None:
                total += count
            else:
                n_errors += 1

    elapsed = time.time() - t0

    print()
    print(f"{'=' * 60}")
    print(f"Files:  {len(files):,}  ({n_errors} unreadable)")
    print(f"Total:  {total:,} structures")
    print(f"Time:   {elapsed:.1f}s ({total / max(elapsed, 1e-9):.0f} reads/s)")

    if n_errors > 0 and not args.per_file:
        print()
        print(f"!! {n_errors} files failed -- rerun with --per-file to see which")
        sys.exit(2)


if __name__ == "__main__":
    main()
