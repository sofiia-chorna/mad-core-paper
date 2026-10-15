#!/usr/bin/env python3
"""
Run the Pyykko covalent-radius filter on every ColabFit dataset under a
parent directory. Each dataset has the standard layout:

    colabfit_<name>/
        ds.parquet              # dataset metadata
        co/co_*.parquet         # configurations  <- this is what we filter
        po/po_*.parquet         # property_objects (optional)

Output mirrors the input layout under --out-root:

    out_root/
        colabfit_<name>/
            co_passed/co_*.parquet
            co_rejected/co_*.parquet   (only when there are rejections)
            stats.json

A combined audit CSV at out_root/audit.csv records per-dataset pass/reject
counts and top rejection causes. Idempotent: datasets with an existing
stats.json are skipped on rerun.

Usage:
    python run_filter_all.py colabfit_data/ out_filtered/ --workers 16

The --workers flag is for *within-dataset* parallelism (one process per
co/ shard inside each dataset). Datasets are processed sequentially in
the outer loop because most of them have only a few shards; if you have
hundreds of single-shard datasets you'd want outer-loop parallelism
instead, but that's a different driver.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

# Reuse our existing modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyykko_filter_parquet import filter_parquet, filter_parquet_parallel


def find_datasets(root: Path, pattern: str = "colabfit_*") -> list[Path]:
    """Find every <root>/<pattern>/co subdirectory."""
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not d.match(pattern):
            continue
        co_dir = d / "co"
        if co_dir.is_dir():
            out.append(d)
    return out


def already_done(out_dataset_dir: Path) -> bool:
    """A dataset is considered done if its stats.json exists."""
    return (out_dataset_dir / "stats.json").exists()


def filter_one_dataset(
    in_ds_dir: Path,
    out_ds_dir: Path,
    scale: float,
    workers: int,
    backend: str,
) -> dict:
    """Filter a single dataset directory. Returns a stats dict."""
    co_dir   = in_ds_dir / "co"
    out_pass = out_ds_dir / "co_passed"
    out_rej  = out_ds_dir / "co_rejected"
    out_ds_dir.mkdir(parents=True, exist_ok=True)

    if workers > 1:
        stats = filter_parquet_parallel(
            co_dir, str(out_pass),
            rejected_dir=str(out_rej),
            n_workers=workers,
            scale=scale,
            backend=backend,
        )
        # per_shard contains tuples (shard_path, stats); strip down for JSON.
        stats["per_shard"] = [
            {"shard": Path(p).name,
             "passed": s["passed"],
             "rejected": s["rejected"]}
            for p, s in stats["per_shard"]
        ]
    else:
        out_pass.mkdir(parents=True, exist_ok=True)
        # Single output file when serial.
        out_file = out_pass / "passed.parquet"
        rej_file = out_rej / "rejected.parquet"
        out_rej.mkdir(parents=True, exist_ok=True)
        stats = filter_parquet(
            co_dir, str(out_file),
            rejected_path=str(rej_file),
            scale=scale,
            backend=backend,
            quiet=True,
        )

    # Convert Counter to plain dict for JSON.
    stats["rejection_pairs"] = {
        f"{a}-{b}": c for (a, b), c in stats["rejection_pairs"].most_common()
    }
    return stats


def write_audit_csv(audit_path: Path, rows: list[dict]) -> None:
    """Write/overwrite a CSV summary of all processed datasets."""
    if not rows:
        return
    fields = ["dataset", "status", "total", "passed", "rejected",
              "reject_pct", "top_pairs", "elapsed_s", "error"]
    with audit_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_existing_audit(audit_path: Path) -> list[dict]:
    """Read existing audit rows so that a rerun preserves prior dataset entries."""
    if not audit_path.exists():
        return []
    rows = []
    with audit_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Run the covalent-radius filter on every ColabFit dataset "
                    "under a parent directory."
    )
    p.add_argument("in_root",  help="Parent directory containing colabfit_* subdirs.")
    p.add_argument("out_root", help="Output directory; mirrors input layout.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor (default 0.5).")
    p.add_argument("--workers", type=int, default=4,
                   help="Worker processes per dataset (default 4). Workers > 1 "
                        "writes one parquet per shard.")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto")
    p.add_argument("--pattern", default="colabfit_*",
                   help="Glob pattern for dataset subdirectories.")
    p.add_argument("--only", nargs="+", default=None,
                   help="Optional: limit to dataset names matching any of "
                        "these substrings (e.g. --only OMat24 Alex).")
    p.add_argument("--exclude", nargs="+", default=None,
                   help="Optional: skip datasets whose names contain any of "
                        "these substrings.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess datasets even if stats.json already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="List datasets that would be processed and exit.")
    args = p.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    audit_path = out_root / "audit.csv"

    datasets = find_datasets(in_root, pattern=args.pattern)
    if args.only:
        datasets = [d for d in datasets
                    if any(o in d.name for o in args.only)]
    if args.exclude:
        datasets = [d for d in datasets
                    if not any(e in d.name for e in args.exclude)]

    print(f"Found {len(datasets)} datasets in {in_root}", flush=True)
    if not datasets:
        sys.exit(0)

    if args.dry_run:
        for d in datasets:
            done = already_done(out_root / d.name)
            tag = "[skip]" if done else "[do  ]"
            print(f"  {tag} {d.name}")
        sys.exit(0)

    # Load existing audit so reruns preserve earlier entries for skipped sets.
    existing = {r["dataset"]: r for r in load_existing_audit(audit_path)}
    audit_rows: list[dict] = list(existing.values())

    t_overall = time.time()
    n_done = n_skip = n_fail = 0

    for k, ds_dir in enumerate(datasets, start=1):
        out_ds_dir = out_root / ds_dir.name
        if not args.force and already_done(out_ds_dir):
            print(f"[{k}/{len(datasets)}] {ds_dir.name}: skip (already done)",
                  flush=True)
            n_skip += 1
            continue

        print(f"\n[{k}/{len(datasets)}] {ds_dir.name}", flush=True)
        t0 = time.time()
        try:
            stats = filter_one_dataset(
                ds_dir, out_ds_dir,
                scale=args.scale,
                workers=args.workers,
                backend=args.backend,
            )
            elapsed = time.time() - t0
            stats["elapsed_s"] = round(elapsed, 2)
            stats["scale"] = args.scale

            # Persist per-dataset stats.
            with (out_ds_dir / "stats.json").open("w") as f:
                json.dump(stats, f, indent=2)

            # Append to in-memory audit list (overwriting any earlier entry).
            top = list(stats["rejection_pairs"].items())[:3]
            top_str = "; ".join(f"{p}:{c}" for p, c in top)
            pct = 100 * stats["rejected"] / max(stats["total"], 1)
            existing[ds_dir.name] = {
                "dataset":    ds_dir.name,
                "status":     "OK",
                "total":      stats["total"],
                "passed":     stats["passed"],
                "rejected":   stats["rejected"],
                "reject_pct": f"{pct:.3f}",
                "top_pairs":  top_str,
                "elapsed_s":  f"{elapsed:.1f}",
                "error":      "",
            }
            print(f"  -> total={stats['total']:,} passed={stats['passed']:,} "
                  f"rejected={stats['rejected']:,} ({pct:.3f}%)  "
                  f"top: {top_str or '-'}  ({elapsed:.1f}s)", flush=True)
            n_done += 1

        except Exception as e:
            elapsed = time.time() - t0
            err = f"{type(e).__name__}: {e}"
            print(f"  FAIL: {err}", flush=True)
            traceback.print_exc()
            existing[ds_dir.name] = {
                "dataset":    ds_dir.name,
                "status":     "FAIL",
                "total":      "",
                "passed":     "",
                "rejected":   "",
                "reject_pct": "",
                "top_pairs":  "",
                "elapsed_s":  f"{elapsed:.1f}",
                "error":      err,
            }
            n_fail += 1

        # Re-write audit after every dataset so progress is durable.
        write_audit_csv(audit_path, list(existing.values()))

    dt = time.time() - t_overall
    print(f"\n{'='*60}")
    print(f"Processed: {n_done} done, {n_skip} skipped, {n_fail} failed "
          f"out of {len(datasets)} datasets")
    print(f"Total wall time: {dt/60:.1f} min")
    print(f"Audit CSV: {audit_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
