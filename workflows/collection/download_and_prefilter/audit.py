#!/usr/bin/env python3
"""
Find discrepancies between ColabFit on HuggingFace and local downloads.

Usage:
    pip install huggingface_hub pyarrow tabulate

    python colabfit_audit.py ./colabfit_data
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
from huggingface_hub import HfApi


def get_remote_datasets():
    """Get all ColabFit datasets from HuggingFace with metadata."""
    api = HfApi()
    datasets = list(api.list_datasets(author="colabfit"))
    remote = {}
    for ds in datasets:
        # Normalize name: strip "colabfit/" prefix
        short_name = ds.id.replace("colabfit/", "")
        remote[short_name] = {
            "repo_id": ds.id,
            "last_modified": str(ds.last_modified) if ds.last_modified else None,
        }
    return remote


def count_local_configs(dataset_dir):
    """Count configs in a local dataset from parquet metadata (fast)."""
    co_dir = dataset_dir / "co"
    if not co_dir.exists():
        return None
    total = 0
    for pf in co_dir.glob("co_*.parquet"):
        try:
            meta = pq.read_metadata(pf)
            total += meta.num_rows
        except Exception:
            pass
    return total


def find_local_datasets(root_dir):
    """Find all local dataset dirs and count their configs."""
    root = Path(root_dir)
    local = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "co").exists():
            continue
        # Normalize name: strip common prefixes from HF download
        name = d.name
        if name.startswith("colabfit_"):
            name = name[len("colabfit_"):]
        n_configs = count_local_configs(d)
        local[name] = {
            "path": str(d),
            "n_configs": n_configs,
        }
    return local


def main():
    parser = argparse.ArgumentParser(
        description="Audit ColabFit downloads: find missing/incomplete datasets"
    )
    parser.add_argument("root_dir", help="Root directory of downloaded datasets")
    parser.add_argument("--csv", type=str, default=None, help="Export to CSV")
    args = parser.parse_args()

    print("Fetching remote dataset list from HuggingFace...")
    remote = get_remote_datasets()
    print(f"  Remote: {len(remote)} datasets on HuggingFace\n")

    print("Scanning local downloads...")
    local = find_local_datasets(args.root_dir)
    print(f"  Local:  {len(local)} datasets found\n")

    # Cross-reference
    remote_names = set(remote.keys())
    local_names = set(local.keys())

    missing = sorted(remote_names - local_names)
    extra = sorted(local_names - remote_names)
    common = sorted(remote_names & local_names)

    # Report missing
    if missing:
        print(f"MISSING from local ({len(missing)} datasets):")
        print("-" * 50)
        for name in missing:
            print(f"  {name}")
        print()

    # Report extra (local but not on HF -- unlikely but possible)
    if extra:
        print(f"LOCAL ONLY - not on HuggingFace ({len(extra)} datasets):")
        print("-" * 50)
        for name in extra:
            n = local[name]["n_configs"]
            print(f"  {name}: {n:,} configs" if n else f"  {name}: unknown")
        print()

    # Summary of what you have
    total_local = sum(
        local[name]["n_configs"]
        for name in local
        if local[name]["n_configs"] is not None
    )
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Remote datasets:  {len(remote)}")
    print(f"  Local datasets:   {len(local)}")
    print(f"  Missing:          {len(missing)}")
    print(f"  Local configs:    {total_local:,}")
    print(f"{'='*60}")

    # Fuzzy match: check if missing datasets have close local names
    if missing:
        print(f"\nTo download missing datasets:")
        print(f"  python download_colabfit.py --download-all --output {args.root_dir} \\")
        # Group missing by likely family
        print(f"    --filter {','.join(missing[:5])}...")
        print(f"\n  Or download individually:")
        for name in missing[:10]:
            print(f"    python download_colabfit.py "
                  f"--download colabfit/{name} --output {args.root_dir}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    # CSV export
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "status", "local_configs", "local_path"])
            for name in sorted(remote_names | local_names):
                if name in missing:
                    writer.writerow([name, "MISSING", "", ""])
                elif name in extra:
                    n = local[name]["n_configs"] or ""
                    writer.writerow([name, "LOCAL_ONLY", n, local[name]["path"]])
                else:
                    n = local[name]["n_configs"] or ""
                    writer.writerow([name, "OK", n, local[name]["path"]])
        print(f"\nExported to {args.csv}")


if __name__ == "__main__":
    main()
