#!/usr/bin/env python3
"""
Bulk download ColabFit datasets from HuggingFace.

Usage:
    # 1. Install dependencies
    pip install huggingface_hub

    # 2. List all available datasets (dry run)
    python download_colabfit.py --list

    # 3. Download everything (WARNING: very large, 100s of GB+)
    python download_colabfit.py --download-all --output ./colabfit_data

    # 4. Download only specific families (recommended)
    python download_colabfit.py --download-all --output ./colabfit_data \
        --filter Alexandria,MAD,OMat24,MPTrj

    # 5. Download a single dataset by full name
    python download_colabfit.py --download colabfit/Alexandria_PBE_set1 \
        --output ./colabfit_data
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from huggingface_hub import HfApi, snapshot_download

print_lock = Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def list_datasets(filter_str=None, exclude_str=None):
    """List all colabfit datasets, optionally filtering/excluding by substring."""
    api = HfApi()
    datasets = list(api.list_datasets(author="colabfit"))
    if filter_str:
        keywords = [k.strip().lower() for k in filter_str.split(",")]
        datasets = [
            ds for ds in datasets
            if any(k in ds.id.lower() for k in keywords)
        ]
    if exclude_str:
        excluded = [k.strip().lower() for k in exclude_str.split(",")]
        datasets = [
            ds for ds in datasets
            if not any(k in ds.id.lower() for k in excluded)
        ]
    datasets.sort(key=lambda d: d.id)
    return datasets


def download_dataset(repo_id, output_dir, max_retries=3, label=""):
    """Download a single dataset with retries."""
    local_dir = Path(output_dir) / repo_id.replace("/", "_")
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(local_dir),
                resume_download=True,
            )
            log(f"  {label} OK: {repo_id}")
            return (repo_id, True)
        except Exception as e:
            log(f"  {label} attempt {attempt + 1}/{max_retries} failed for {repo_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    log(f"  {label} FAILED: {repo_id}")
    return (repo_id, False)


def main():
    parser = argparse.ArgumentParser(
        description="Download ColabFit datasets from HuggingFace"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--download", type=str, help="Download a single dataset by repo_id")
    parser.add_argument("--download-all", action="store_true", help="Download all datasets")
    parser.add_argument("--output", type=str, default="./colabfit_data", help="Output directory")
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Comma-separated substrings to INCLUDE (e.g. 'Alexandria,MAD,OMat24')"
    )
    parser.add_argument(
        "--exclude", type=str, default=None,
        help="Comma-separated substrings to EXCLUDE (e.g. 'Alexandria,OMol25')"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download workers (default: 4)"
    )
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if args.list:
        datasets = list_datasets(args.filter, args.exclude)
        print(f"Found {len(datasets)} datasets:\n")
        for ds in datasets:
            print(f"  {ds.id}")
        return

    if args.download:
        print(f"Downloading {args.download} -> {args.output}")
        _, ok = download_dataset(args.download, args.output, args.max_retries)
        sys.exit(0 if ok else 1)

    if args.download_all:
        datasets = list_datasets(args.filter, args.exclude)
        n = len(datasets)
        w = args.workers
        print(f"Downloading {n} datasets to {args.output} with {w} workers\n")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        failed = []
        done = 0
        with ThreadPoolExecutor(max_workers=w) as pool:
            futures = {
                pool.submit(
                    download_dataset, ds.id, args.output, args.max_retries,
                    label=f"[{i}/{n}]"
                ): ds.id
                for i, ds in enumerate(datasets, 1)
            }
            for fut in as_completed(futures):
                repo_id, ok = fut.result()
                done += 1
                if not ok:
                    failed.append(repo_id)
                if done % 10 == 0 or done == n:
                    log(f"Progress: {done}/{n} done, {len(failed)} failed")

        print(f"\nDone. {n - len(failed)}/{n} succeeded.")
        if failed:
            print("Failed datasets:")
            for f in sorted(failed):
                print(f"  {f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
