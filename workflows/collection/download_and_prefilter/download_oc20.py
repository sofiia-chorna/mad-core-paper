#!/usr/bin/env python3

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from huggingface_hub import snapshot_download

# explicit allowlist. Bare names the colabfit/ prefix is added below
OC20_DATASETS = [
    # IS2RES validation
    "OC20_IS2RES_val_id",
    "OC20_IS2RES_val_ood_ads",
    "OC20_IS2RES_val_ood_both",
    "OC20_IS2RES_val_ood_cat",
    # S2EF training (huge)
    "OC20_S2EF_train_all",
    # S2EF validation
    "OC20_S2EF_val_id",
    "OC20_S2EF_val_ood_ads",
    "OC20_S2EF_val_ood_both",
    "OC20_S2EF_val_ood_cat",
]

REPO_PREFIX = "colabfit/"
TRAIN_DATASETS = {"OC20_S2EF_train_all"}

print_lock = Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def select_datasets(only=None, skip_train=False):
    names = list(OC20_DATASETS)
    if skip_train:
        names = [n for n in names if n not in TRAIN_DATASETS]
    if only:
        keywords = [k.strip().lower() for k in only.split(",")]
        names = [n for n in names if any(k in n.lower() for k in keywords)]
    return [REPO_PREFIX + n for n in names]


def download_dataset(repo_id, output_dir, max_retries=3, label=""):
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
        description="Download OC20 datasets from ColabFit on HuggingFace"
    )
    parser.add_argument("--list", action="store_true", help="List datasets that would be downloaded")
    parser.add_argument("--output", type=str, default="./oc20_data", help="Output directory")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated substrings to restrict the OC20 allowlist "
             "(e.g. 'IS2RES' or 'val_id,val_ood_ads')"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip OC20_S2EF_train_all (the ~63M structure training set)"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Parallel download workers (default: 2; OC20 files are large, "
             "don't crank this without bandwidth)"
    )
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    repo_ids = select_datasets(only=args.only, skip_train=args.skip_train)

    if not repo_ids:
        print("No datasets selected. Check --only filter.", file=sys.stderr)
        sys.exit(2)

    if args.list:
        print(f"Would download {len(repo_ids)} datasets:")
        for r in repo_ids:
            marker = "  [TRAIN, large]" if r.split("/", 1)[1] in TRAIN_DATASETS else ""
            print(f"  {r}{marker}")
        return

    n = len(repo_ids)
    w = args.workers
    print(f"Downloading {n} datasets to {args.output} with {w} workers\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=w) as pool:
        futures = {
            pool.submit(
                download_dataset, repo_id, args.output, args.max_retries,
                f"[{i}/{n}]"
            ): repo_id
            for i, repo_id in enumerate(repo_ids, 1)
        }
        for fut in as_completed(futures):
            repo_id, ok = fut.result()
            done += 1
            if not ok:
                failed.append(repo_id)
            log(f"Progress: {done}/{n} done, {len(failed)} failed")

    print(f"\nDone. {n - len(failed)}/{n} succeeded.")
    if failed:
        print("Failed datasets:")
        for f in sorted(failed):
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
