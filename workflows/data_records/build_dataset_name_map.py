import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
RELEASE = "${STORE_ROOT}/mad-core/release"


def dataset_ids_of(subfolder):
    """Every distinct dataset_id in the subfolder, counted by row.

    Must read all shards: sampling one row group misses subfolders that carry
    rows from more than one source dataset, which is exactly the case that makes
    a dataset_id -> name rename ambiguous.
    """
    paths = sorted(glob.glob(f"{subfolder}/*.parquet"))
    if not paths:
        paths = sorted(glob.glob(f"{subfolder}/**/*.parquet", recursive=True))

    counts = {}
    for path in paths:
        handle = pq.ParquetFile(path)
        if "dataset_id" not in handle.schema_arrow.names:
            return {}
        for group in range(handle.num_row_groups):
            column = handle.read_row_group(group, columns=["dataset_id"])
            values, n = np.unique(np.asarray(column.column("dataset_id")).astype("U"),
                                  return_counts=True)
            for value, k in zip(values, n):
                counts[str(value)] = counts.get(str(value), 0) + int(k)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    groups = sorted(os.listdir(STORE))[args.task_id::args.num_tasks]
    mapping = {}
    collisions = []

    for index, group in enumerate(groups):
        for subfolder in sorted(glob.glob(f"{STORE}/{group}/*")):
            if not os.path.isdir(subfolder):
                continue
            name = os.path.basename(subfolder)
            for dataset_id, rows in dataset_ids_of(subfolder).items():
                key = f"{group}\t{dataset_id}"
                if key in mapping and mapping[key][0] != name:
                    collisions.append([key, mapping[key], [name, rows]])
                mapping[key] = [name, rows]
        if index % 20 == 0:
            print(f"task {args.task_id}: {index + 1}/{len(groups)} groups", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/map_{args.task_id:04d}.json", "w") as fh:
        json.dump({"mapping": mapping, "collisions": collisions}, fh)
    print(f"task {args.task_id}: {len(mapping)} pairs, {len(collisions)} collisions",
          flush=True)


if __name__ == "__main__":
    main()
