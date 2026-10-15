import argparse
import glob
import hashlib
import json
import os

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
CHUNK = 2_000_000


def coreset_index(tier):
    """configuration_id -> row index, for the tier we are filling."""
    path = f"{RELEASE}/{tier}"
    ns = int(np.load(f"{path}/ns.npy")[0])
    cid = np.load(f"{path}/configuration_id.npy", mmap_mode="r")
    index = {}
    for start in range(0, ns, CHUNK):
        block = np.asarray(cid[start:min(start + CHUNK, ns)]).astype("U")
        for offset, value in enumerate(block):
            index[str(value)] = start + offset
    return ns, index


def harvest(group, index, found):
    """Collect source_reference for any collection row whose cid is in `index`."""
    path = f"{RELEASE}/collection/{group}"
    ns = int(np.load(f"{path}/ns.npy")[0])
    if not os.path.exists(f"{path}/source_reference_bytes.bin"):
        return 0
    cid = np.load(f"{path}/configuration_id.npy", mmap_mode="r")
    offsets = np.load(f"{path}/source_reference_offsets.npy", mmap_mode="r")
    blob = np.memmap(f"{path}/source_reference_bytes.bin", dtype=np.uint8, mode="r")

    hits = 0
    for start in range(0, ns, CHUNK):
        stop = min(start + CHUNK, ns)
        block = np.asarray(cid[start:stop]).astype("U")
        for offset, value in enumerate(block):
            row = index.get(str(value))
            if row is not None and row not in found:
                lo, hi = int(offsets[start + offset]), int(offsets[start + offset + 1])
                found[row] = bytes(blob[lo:hi])
                hits += 1
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ns, index = coreset_index(args.tier)
    groups = sorted(os.listdir(f"{RELEASE}/collection"))[args.task_id::args.num_tasks]

    found = {}
    for position, group in enumerate(groups):
        hits = harvest(group, index, found)
        if position % 20 == 0:
            print(f"task {args.task_id}: {position + 1}/{len(groups)} {group} "
                  f"(+{hits}, total {len(found):,})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = np.array(sorted(found), dtype=np.int64)
    payload = b"".join(found[int(r)] for r in rows)
    lengths = np.array([len(found[int(r)]) for r in rows], dtype=np.int64)
    np.savez(f"{args.out}/part_{args.task_id:04d}.npz",
             rows=rows, lengths=lengths, payload=np.frombuffer(payload, dtype=np.uint8))
    print(f"task {args.task_id}: matched {len(rows):,} of {ns:,}", flush=True)


if __name__ == "__main__":
    main()
