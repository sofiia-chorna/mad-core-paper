#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from coresets.cuda.multi.sharded import accrete_sharded

FEATURE_COLUMN = "phace_features_bb_mean"
FEATURE_DIM = 512


def get_rank_and_world():
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

    return 0, 1


def init_distributed(rank, world_size):
    if "MASTER_ADDR" not in os.environ:
        node = os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]

        if "[" in node:
            base, suffix = node.split("[", 1)
            node = base + suffix.split("-")[0].split(",")[0].rstrip("]")
        os.environ["MASTER_ADDR"] = node

    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=3))


def log(message, rank=0):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_scaler(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return np.asarray(obj["mean_"], dtype=np.float32), float(obj["scale_"])
    return np.asarray(obj.mean_, dtype=np.float32), float(obj.scale_)


def build_blocks(manifest_path, anchor_features, anchor_rows):
    blocks = [dict(path=anchor_features, rows=anchor_rows)]

    for line in open(manifest_path):
        path = line.strip()
        if path:
            blocks.append(dict(path=path, rows=pq.ParquetFile(path).metadata.num_rows))

    return blocks


def assign_blocks(blocks, rank, world_size):
    target = sum(b["rows"] for b in blocks) / world_size
    owner, current, running = [], 0, 0

    for b in blocks:
        owner.append(current)
        running += b["rows"]

        if current < world_size - 1 and running >= target * (current + 1):
            current += 1

    return [b for b, o in zip(blocks, owner) if o == rank]


def load_scaled_features(path, mean, scale):
    column = pq.read_table(path, columns=[FEATURE_COLUMN]).column(0).combine_chunks()
    raw = column.values.to_numpy(zero_copy_only=False).reshape(len(column), -1).astype(np.float32)

    return (raw - mean) / scale


def anchor_order_from_fps(anchor_fps):
    table = pq.read_table(anchor_fps, columns=["fps_order", "input_row"])
    fps_order = table.column("fps_order").to_numpy()
    input_row = table.column("input_row").to_numpy()

    return input_row[np.argsort(fps_order)].astype(np.int64)


def write_coreset_parquet(selected, blocks, out_path):
    rank_of = {int(g): i for i, g in enumerate(selected)}
    rows = [None] * len(selected)
    start = 0

    for b in blocks:
        chosen = [g for g in range(start, start + b["rows"]) if g in rank_of]

        if chosen:
            ids = pq.read_table(b["path"], columns=["configuration_id", "dataset_id"])
            configuration_id = ids.column("configuration_id").to_pylist()
            dataset_id = ids.column("dataset_id").to_pylist()
            for g in chosen:
                rows[rank_of[g]] = (rank_of[g], configuration_id[g - start], dataset_id[g - start], g)

        start += b["rows"]

    rows = [r for r in rows if r is not None]
    table = pa.table({
        "fps_order": pa.array([r[0] for r in rows], pa.int64()),
        "configuration_id": pa.array([r[1] for r in rows], pa.string()),
        "dataset_id": pa.array([r[2] for r in rows], pa.string()),
        "global_id": pa.array([r[3] for r in rows], pa.int64())
    })

    pq.write_table(table, out_path)
    print(f"wrote {out_path} ({table.num_rows:,} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--anchor-features", required=True)
    parser.add_argument("--anchor-fps", default=None)
    parser.add_argument("--n-target", type=int, default=None,
                        help="exact coreset size, omit for a full accrete over the whole pool")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rank, world_size = get_rank_and_world()
    init_distributed(rank, world_size)
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    mean, scale = load_scaler(args.scaler)
    anchor_rows = pq.ParquetFile(args.anchor_features).metadata.num_rows
    if args.anchor_fps:
        anchor_order = anchor_order_from_fps(args.anchor_fps)
        if len(anchor_order) != anchor_rows:
            raise SystemExit(f"anchor fps rows ({len(anchor_order)}) != anchor features rows ({anchor_rows})")
    else:
        anchor_order = np.arange(anchor_rows, dtype=np.int64)

    blocks = build_blocks(args.manifest, args.anchor_features, anchor_rows)
    pool_size = sum(b["rows"] for b in blocks)
    n_target = args.n_target if args.n_target is not None else pool_size
    max_k = max(1, math.ceil(math.log2(n_target)))

    checkpoint = resume_from = None
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = f"{args.checkpoint_dir}/ckpt"
        resume_from = checkpoint if (args.resume and os.path.exists(f"{checkpoint}_latest.txt")) else None

        fingerprint = f"{pool_size}:{anchor_rows}:{len(blocks)}:" + hashlib.sha256(
            open(args.manifest, "rb").read()).hexdigest()[:16]

        fingerprint_path = f"{args.checkpoint_dir}/fingerprint.txt"
        if resume_from is not None:
            saved = open(fingerprint_path).read().strip() if os.path.exists(fingerprint_path) else ""
    
            if saved != fingerprint:
                raise SystemExit(f"resume fingerprint mismatch: saved={saved!r} now={fingerprint!r}")

        elif rank == 0:
            open(fingerprint_path, "w").write(fingerprint + "\n")

    my_blocks = assign_blocks(blocks, rank, world_size)
    if my_blocks:
        total = sum(b["rows"] for b in my_blocks)               # preallocate + fill (avoids the vstack copy)
        features = np.empty((total, FEATURE_DIM), dtype=np.float32)
        pos = 0
        for b in my_blocks:
            arr = load_scaled_features(b["path"], mean, scale)
            features[pos:pos + len(arr)] = arr
            pos += len(arr)
        local_features = torch.from_numpy(features).cuda().contiguous()
    else:
        local_features = torch.zeros((0, FEATURE_DIM), dtype=torch.float32, device="cuda")

    log(f"world={world_size} pool={pool_size:,} anchor={anchor_rows:,} n_target={n_target:,} "
        f"max_k={max_k} resume={'yes' if resume_from else 'no'}", rank)

    coreset = accrete_sharded(local_features,
                              anchor_order=torch.from_numpy(anchor_order).long().cuda(),
                              n_anchor=anchor_rows,
                              max_k=max_k,
                              n_target=n_target,
                              checkpoint_path=checkpoint,
                              resume_from=resume_from,
                              checkpoint_every=args.checkpoint_every,
                              verbose=(rank == 0)
                            )
    selected = coreset.cpu().numpy().astype(np.int64)

    if rank == 0:
        np.save(f"{args.out_dir}/coreset_global_ids.npy", selected)
        shutil.copyfile(args.manifest, f"{args.out_dir}/manifest.txt")
        json.dump({"K": int(selected.shape[0]), "n_target": int(n_target), "max_k": max_k,
                   "n_anchor": anchor_rows, "anchor_features": args.anchor_features,
                   "anchor_fps": args.anchor_fps, "scaler": args.scaler, "pool_size": pool_size,
                   "world_size": world_size, "resumed": resume_from is not None,
                   "wall_seconds": time.time() - started},
                  open(f"{args.out_dir}/accrete.parquet.meta.json", "w"), indent=2)

        log(f"done: {selected.shape[0]:,} picks in {time.time() - started:.0f}s", rank)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        write_coreset_parquet(selected, blocks, f"{args.out_dir}/accrete.parquet")


if __name__ == "__main__":
    main()
