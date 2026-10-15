import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voronoi_net import brute_all_levels, load_scaler

MAX_LEVEL = 24
voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
tree_dir = f"{voronoi_dir}/out/tree"
scaler_path = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--manifest", default=f"{voronoi_dir}/out/manifest_all.txt")
    parser.add_argument("--out-dir", default=f"{voronoi_dir}/out/assign")
    parser.add_argument("--batch", type=int, default=131_072)
    parser.add_argument("--gpu-batch", type=int, default=8_192)
    parser.add_argument("--center-chunk", type=int, default=131_072)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--reverse", action="store_true",
                        help="work the manifest from the end (safe helper mode alongside a forward job)")
    return parser.parse_args()


def output_path(out_dir, shard_path):
    if "/phace_atomic_ep92/" in shard_path:
        relative = shard_path.split("/phace_atomic_ep92/")[1]
        family = relative.split("/")[0]
        stem = relative.replace("/", "__").removesuffix(".parquet")
    else:
        family = "mad_union"
        stem = "mad_union__" + os.path.basename(shard_path).removesuffix(".parquet")
    return f"{out_dir}/{family}/{stem}.parquet", family


def load_center_features(device):
    feats = np.memmap(f"{tree_dir}/center_feats_scaled.f32", dtype=np.float32, mode="r",
                      shape=(2 ** MAX_LEVEL, 512))
    return torch.from_numpy(np.ascontiguousarray(feats)).to(device)


def load_center_configuration_ids():
    return np.load(f"{tree_dir}/center_configuration_ids.npy", allow_pickle=True)


def main():
    args = parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False

    shards = [line.strip() for line in open(args.manifest) if line.strip()]
    if args.reverse:
        shards = shards[::-1]
    my_shards = shards[args.task_id::args.num_tasks]
    if args.limit_shards:
        my_shards = my_shards[:args.limit_shards]
    print(f"task {args.task_id}/{args.num_tasks}{' (reverse)' if args.reverse else ''}: "
          f"{len(my_shards)} shards", flush=True)

    feats = load_center_features(device)
    mean, scale = load_scaler(scaler_path)
    mean_t = torch.from_numpy(mean).to(device)
    print(f"center features on GPU: {tuple(feats.shape)}", flush=True)

    level_columns = [f"a{level:02d}" for level in range(MAX_LEVEL + 1)]
    center_cid_columns = [f"c{level:02d}" for level in range(MAX_LEVEL + 1)]
    dist_columns = [f"d{level:02d}" for level in range(MAX_LEVEL + 1)]
    schema = pa.schema([("configuration_id", pa.string()), ("row", pa.int64()),
                        ("family", pa.string()), ("dataset_id", pa.string())]
                       + [(name, pa.int32()) for name in level_columns]
                       + [(name, pa.string()) for name in center_cid_columns]
                       + [(name, pa.float32()) for name in dist_columns])
    center_cids = load_center_configuration_ids()

    stats = {"shards_done": 0, "points": 0, "overflow_points": 0, "seconds_io": 0.0, "seconds_gpu": 0.0}
    for shard in my_shards:
        out_path, family = output_path(args.out_dir, shard)
        if os.path.exists(out_path):
            stats["shards_done"] += 1
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        tmp_path = f"{out_path}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
        writer = pq.ParquetWriter(tmp_path, schema, compression="zstd")
        parquet_file = pq.ParquetFile(shard)
        row_offset = 0
        source_columns = ["configuration_id", "phace_features_bb_mean"]
        has_dataset_id = "dataset_id" in parquet_file.schema_arrow.names
        if has_dataset_id:
            source_columns.append("dataset_id")
        for record_batch in parquet_file.iter_batches(batch_size=args.batch, columns=source_columns):
            t0 = time.time()
            table = pa.Table.from_batches([record_batch])
            configuration_ids = table.column("configuration_id")
            dataset_ids = (table.column("dataset_id") if has_dataset_id
                           else pa.array([None] * table.num_rows, pa.string()))
            block = np.asarray(table.column("phace_features_bb_mean").combine_chunks()
                               .values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 512)
            n = block.shape[0]
            X = (torch.from_numpy(block).to(device) - mean_t) / scale
            t1 = time.time()

            assigns = torch.empty((n, MAX_LEVEL + 1), dtype=torch.int64, device=device)
            dists = torch.empty((n, MAX_LEVEL + 1), dtype=torch.float32, device=device)
            for lo in range(0, n, args.gpu_batch):
                hi = min(lo + args.gpu_batch, n)
                assigns[lo:hi], dists[lo:hi] = brute_all_levels(X[lo:hi], feats, MAX_LEVEL,
                                                                chunk=args.center_chunk)
            t2 = time.time()

            columns = {"configuration_id": configuration_ids,
                       "row": pa.array(np.arange(row_offset, row_offset + n, dtype=np.int64)),
                       "family": pa.array([family] * n, pa.string()),
                       "dataset_id": dataset_ids}
            assigns_np = assigns.cpu().numpy().astype(np.int32)
            dists_np = dists.cpu().numpy()
            for level in range(MAX_LEVEL + 1):
                columns[level_columns[level]] = pa.array(assigns_np[:, level])
                columns[center_cid_columns[level]] = pa.array(center_cids[assigns_np[:, level]].tolist(), pa.string())
                columns[dist_columns[level]] = pa.array(dists_np[:, level])
            writer.write_table(pa.table(columns, schema=schema))
            row_offset += n
            stats["points"] += n
            stats["seconds_io"] += t1 - t0
            stats["seconds_gpu"] += t2 - t1
        writer.close()
        os.rename(tmp_path, out_path)
        stats["shards_done"] += 1
        if stats["shards_done"] % 10 == 0:
            print(f"  {stats['shards_done']}/{len(my_shards)} shards, {stats['points']:,} points, "
                  f"overflow={stats['overflow_points']:,}, io={stats['seconds_io']:.0f}s "
                  f"gpu={stats['seconds_gpu']:.0f}s", flush=True)

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    with open(f"{args.out_dir}/_stats_task{args.task_id:04d}_{job_id}.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print(f"DONE task {args.task_id}: {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
