import argparse
import glob
import json
import os
import time
from datetime import timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from coresets import fps_sharded
from fps_dense_sharded import fps_dense_sharded

RELEASE = "${STORE_ROOT}/mad-core/release"
CORESET = f"{RELEASE}/coreset224"
WRITE_CHUNK = 4_000_000
STRING_COLS = ("configuration_id", "dataset_group", "dataset_id")


def widen_strings(table):
    """Cast id columns to large_string: `take` concatenates chunks internally and
    int32 offsets overflow past 2 GB of string data (~100M ids)."""
    for name in STRING_COLS:
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, table.column(name).cast(pa.large_string()))
    return table

OUT_SCHEMA = pa.schema([
    ("cell", pa.int32()),
    ("cell_rank", pa.int32()),
    ("eps_sq", pa.float32()),
    ("configuration_id", pa.string()),
    ("dataset_group", pa.string()),
    ("dataset_id", pa.string()),
])


def init_dist(rank, world_size):
    if "MASTER_ADDR" not in os.environ:
        node = os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]
        if "[" in node:
            base, suffix = node.split("[", 1)
            node = base + suffix.split("-")[0].split(",")[0].rstrip("]")
        os.environ["MASTER_ADDR"] = node
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(hours=3))


def split_files(paths, world_size):
    """Contiguous row-balanced file runs, one per rank, none empty.

    Runs must stay contiguous in sorted-path order: a row's global index is its
    position in the concatenation of the parts in that order. Assigning purely by
    cumulative position can hand a rank zero files when one part dominates (which
    then launches kernels with zero blocks -> cudaErrorInvalidConfiguration), so
    each rank is guaranteed at least one part.
    """
    counts = [pq.read_metadata(p).num_rows for p in paths]
    total = sum(counts)
    if len(paths) < world_size:
        raise ValueError(f"{len(paths)} parts cannot fill {world_size} ranks; "
                         "run this cell with fewer nodes")

    runs, run_rows, start = [], [], 0
    for rank in range(world_size):
        ranks_left = world_size - rank
        if ranks_left == 1:
            end = len(paths)
        else:
            last_start = len(paths) - (ranks_left - 1)
            target = (total - sum(run_rows)) / ranks_left
            end, taken = start + 1, counts[start]
            while end < last_start and taken + counts[end] <= target:
                taken += counts[end]
                end += 1
        runs.append(paths[start:end])
        run_rows.append(sum(counts[start:end]))
        start = end
    return runs, run_rows, total


def load_features(paths):
    parts = []
    for path in paths:
        column = pq.read_table(path, columns=["features"]).column("features")
        flat = column.combine_chunks().flatten().to_numpy()
        parts.append(np.asarray(flat).reshape(-1, 512))
    if not parts:
        return np.empty((0, 512), dtype=np.float32)
    return np.concatenate(parts)


def find_seed(paths, shard_offset, center_cid):
    offset = shard_offset
    for path in paths:
        column = pq.read_table(path, columns=["configuration_id"])
        index = pc.index(column.column("configuration_id"), center_cid).as_py()
        if index >= 0:
            return offset + index
        offset += len(column)
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--parts-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=4_000_000)
    parser.add_argument("--dense", action="store_true",
                        help="use the dense driver (no dSL bound pass) instead of fps_sharded")
    parser.add_argument("--max-picks", type=int, default=None,
                        help="validation only: stop after this many new picks, write nothing")
    args = parser.parse_args()

    cell_name = f"c{args.cell:08d}"
    done_marker = f"{args.out_dir}/cell_{cell_name}.done"
    if os.path.exists(done_marker):
        print(f"{cell_name}: already done")
        return

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)
    os.environ["CORESETS_FPS_SHARD_XSEL"] = "1"
    init_dist(rank, world_size)

    paths = sorted(glob.glob(f"{args.parts_root}/{cell_name}/part_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parts for {cell_name}")
    runs, run_rows, n_global = split_files(paths, world_size)
    shard_offset = sum(run_rows[:rank])

    center_cid = str(np.load(f"{CORESET}/configuration_id.npy",
                             mmap_mode="r")[args.cell].astype("U"))

    t0 = time.time()
    features = load_features(runs[rank])
    x_local = torch.from_numpy(features).cuda()
    del features

    seed_local = find_seed(runs[rank], shard_offset, center_cid)
    seed_t = torch.tensor([seed_local], dtype=torch.int64, device="cuda")
    dist.all_reduce(seed_t, op=dist.ReduceOp.MAX)
    seed = int(seed_t.item())
    if seed < 0:
        raise RuntimeError(f"{cell_name}: center row {center_cid!r} not found in pool")
    if rank == 0:
        print(f"{cell_name}: n={n_global:,} world={world_size} seed={seed} "
              f"load {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.checkpoint_root, exist_ok=True)
    checkpoint_path = f"{args.checkpoint_root}/{cell_name}"
    resume_from = checkpoint_path if os.path.exists(f"{checkpoint_path}_latest.txt") else None

    t0 = time.time()
    if args.dense:
        selected, epsilons = fps_dense_sharded(
            x_local, n_global, seed=seed,
            checkpoint_path=None if args.max_picks else checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_from=resume_from,
            max_picks=args.max_picks,
        )
    else:
        selected, epsilons = fps_sharded(
            x_local, n_global, seed=seed,
            checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            resume_from=resume_from,
        )
    fps_seconds = time.time() - t0

    if args.max_picks:
        if rank == 0:
            np.save(f"{args.out_dir}/validate_{cell_name}_selected.npy",
                    selected.cpu().numpy())
            print(f"{cell_name}: validation run, {len(selected):,} picks in "
                  f"{fps_seconds:.0f}s", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return

    if rank == 0:
        selected_np = selected.cpu().numpy()
        eps_np = np.asarray(epsilons, dtype=np.float32).copy()
        eps_np[0] = 0.0

        ids = pa.concat_tables(
            pq.read_table(p, columns=["cell", "d24", "configuration_id",
                                      "dataset_group", "dataset_id"])
            for p in paths)
        assert len(ids) == n_global

        d24 = ids.column("d24").to_numpy()
        radius_sq = float(np.max(d24.astype(np.float64) ** 2))
        problems = {
            "all_rows_this_cell": bool(
                (ids.column("cell").to_numpy() == args.cell).all()),
            "seed_is_center": bool(selected_np[0] == seed),
            "selected_is_permutation": bool(
                len(np.unique(selected_np)) == n_global),
            "eps_increase_steps": int(np.sum(
                np.diff(eps_np[1:]) > 1e-5 * np.abs(eps_np[1:-1]) + 1e-7)),
            "radius_rel_diff": abs(float(eps_np[1]) - radius_sq) / radius_sq,
        }

        # take/write in slices: a single `string` array caps at 2 GB of data
        # (int32 offsets), which 100M+ ids overflow.
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = f"{args.out_dir}/cell_{cell_name}.parquet"
        wide = widen_strings(ids)
        writer = pq.ParquetWriter(out_path, OUT_SCHEMA)
        for start in range(0, n_global, WRITE_CHUNK):
            stop = min(start + WRITE_CHUNK, n_global)
            ordered = wide.take(pa.array(selected_np[start:stop]))
            writer.write_table(pa.table({
                "cell": pa.array(np.full(stop - start, args.cell, dtype=np.int32)),
                "cell_rank": pa.array(np.arange(start, stop, dtype=np.int32)),
                "eps_sq": pa.array(eps_np[start:stop]),
                "configuration_id": ordered.column("configuration_id").cast(pa.string()),
                "dataset_group": ordered.column("dataset_group").cast(pa.string()),
                "dataset_id": ordered.column("dataset_id").cast(pa.string()),
            }, schema=OUT_SCHEMA))
        writer.close()

        summary = {"cell": args.cell, "n_rows": int(n_global), "world_size": world_size,
                   "seed": seed, "fps_seconds": round(fps_seconds, 1),
                   "problems": problems}
        with open(f"{args.out_dir}/cell_{cell_name}.json", "w") as fh:
            json.dump(summary, fh, indent=1)
        with open(done_marker, "w") as fh:
            fh.write("done\n")
        print(f"{cell_name}: DONE n={n_global:,} fps {fps_seconds:.0f}s "
              f"problems {problems}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
