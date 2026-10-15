import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voronoi_net import brute_all_levels, load_scaler

MAX_LEVEL = 24
NUM_PIVOTS = 2 ** 11
COPY_CHUNK = 262_144
voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
tree_dir = f"{voronoi_dir}/out/tree"
scaler_path = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
device = "cuda"

TEST_SHARDS = [
    "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92/AENET_liquid_water_dataset_JCP2021/colabfit_AENET_liquid_water_dataset_JCP2021/passed_co_0.parquet",
    "${STORE_LP133}/mad-core/datasets/phace_atomic_ep92/gdb13/t13_c0587.parquet",
    "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92/ANI-1/colabfit_ANI-1/passed_co_14.parquet",
]
POINTS_PER_SHARD = 65_536
GPU_BATCH = 8_192


def build_blocked_structure(feats):
    pivots = feats[:NUM_PIVOTS].clone()
    print("assigning all centers to the 2048 level-11 pivots", flush=True)
    cell = torch.empty(2 ** MAX_LEVEL, dtype=torch.int64, device=device)
    for lo in range(0, 2 ** MAX_LEVEL, COPY_CHUNK):
        hi = min(lo + COPY_CHUNK, 2 ** MAX_LEVEL)
        cell[lo:hi] = torch.cdist(feats[lo:hi], pivots).argmin(1)
    cell[:NUM_PIVOTS] = torch.arange(NUM_PIVOTS, device=device)

    reordered = torch.empty_like(feats)
    reordered[:2 ** 11] = feats[:2 ** 11]
    segments = {}
    for level in range(12, MAX_LEVEL + 1):
        lo, hi = 2 ** (level - 1), 2 ** level
        order = torch.argsort(cell[lo:hi], stable=True)
        perm = order + lo
        sorted_cell = cell[lo:hi][order]
        counts = torch.bincount(sorted_cell, minlength=NUM_PIVOTS)
        offsets = torch.zeros(NUM_PIVOTS + 1, dtype=torch.int64, device=device)
        offsets[1:] = counts.cumsum(0)

        radius = torch.zeros(NUM_PIVOTS, dtype=torch.float32, device=device)
        for c in range(0, hi - lo, COPY_CHUNK):
            c_hi = min(c + COPY_CHUNK, hi - lo)
            reordered[lo + c:lo + c_hi] = feats[perm[c:c_hi]]
            member_dist = (reordered[lo + c:lo + c_hi] - pivots[sorted_cell[c:c_hi]]).pow(2).sum(1).sqrt_()
            radius.scatter_reduce_(0, sorted_cell[c:c_hi], member_dist, reduce="amax", include_self=True)

        segments[level] = {"perm": perm, "offsets": offsets, "radius": radius}
        print(f"  level {level}: {hi - lo:,} centers, nonempty groups "
              f"{int((counts > 0).sum())}/{NUM_PIVOTS}, max radius {float(radius.max()):.3f}", flush=True)
    return reordered, pivots, segments


def group_member_indices(offsets, group_ids, base):
    sizes = offsets[group_ids + 1] - offsets[group_ids]
    total = int(sizes.sum())
    if total == 0:
        return None
    starts = offsets[group_ids]
    flat = torch.arange(total, device=device) - (sizes.cumsum(0) - sizes).repeat_interleave(sizes) \
        + starts.repeat_interleave(sizes)
    return flat + base


def blocked_all_levels(X, reordered, pivots, segments, stats):
    num_points = X.shape[0]
    pivot_d = torch.cdist(X, pivots)

    best_d = (X - reordered[0]).pow(2).sum(1).sqrt_()
    best_id = torch.zeros(num_points, dtype=torch.int64, device=device)
    assigns = torch.empty((num_points, MAX_LEVEL + 1), dtype=torch.int64, device=device)
    dists = torch.empty((num_points, MAX_LEVEL + 1), dtype=torch.float32, device=device)
    assigns[:, 0] = 0
    dists[:, 0] = best_d

    for level in range(1, 12):
        lo, hi = 2 ** (level - 1), 2 ** level
        d = torch.cdist(X, reordered[lo:hi])
        chunk_min, chunk_arg = d.min(1)
        improve = chunk_min < best_d
        best_d[improve] = chunk_min[improve]
        best_id[improve] = chunk_arg[improve] + lo
        assigns[:, level] = best_id
        dists[:, level] = best_d

    for level in range(12, MAX_LEVEL + 1):
        seg = segments[level]
        lo = 2 ** (level - 1)
        need = (pivot_d - best_d.unsqueeze(1)).min(0).values < seg["radius"]
        need &= (seg["offsets"][1:] - seg["offsets"][:-1]) > 0
        stats[f"centers_evaluated_l{level}"] = stats.get(f"centers_evaluated_l{level}", 0) + \
            int((seg["offsets"][need.nonzero(as_tuple=True)[0] + 1]
                 - seg["offsets"][need.nonzero(as_tuple=True)[0]]).sum())
        stats[f"batches_l{level}"] = stats.get(f"batches_l{level}", 0) + 1
        if bool(need.any()):
            local = group_member_indices(seg["offsets"], need.nonzero(as_tuple=True)[0], lo)
            if local is not None:
                for c in range(0, local.shape[0], 131_072):
                    piece = local[c:c + 131_072]
                    d = torch.cdist(X, reordered[piece])
                    chunk_min, chunk_arg = d.min(1)
                    improve = chunk_min < best_d
                    best_d[improve] = chunk_min[improve]
                    best_id[improve] = seg["perm"][piece[chunk_arg[improve]] - lo]
        assigns[:, level] = best_id
        dists[:, level] = best_d

    return assigns, dists


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    feats_mm = np.memmap(f"{tree_dir}/center_feats_scaled.f32", dtype=np.float32, mode="r",
                         shape=(2 ** MAX_LEVEL, 512))
    feats = torch.from_numpy(np.ascontiguousarray(feats_mm)).to(device)
    mean, scale = load_scaler(scaler_path)
    mean_t = torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(device)
    print("centers on GPU", flush=True)

    t0 = time.time()
    reordered, pivots, segments = build_blocked_structure(feats)
    print(f"blocked structure built in {time.time() - t0:.1f}s", flush=True)

    for shard in TEST_SHARDS:
        name = shard.split("/phace_atomic_ep92/")[1]
        table = pq.read_table(shard, columns=["phace_features_bb_mean"])
        block = np.asarray(table.column("phace_features_bb_mean").combine_chunks()
                           .values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 512)
        block = block[:POINTS_PER_SHARD]
        X_all = (torch.from_numpy(block).to(device) - mean_t) / scale
        n = X_all.shape[0]

        torch.cuda.synchronize()
        t0 = time.time()
        brute_assigns = []
        brute_dists = []
        for lo in range(0, n, GPU_BATCH):
            a, d = brute_all_levels(X_all[lo:lo + GPU_BATCH], feats, MAX_LEVEL, chunk=131_072)
            brute_assigns.append(a)
            brute_dists.append(d)
        torch.cuda.synchronize()
        t_brute = time.time() - t0

        stats = {}
        torch.cuda.synchronize()
        t0 = time.time()
        blocked_assigns = []
        blocked_dists = []
        for lo in range(0, n, GPU_BATCH):
            a, d = blocked_all_levels(X_all[lo:lo + GPU_BATCH], reordered, pivots, segments, stats)
            blocked_assigns.append(a)
            blocked_dists.append(d)
        torch.cuda.synchronize()
        t_blocked = time.time() - t0

        identical = bool((torch.cat(brute_assigns) == torch.cat(blocked_assigns)).all())
        max_d_err = float((torch.cat(brute_dists) - torch.cat(blocked_dists)).abs().max())
        evaluated = sum(v for k, v in stats.items() if k.startswith("centers_evaluated"))
        batches = stats.get("batches_l24", 1)
        total_possible = batches * (2 ** MAX_LEVEL - 2 ** 11)
        print(f"\n=== {name} ({n:,} points) ===", flush=True)
        print(f"  brute:   {t_brute:7.1f}s  ({n / t_brute:7.0f} pts/s)")
        print(f"  blocked: {t_blocked:7.1f}s  ({n / t_blocked:7.0f} pts/s)   speedup x{t_brute / t_blocked:.2f}")
        print(f"  identical assignments: {identical}   max |d| err: {max_d_err:.2e}")
        print(f"  centers evaluated (levels>=12): {100 * evaluated / total_possible:.1f}% of dense", flush=True)


if __name__ == "__main__":
    main()
