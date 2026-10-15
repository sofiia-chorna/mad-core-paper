import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voronoi_net import descend, load_scaler

MAX_LEVEL = 24
BEAM_WIDTH = 512
BATCH = 262_144

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
tree_dir = f"{voronoi_dir}/out/tree"
centers_dir = f"{voronoi_dir}/out/centers"
coreset_full = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/coreset224/out/coreset_224_full.parquet"
scaler_path = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"

num_centers = 2 ** MAX_LEVEL
device = "cuda"


def load_center_features():
    memmap_path = f"{tree_dir}/center_feats_scaled.f32"
    ids_path = f"{tree_dir}/center_configuration_ids.npy"
    if os.path.exists(memmap_path) and os.path.exists(ids_path):
        print("scaled center features exist, reusing", flush=True)
        return np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(num_centers, 512))
    mean, scale = load_scaler(scaler_path)
    feats = np.empty((num_centers, 512), dtype=np.float32)
    configuration_ids = np.empty(num_centers, dtype=object)
    parquet_file = pq.ParquetFile(coreset_full)
    done = 0
    for batch in parquet_file.iter_batches(batch_size=65_536,
                                           columns=["fps_order", "configuration_id", "phace_features_bb_meanstd"]):
        table = pa.Table.from_batches([batch])
        fps_order = table.column("fps_order").to_numpy()
        block = np.asarray(table.column("phace_features_bb_meanstd").combine_chunks()
                           .values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 512)
        feats[fps_order] = (block - mean) / scale
        configuration_ids[fps_order] = np.asarray(table.column("configuration_id").to_pylist(), dtype=object)
        done += len(fps_order)
        if done % 2_000_000 < len(fps_order):
            print(f"  loaded {done:,}/{num_centers:,}", flush=True)
    assert done == num_centers
    with open(memmap_path + ".tmp", "wb") as fh:
        feats.tofile(fh)
    os.rename(memmap_path + ".tmp", memmap_path)
    np.save(ids_path, configuration_ids)
    print(f"center features scaled + saved sequentially ({done:,})", flush=True)
    del feats
    return np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(num_centers, 512))


def main():
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(centers_dir, exist_ok=True)
    feats_mm = load_center_features()
    feats = torch.from_numpy(np.ascontiguousarray(feats_mm)).to(device)
    print(f"features on GPU: {tuple(feats.shape)}", flush=True)

    parent = np.zeros(num_centers, dtype=np.int64)
    parent_distance = np.zeros(num_centers, dtype=np.float32)
    pcd = np.zeros(MAX_LEVEL + 2, dtype=np.float32)
    net = {"feats": feats, "suffix": np.zeros(MAX_LEVEL + 3, dtype=np.float32), "levels": {}}

    for level in range(1, MAX_LEVEL + 1):
        t0 = time.time()
        lo, hi = 2 ** (level - 1), 2 ** level
        net["suffix"] = np.concatenate([np.cumsum(pcd[::-1])[::-1], [0.0, 0.0]]).astype(np.float32)

        new_parent = torch.empty(hi - lo, dtype=torch.int64, device=device)
        new_distance = torch.empty(hi - lo, dtype=torch.float32, device=device)
        n_overflow = 0
        for batch_lo in range(lo, hi, BATCH):
            batch_hi = min(batch_lo + BATCH, hi)
            X = feats[batch_lo:batch_hi]
            if level == 1:
                new_parent[batch_lo - lo:batch_hi - lo] = 0
                new_distance[batch_lo - lo:batch_hi - lo] = (X - feats[0]).pow(2).sum(1).sqrt_()
                continue
            assigns, dists, overflow = descend(X, net, level - 1, BEAM_WIDTH, record_from=level - 1)
            if bool(overflow.any()):
                bad = overflow.nonzero(as_tuple=True)[0]
                n_overflow += int(bad.numel())
                for point in bad.tolist():
                    exact = torch.cdist(X[point:point + 1], feats[:lo]).squeeze(0)
                    dists[point, 0], assigns[point, 0] = exact.min(0)
            new_parent[batch_lo - lo:batch_hi - lo] = assigns[:, 0]
            new_distance[batch_lo - lo:batch_hi - lo] = dists[:, 0]

        order = torch.argsort(new_parent, stable=True)
        sorted_parent = new_parent[order]
        child_ids = torch.arange(lo, hi, dtype=torch.int64, device=device)[order]
        child_dp = new_distance[order]
        indptr = torch.zeros(lo + 1, dtype=torch.int64, device=device)
        indptr.scatter_add_(0, sorted_parent + 1, torch.ones_like(sorted_parent))
        indptr = indptr.cumsum(0)
        net["levels"][level] = {"indptr": indptr, "child": child_ids, "dp": child_dp}

        parent[lo:hi] = new_parent.cpu().numpy()
        parent_distance[lo:hi] = new_distance.cpu().numpy()
        pcd[level] = float(new_distance.max())
        print(f"level {level:>2}: new={hi - lo:,}  pcd={pcd[level]:.4f}  overflow={n_overflow}  "
              f"t={time.time() - t0:.1f}s", flush=True)

    suffix = np.concatenate([np.cumsum(pcd[::-1])[::-1], [0.0, 0.0]]).astype(np.float32)
    np.savez(f"{tree_dir}/net_arrays.npz",
             parent=parent, parent_distance=parent_distance, pcd=pcd, suffix=suffix)
    torch.save({level: {key: value.cpu() for key, value in csr.items()}
                for level, csr in net["levels"].items()}, f"{tree_dir}/levels_csr.pt")
    print("net saved", flush=True)

    configuration_ids = np.load(f"{tree_dir}/center_configuration_ids.npy", allow_pickle=True)
    for level in range(MAX_LEVEL + 1):
        size = 2 ** level
        pq.write_table(pa.table({
            "fps_order": pa.array(np.arange(size, dtype=np.int64)),
            "configuration_id": pa.array(configuration_ids[:size].tolist(), pa.string()),
        }), f"{centers_dir}/centers_l{level:02d}.parquet", compression="zstd")
    print("per-level center lists saved  DONE", flush=True)


if __name__ == "__main__":
    main()
