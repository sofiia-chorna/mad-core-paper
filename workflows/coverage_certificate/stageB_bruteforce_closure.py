import glob
import json
import os
import pickle
import time

import numpy as np
import pyarrow.parquet as pq
import torch

RELEASE = "${STORE_ROOT}/mad-core/release"
FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
SMALL_ROOT = "${STORE_ROOT}/mad-core/accrete_pools/madcore_prod/small"
ANCHOR = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
POP = f"{HERE}/out/population"
STAGE_A = f"{HERE}/out/stageA"
OUT = f"{HERE}/out/stageB"
N_CORE = 16777216
LEVELS = 25
REL_BAND = 2e-3
STAGE_A_KEY = {"release": "rel", "selection": "sel"}
CENTER_BLOCK = 1 << 20
QUERY_BLOCK = 1024


def features_of(table):
    col = table.column("phace_features_bb_mean").combine_chunks()
    return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def folder_of(tag):
    kind, name = tag.split("__", 1)
    return kind, name, (f"{RELEASE}/collection/{name}" if kind != "anchor" else f"{RELEASE}/mad-next-coreset")


def folder_rows(folder):
    return int(np.load(f"{folder}/ns.npy")[0]) if "collection" in folder else np.load(f"{folder}/configuration_id.npy", mmap_mode="r").shape[0]


def gather_candidates(variant):
    per_level = [[] for _ in range(LEVELS)]
    truncated = [[] for _ in range(LEVELS)]
    global_max = np.full(LEVELS, -1.0)
    for path in sorted(glob.glob(f"{STAGE_A}/*.npz")):
        z = np.load(path)
        vals = z[f"top_vals_{STAGE_A_KEY[variant]}"]
        global_max = np.maximum(global_max, vals[:, 0])
    for path in sorted(glob.glob(f"{STAGE_A}/*.npz")):
        tag = os.path.basename(path)[:-4]
        z = np.load(path)
        vals, rows = z[f"top_vals_{STAGE_A_KEY[variant]}"], z[f"top_rows_{STAGE_A_KEY[variant]}"]
        for level in range(LEVELS):
            thr = global_max[level] * (1.0 - REL_BAND)
            keep = vals[level] >= thr
            if keep.all() and vals[level].size > 0:
                truncated[level].append(tag)
            for r, v in zip(rows[level][keep], vals[level][keep]):
                per_level[level].append((tag, int(r), float(v)))
    return per_level, truncated, global_max


def load_row_features(tag, rows, variant):
    kind, name, folder = folder_of(tag)
    if variant == "selection" and kind == "anchor":
        feats = features_of(pq.read_table(ANCHOR, columns=["phace_features_bb_mean"]))
        return feats[rows]
    if variant == "selection" and kind in ("big", "small"):
        first_row = np.load(f"{POP}/pool_row_to_release__{name}.npy")
        sizes = np.load(f"{POP}/pool_shard_sizes__{name}.npy")
        root = f"{FPS_ROOT}/{name}/with_features" if kind == "big" else f"{SMALL_ROOT}/{name}"
        shards = sorted(glob.glob(f"{root}/*.parquet"))
        uniq, first_idx = np.unique(first_row, return_index=True)
        wf_pos = first_idx[np.searchsorted(uniq, rows)]
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        shard_of = np.searchsorted(bounds, wf_pos, side="right") - 1
        out = np.empty((len(rows), 512), dtype=np.float32)
        for s in np.unique(shard_of):
            sel = np.where(shard_of == s)[0]
            feats = features_of(pq.read_table(shards[s], columns=["phace_features_bb_mean"]))
            out[sel] = feats[wf_pos[sel] - bounds[s]]
        return out
    n = folder_rows(folder)
    feat = np.memmap(f"{folder}/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(n, 512))
    return np.asarray(feat[rows])


def nearest_center_block(x, centers_gpu, n_centers, scale):
    q = torch.from_numpy(x.astype(np.float64)).cuda()
    qn = (q * q).sum(dim=1)
    best = torch.full((q.shape[0],), float("inf"), dtype=torch.float64, device="cuda")
    arg = torch.full((q.shape[0],), -1, dtype=torch.int64, device="cuda")
    for s in range(0, n_centers, CENTER_BLOCK):
        c = centers_gpu[s:min(s + CENTER_BLOCK, n_centers)].double()
        cn = (c * c).sum(dim=1)
        d = qn[:, None] + cn[None, :] - 2.0 * (q @ c.T)
        val, idx = d.min(dim=1)
        better = val < best
        best = torch.where(better, val, best)
        arg = torch.where(better, idx + s, arg)
    arg = arg.cpu().numpy()
    direct = np.sqrt(((x.astype(np.float64) - centers_gpu[torch.from_numpy(arg).cuda()].cpu().numpy().astype(np.float64)) ** 2).sum(axis=1)) / scale
    return direct, arg


def nearest_center(x, centers_gpu, n_centers, scale):
    parts = [nearest_center_block(x[s:s + QUERY_BLOCK], centers_gpu, n_centers, scale) for s in range(0, x.shape[0], QUERY_BLOCK)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SCALER, "rb") as f:
        scale = float(pickle.load(f)["scale_"])
    report = {}
    for variant in ("release", "selection"):
        started = time.time()
        if variant == "release":
            centers = np.fromfile(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32).reshape(N_CORE, 512)
        else:
            centers = np.load(f"{HERE}/out/selection_features_madcore.npy")
        centers_gpu = torch.from_numpy(centers).cuda()
        per_level, truncated, global_max = gather_candidates(variant)
        levels = {}
        for level in range(LEVELS):
            cands = per_level[level]
            by_tag = {}
            for tag, r, v in cands:
                by_tag.setdefault(tag, []).append((r, v))
            exact = []
            for tag, lst in by_tag.items():
                rows = np.asarray([r for r, _ in lst], dtype=np.int64)
                ub = np.asarray([v for _, v in lst])
                x = load_row_features(tag, rows, variant)
                d, arg = nearest_center(x, centers_gpu, 2 ** level, scale)
                for r, u, dd, a in zip(rows, ub, d, arg):
                    exact.append((tag, int(r), float(u), float(dd), int(a)))
            exact.sort(key=lambda t: -t[3])
            cov = exact[0][3] if exact else float("nan")
            ub_above_cov = sum(1 for e in exact if e[2] > cov)
            levels[level] = {
                "n_candidates": len(exact), "cov_exact_direct": cov,
                "witness": {"folder": exact[0][0], "row": exact[0][1], "assigned_ub": exact[0][2], "nearest_center": exact[0][4]} if exact else None,
                "global_assigned_ub": float(global_max[level]),
                "candidates_with_ub_above_cov": ub_above_cov,
                "truncated_folders": truncated[level],
                "band_threshold": float(global_max[level] * (1.0 - REL_BAND)),
                "closure_ok": bool((len(truncated[level]) == 0) and (global_max[level] * (1.0 - REL_BAND) <= cov)),
                "cov_minus_global_ub_rel": float(cov / global_max[level] - 1.0),
            }
            print(f"[{variant}] level {level:2d}: cov={cov:.8f} UB={global_max[level]:.8f} cands={len(exact)} "
                  f"truncated={len(truncated[level])} closure_ok={levels[level]['closure_ok']}", flush=True)
        report[variant] = levels
        del centers_gpu, centers
        torch.cuda.empty_cache()
        print(f"[{variant}] done ({time.time() - started:.0f}s)", flush=True)
    json.dump(report, open(f"{OUT}/coverage_exact.json", "w"), indent=1)


if __name__ == "__main__":
    main()
