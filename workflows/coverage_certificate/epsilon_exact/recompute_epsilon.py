import json
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

RELEASE = "${STORE_ROOT}/mad-core/release/mad-core"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
OUT = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact/out"
SELECTION = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out/selection_features_madcore.npy"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "release"
TAG = "" if VARIANT == "release" else f"_{VARIANT}"

N_RELEASE = 16777216
N_ANCHOR = 856012
N_CHECK = 100000
FEATURE_DIM = 512
CHUNK = 1024


def load_scaled_features(n_rows):
    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler["mean_"], dtype=np.float64)
    scale = float(scaler["scale_"])

    if VARIANT == "release":
        raw = np.memmap(f"{RELEASE}/ecumetric_features.bin", dtype=np.float32, mode="r",
                        shape=(N_RELEASE, FEATURE_DIM))
    else:
        raw = np.load(SELECTION, mmap_mode="r")
    scaled = (np.asarray(raw[:n_rows], dtype=np.float64) - mean) / scale

    return scaled, mean, scale


def nearest_previous(features):
    n = features.shape[0]
    x = torch.from_numpy(features).cuda()
    norms = (x * x).sum(dim=1)
    separation_sq = torch.full((n,), float("nan"), dtype=torch.float64, device="cuda")
    nearest = torch.full((n,), -1, dtype=torch.int64, device="cuda")

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        gram = x[start:stop] @ x[:stop].T
        dist_sq = norms[start:stop, None] + norms[None, :stop] - 2.0 * gram

        rows = torch.arange(start, stop, device="cuda")[:, None]
        cols = torch.arange(stop, device="cuda")[None, :]
        dist_sq.masked_fill_(cols >= rows, float("inf"))

        best, arg = dist_sq.min(dim=1)
        has_previous = torch.arange(start, stop, device="cuda") > 0
        separation_sq[start:stop] = torch.where(has_previous, best.clamp_min(0.0), torch.nan)
        nearest[start:stop] = torch.where(has_previous, arg, -1)

    return separation_sq.cpu().numpy(), nearest.cpu().numpy()


def level_report(exact_sq, stored_sq, k_set):
    levels = []
    cummin_sq = np.fmin.accumulate(np.nan_to_num(exact_sq, nan=np.inf))
    cummin_d = np.sqrt(cummin_sq)

    for level in np.unique(k_set[:N_ANCHOR]):
        rows = np.where(k_set[:N_ANCHOR] == level)[0]
        first, last = int(rows[0]), int(rows[-1])
        values = exact_sq[rows]
        finite = values[np.isfinite(values)]
        steps = np.diff(finite)
        levels.append({
            "level": int(level), "rows": [first, last], "n": int(rows.size),
            "stored_stair_sq": float(stored_sq[last]),
            "cummin_exact_sq_at_level_end": float(cummin_sq[last]),
            "cummin_exact_d_at_level_end": float(cummin_d[last]),
            "exact_sq_min": float(finite.min()), "exact_sq_max": float(finite.max()),
            "upward_steps": int((steps > 0).sum()),
            "largest_upward_step_sq": float(steps.max()) if steps.size else 0.0,
        })

    return levels


def continuation_report(exact_sq, stored_sq):
    rows = np.arange(N_ANCHOR, N_ANCHOR + N_CHECK)
    rel = np.abs(exact_sq[rows] - stored_sq[rows]) / stored_sq[rows]

    return {
        "rows": [int(rows[0]), int(rows[-1])],
        "rel_diff_max": float(rel.max()),
        "rel_diff_median": float(np.median(rel)),
        "rel_diff_p99": float(np.percentile(rel, 99)),
        "rows_within_1e-3": int((rel <= 1e-3).sum()),
        "rows_beyond_1e-3": [int(r) for r in rows[rel > 1e-3][:20]],
    }


def plot(exact_sq, stored_sq):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = np.arange(1, N_ANCHOR)
    ax.plot(rows, np.sqrt(exact_sq[1:N_ANCHOR]), lw=0.3, color="0.4", label="exact per-pick separation")
    ax.plot(rows, np.sqrt(stored_sq[1:N_ANCHOR]), lw=1.5, color="C3", label="stored (per-level running min)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pick index")
    ax.set_ylabel("epsilon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/anchor_exact_vs_stored{TAG}.png", dpi=150)


def main():
    started = time.time()
    n_rows = N_ANCHOR + N_CHECK

    features, mean, scale = load_scaled_features(n_rows)
    print(f"loaded {n_rows:,} rows, scale={scale}, scaled row norm[:3]={np.linalg.norm(features[:3], axis=1)}", flush=True)

    exact_sq, nearest = nearest_previous(features)
    print(f"computed in {time.time() - started:.0f}s", flush=True)

    stored_sq = np.load(f"{RELEASE}/epsilon_sq.npy", mmap_mode="r")[:n_rows]
    k_set = np.load(f"{RELEASE}/k_set.npy", mmap_mode="r")[:n_rows]

    np.save(f"{OUT}/separation_sq{TAG}.npy", exact_sq)
    np.save(f"{OUT}/nearest_prev{TAG}.npy", nearest)

    report = {
        "definition": "separation_sq[i] = min_{j<i} ||s_i - s_j||^2, s = (ecumetric_features - mean_) / scale_, float64; row 0 NaN",
        "rows_computed": n_rows,
        "n_anchor": N_ANCHOR,
        "d2_row1_row0": float(exact_sq[1]),
        "stored_row0": float(stored_sq[0]),
        "anchor_levels": level_report(exact_sq, stored_sq, k_set),
        "continuation_check": continuation_report(exact_sq, stored_sq),
        "wall_seconds": time.time() - started,
    }
    json.dump(report, open(f"{OUT}/report{TAG}.json", "w"), indent=2)

    plot(exact_sq, stored_sq)

    print(json.dumps({k: v for k, v in report.items() if k != "anchor_levels"}, indent=2))
    for lv in report["anchor_levels"]:
        print(f"level {lv['level']:2d} n={lv['n']:7d} stored={lv['stored_stair_sq']:.6f} "
              f"cummin_sq={lv['cummin_exact_sq_at_level_end']:.6f} cummin_d={lv['cummin_exact_d_at_level_end']:.6f} "
              f"min={lv['exact_sq_min']:.4f} max={lv['exact_sq_max']:.4f} up={lv['upward_steps']}")


if __name__ == "__main__":
    main()
