import glob
import json
import pickle

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release/mad-core"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
STAGE_C = f"{HERE}/out/stageC"
N = 16777216
N_ANCHOR = 856012


def merge(variant):
    sep = np.empty(N, dtype=np.float64)
    nearest = np.empty(N, dtype=np.int64)
    paths = sorted(glob.glob(f"{STAGE_C}/range_{variant}_rank*.npy"))
    assert len(paths) == 4, f"{variant}: expected 4 rank files, found {len(paths)}"
    for path in paths:
        rank = path.split("rank")[-1][:-4]
        lo, hi = np.load(path)
        sep[lo:hi] = np.load(f"{STAGE_C}/sep_sq_{variant}_rank{rank}.npy")
        nearest[lo:hi] = np.load(f"{STAGE_C}/nearest_{variant}_rank{rank}.npy")
    np.save(f"{STAGE_C}/separation_sq_{variant}_full.npy", sep)
    np.save(f"{STAGE_C}/nearest_prev_{variant}_full.npy", nearest)
    return sep, nearest


def main():
    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler["mean_"], dtype=np.float64)
    scale = float(scaler["scale_"])
    k_set = np.load(f"{RELEASE}/k_set.npy", mmap_mode="r")
    stored = np.load(f"{RELEASE}/epsilon_sq.npy", mmap_mode="r")
    features = {
        "release": np.fromfile(f"{RELEASE}/ecumetric_features.bin", dtype=np.float32).reshape(N, 512),
        "selection": np.load(f"{HERE}/out/selection_features_madcore.npy"),
    }

    def direct_sq(feat, i, j):
        a = (feat[i].astype(np.float64) - mean) / scale
        b = (feat[j].astype(np.float64) - mean) / scale
        return float(np.sum((a - b) ** 2))

    report = {}
    for variant in ("release", "selection"):
        sep, nearest = merge(variant)
        feat = features[variant]
        sep[0] = np.inf
        cummin = np.minimum.accumulate(sep)
        rows = []
        sizes = [2 ** k for k in range(1, 25)]
        sizes.insert(19, N_ANCHOR)
        for size in sizes:
            last = size - 1
            sep_pick = int(np.argmin(sep[:size]))
            sep_direct = direct_sq(feat, sep_pick, int(nearest[sep_pick]))
            if size < N:
                nxt_end = min(2 * size, N) if size != N_ANCHOR else 2 ** 20
                nxt = np.arange(size, nxt_end)
                lb_pick = int(nxt[np.argmax(sep[nxt])])
                lb_direct = direct_sq(feat, lb_pick, int(nearest[lb_pick]))
                first_next = direct_sq(feat, size, int(nearest[size]))
            else:
                lb_pick, lb_direct, first_next = -1, float("nan"), float("nan")
            rows.append({
                "size": size, "level": int(np.log2(size)) if size != N_ANCHOR else "anchor",
                "sep_sq_direct": sep_direct, "sep_pair": [sep_pick, int(nearest[sep_pick])],
                "sep_sq_expanded_cummin": float(cummin[last]),
                "stored_eps_sq_at_last_row": float(stored[last]),
                "lb_sq_direct_next_level_max": lb_direct, "lb_pick": lb_pick,
                "first_next_pick_sq_direct": first_next,
                "lb_over_sep_sq_minus_1": lb_direct / sep_direct - 1.0 if size < N else float("nan"),
            })
            print(f"[{variant}] size {size:>9,} sep={np.sqrt(sep_direct):.8f} lb_next={np.sqrt(lb_direct) if size < N else float('nan'):.8f} "
                  f"lb/sep^2-1={rows[-1]['lb_over_sep_sq_minus_1']:+.3e} stored={np.sqrt(stored[last]):.8f}", flush=True)
        cont = sep[N_ANCHOR:]
        rows_meta = {
            "continuation_upward_steps": int((np.diff(cont) > 0).sum()),
            "continuation_max_upward_rel": float(np.max(np.diff(cont) / cont[:-1])),
            "continuation_stored_vs_exact_rel_max": float(np.max(np.abs(cont - np.asarray(stored[N_ANCHOR:])) / np.asarray(stored[N_ANCHOR:]))),
        }
        report[variant] = {"levels": rows, **rows_meta}
    json.dump(report, open(f"{STAGE_C}/levels_full.json", "w"), indent=1)


if __name__ == "__main__":
    main()
