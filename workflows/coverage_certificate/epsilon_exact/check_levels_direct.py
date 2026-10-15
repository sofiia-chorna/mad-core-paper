import json
import pickle
import sys

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release/mad-core"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
OUT = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact/out"
N_ANCHOR = 856012
N_ROWS = 956012
SELECTION = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out/selection_features_madcore.npy"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "release"
TAG = "" if VARIANT == "release" else f"_{VARIANT}"


class NumpyCoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def main():
    with open(SCALER, "rb") as f:
        scaler = NumpyCoreUnpickler(f).load()
    mean = np.asarray(scaler["mean_"], dtype=np.float64)
    scale = float(scaler["scale_"])

    if VARIANT == "release":
        raw = np.memmap(f"{RELEASE}/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(16777216, 512))
    else:
        raw = np.load(SELECTION, mmap_mode="r")
    sep_sq = np.load(f"{OUT}/separation_sq{TAG}.npy")
    nearest = np.load(f"{OUT}/nearest_prev{TAG}.npy")
    k_set = np.asarray(np.load(f"{RELEASE}/k_set.npy", mmap_mode="r")[:N_ROWS])
    rd = np.asarray(np.load(f"{RELEASE}/release_directory.npy", mmap_mode="r")[:N_ROWS])
    names = np.load(f"{RELEASE}/release_directory_names.npy")
    anchor_dir = [i for i, n in enumerate(names) if n == b"mad-next-coreset"][0]
    is_anchor = rd == anchor_dir

    def direct_sq(i, j):
        a = (np.asarray(raw[i], dtype=np.float64) - mean) / scale
        b = (np.asarray(raw[j], dtype=np.float64) - mean) / scale
        return float(np.sum((a - b) ** 2))

    cummin = np.fmin.accumulate(np.nan_to_num(sep_sq, nan=np.inf))
    rows = []
    for level in range(1, 21):
        members = np.where(k_set == level)[0]
        if level < 20:
            last = int(members[-1])
        else:
            last = N_ANCHOR - 1
            members = members[members < N_ANCHOR]
        sep_pick = int(np.argmin(np.nan_to_num(sep_sq[: last + 1], nan=np.inf)))
        sep_direct = direct_sq(sep_pick, int(nearest[sep_pick]))

        nxt = np.where(k_set == level + 1)[0] if level < 20 else np.arange(N_ANCHOR, N_ROWS)
        nxt_vals = sep_sq[nxt]
        order = np.argsort(-nxt_vals)
        lb_pick = int(nxt[order[0]])
        lb_direct = direct_sq(lb_pick, int(nearest[lb_pick]))
        second = float(nxt_vals[order[1]]) if nxt.size > 1 else float("nan")
        first_non_anchor = int(np.argmin(is_anchor[nxt])) if (~is_anchor[nxt]).any() else -1
        anchor_prefix = int(np.argmin(is_anchor[nxt])) if (~is_anchor[nxt]).any() else int(nxt.size)
        anchors_in_next = int(is_anchor[nxt].sum())

        rows.append({
            "level": level, "size": last + 1,
            "sep_sq_expanded": float(cummin[last]), "sep_sq_direct": sep_direct,
            "sep_pair": [sep_pick, int(nearest[sep_pick])],
            "lb_sq_expanded": float(nxt_vals[order[0]]), "lb_sq_direct": lb_direct,
            "lb_pick": lb_pick, "lb_pick_nearest": int(nearest[lb_pick]),
            "lb_pos_in_next_level": int(order[0]), "lb_pick_is_anchor": bool(is_anchor[lb_pick]),
            "second_largest_sq_in_next": second,
            "next_level_anchor_prefix_len": anchor_prefix, "next_level_anchor_members": anchors_in_next,
            "next_level_first_non_anchor_pos": first_non_anchor,
            "lb_over_sep_minus_1_sq": lb_direct / sep_direct - 1.0,
        })

    json.dump(rows, open(f"{OUT}/levels_direct{TAG}.json", "w"), indent=2)
    print("lvl  size     sep_d(sqrt)     LB_d(sqrt)   LB^2/sep^2-1   LBpos  anchPrefix anchMembers LBisAnchor  (exp-direct)/direct sep,LB")
    for r in rows:
        print(f"{r['level']:2d} {r['size']:8d} {np.sqrt(r['sep_sq_direct']):.8f} {np.sqrt(r['lb_sq_direct']):.8f} "
              f"{r['lb_over_sep_minus_1_sq']:+.3e} {r['lb_pos_in_next_level']:6d} {r['next_level_anchor_prefix_len']:6d} "
              f"{r['next_level_anchor_members']:6d} {str(r['lb_pick_is_anchor']):5s} "
              f"{r['sep_sq_expanded']/r['sep_sq_direct']-1:+.1e} {r['lb_sq_expanded']/r['lb_sq_direct']-1:+.1e}")


if __name__ == "__main__":
    main()
