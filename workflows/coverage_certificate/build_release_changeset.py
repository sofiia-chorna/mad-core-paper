import hashlib
import json
import os

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release/mad-core"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
STAGE_C = f"{HERE}/out/stageC"
STAGING = f"{HERE}/out/release_changeset"
N = 16777216
N_ANCHOR = 856012

TABLE_COV_F32 = {1: 23.080, 2: 14.613, 3: 12.233, 4: 9.186, 5: 6.960, 6: 5.508, 7: 4.613, 8: 3.833, 9: 3.227,
                 10: 2.733, 11: 2.318, 12: 1.939, 13: 1.603, 14: 1.319, 15: 1.103, 16: 0.941, 17: 0.815,
                 18: 0.710, 19: 0.617, "anchor": 0.556}
TABLE_MARGIN = {1: 8.6e-2, 2: 5.7e-1, 3: 5.0e-2, 4: 1.4e-2, 5: 1.5e-2, 6: 2.0e-3, 7: 8.6e-4, 8: 1.8e-3, 9: 4.8e-4,
                10: 2.8e-4, 11: 3.2e-4, 12: 6.5e-5, 13: 8.5e-6, 14: 3.0e-5, 15: 1.6e-5, 16: 2.7e-6, 17: 5.7e-6,
                18: 2.8e-6, 19: 9.4e-7, "anchor": 1.5e-6}
TABLE_FROM_A = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 4, 8: 8, 9: 17, 10: 36, 11: 88, 12: 224, 13: 680,
                14: 2056, 15: 5384, 16: 11785, 17: 22364, 18: 38355, 19: 62967, "anchor": 86910}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 26), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(STAGING, exist_ok=True)
    sep_rel = np.load(f"{STAGE_C}/separation_sq_release_full.npy")
    sep_sel = np.load(f"{STAGE_C}/separation_sq_selection_full.npy")
    nearest = np.load(f"{STAGE_C}/nearest_prev_release_full.npy")
    stored = np.load(f"{RELEASE}/epsilon_sq.npy")
    levels_full = json.load(open(f"{STAGE_C}/levels_full.json"))
    release_directory = np.load(f"{RELEASE}/release_directory.npy")
    names = np.load(f"{RELEASE}/release_directory_names.npy")
    anchor_dir = int(np.where(names == b"mad-next-coreset")[0][0])
    anchor_cum = np.cumsum(release_directory == anchor_dir)
    expanded_vs_direct = max(abs(e["sep_sq_expanded_cummin"] / e["sep_sq_direct"] - 1.0) for e in levels_full["release"]["levels"])
    assert sep_rel.shape == (N,) and np.isinf(sep_rel[0]) and np.all(sep_rel[1:] > 0)

    np.save(f"{STAGING}/epsilon_sq.npy", sep_rel)
    np.save(f"{STAGING}/nearest_earlier_pick.npy", nearest)

    cummin_rel = np.minimum.accumulate(sep_rel)
    cummin_sel = np.minimum.accumulate(sep_sel)
    stored_cont_rel = (sep_rel[N_ANCHOR:] - stored[N_ANCHOR:]) / stored[N_ANCHOR:]
    rows = []
    for entry in levels_full["release"]["levels"]:
        size = entry["size"]
        key = entry["level"] if entry["level"] != "anchor" else "anchor"
        sel_entry = next(e for e in levels_full["selection"]["levels"] if e["size"] == size)
        cov_lb = np.sqrt(entry["lb_sq_direct_next_level_max"]) if size < N else None
        row = {
            "level": entry["level"], "size": size,
            "from_anchor": int(anchor_cum[size - 1]),
            "from_anchor_construction_log": TABLE_FROM_A.get(key),
            "separation_release_features": float(np.sqrt(cummin_rel[size - 1])),
            "separation_selection_features": float(np.sqrt(cummin_sel[size - 1])),
            "separation_pair_rows": entry["sep_pair"],
            "coverage_lower_bound_release_features": cov_lb,
            "coverage_lower_bound_witness_row": entry["lb_pick"] if size < N else None,
            "coverage_runtime_float32_construction_log": TABLE_COV_F32.get(key) if size <= N_ANCHOR else float(np.sqrt(stored[size])) if size < N else 0.2179064154624939,
            "coverage_runtime_float32_source": ("appendix table (run log deleted)" if size <= N_ANCHOR else
                                               "float32 FPS insertion distance of pick size+1 (release epsilon_sq)" if size < N else
                                               "float32 FPS insertion distance of pick 2^24+1 from the unreleased 25M continuation"),
            "coverage_runtime_float32_reconstructed": (float(np.sqrt(stored[size - 1]) / np.sqrt(1.0 + TABLE_MARGIN[key]))
                                                       if key in TABLE_MARGIN else None),
            "coverage_runtime_float32_reconstruction_note": ("stored float32 running-min separation divided by sqrt(1 + margin), margin = appendix column sep^2/cov^2 - 1 (2 significant digits, so the reconstruction is good to ~5% of the margin)"
                                                             if key in TABLE_MARGIN else None),
            "coverage_exact_upper_bound": None,
            "coverage_status": "pending: assigned-center pass over the 184,180,658-row pool not yet run",
            "population": "A ∪ C, 184,180,658 rows (856,012 anchor + 183,324,646 candidates)",
            "exact_inequality_sep_ge_cov": ("violated" if (cov_lb is not None and cov_lb > np.sqrt(cummin_rel[size - 1]))
                                            else "undetermined"),
            "cov_over_sep_lower_bound": (cov_lb / np.sqrt(cummin_rel[size - 1])) if cov_lb is not None else None,
            "implemented_acceptance": "sep^2 >= 0.999 cov^2 (cov/sep <= 1.0005004), float32 at run time",
        }
        rows.append(row)

    levels = {
        "definition": {
            "epsilon_sq": "epsilon_sq[i] = min_{j<i} ||s_i - s_j||^2 with s = (ecumetric_features - mean)/scale of the released rows; float64 direct differences; epsilon_sq[0] = +inf",
            "separation": "sep(S_n) = sqrt(min(epsilon_sq[1:n])), the minimum pairwise distance of the first n rows",
            "coverage": "cov_X(S) = max over the accrete pool X of the distance to the nearest member of S; not derivable from this column",
            "numerical_uncertainty": f"float64 arithmetic on the released float32 inputs (mean cancels in differences, scale applied once in float64); the expanded-form values agree with direct differences at every level's argmin pair to {expanded_vs_direct:.1e} relative; selection-time and released features differ by featurizer rerun noise (scaled shift <= 1.8e-5 per row)",
        },
        "continuation_float32_vs_exact": {
            "rows": [N_ANCHOR, N],
            "rel_diff_min": float(stored_cont_rel.min()), "rel_diff_median": float(np.median(stored_cont_rel)),
            "rel_diff_max": float(stored_cont_rel.max()),
        },
        "levels": rows,
    }
    json.dump(levels, open(f"{STAGING}/levels.json", "w"), indent=1)

    manifest_entry = {"epsilon_sq.npy": {"bytes": os.path.getsize(f"{STAGING}/epsilon_sq.npy"),
                                         "sha256": sha256(f"{STAGING}/epsilon_sq.npy"), "shape": [N], "dtype": "float64"},
                      "nearest_earlier_pick.npy": {"bytes": os.path.getsize(f"{STAGING}/nearest_earlier_pick.npy"),
                                                   "sha256": sha256(f"{STAGING}/nearest_earlier_pick.npy"), "shape": [N], "dtype": "int64"},
                      "levels.json": {"bytes": os.path.getsize(f"{STAGING}/levels.json"), "sha256": sha256(f"{STAGING}/levels.json")}}
    json.dump(manifest_entry, open(f"{STAGING}/manifest_entries.json", "w"), indent=1)

    format_note = """| epsilon_sq.npy | (ns,) | float64 | squared distance from row i to its nearest earlier row (min over j < i), in the scaled ecumetric metric, recomputed in float64 from the released features; +inf at row 0. The running minimum over rows 0..n-1 is the separation of the first n picks. Inside the accrete-anchor block (rows < 856,012) the sequence is not monotone: each accretion level opens with anchor points admitted farthest-first within the anchor block, then plain FPS. Not an interatomic distance |
| nearest_earlier_pick.npy | (ns,) | int64 | row index of the nearest earlier pick that realises epsilon_sq; -1 at row 0 |
| levels.json | | | per-level table: size, anchor members, separation (exact), coverage over the 184,180,658-row accrete pool (run-time float32 value from the construction log, plus the exact lower bound and, when available, the exact upper bound), status of the separation-coverage check. The separation-coverage condition sep(S_k) >= cov_X(S_k) refers to the accrete pool X, not to the full collection, and is asserted at completed levels only |
"""
    open(f"{STAGING}/FORMAT_rows.md", "w").write(format_note)
    print(json.dumps(manifest_entry, indent=1))
    print("expanded vs direct max rel diff at argmin pairs:", f"{expanded_vs_direct:.2e}")
    for r in rows:
        print(r["level"], r["size"], f"sep_rel={r['separation_release_features']:.8f}", f"sep_sel={r['separation_selection_features']:.8f}",
              f"cov_lb={r['coverage_lower_bound_release_features']}", r["exact_inequality_sep_ge_cov"])


if __name__ == "__main__":
    main()
