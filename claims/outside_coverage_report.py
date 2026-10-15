import json
import pickle

import numpy as np
import pyarrow.parquet as pq

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--store-root", required=True, help="the aa002 mad-core store, holding release/, fps/ and models/")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
ARGS = parser.parse_args()
STORE = ARGS.store_root
FAMILY_FPS = f"{STORE}/fps/phace_atomic_ep92"
FAMILY_SCALER = f"{STORE}/models/scaler_mad15_phaceatomic_meanstd.pkl"
RELEASE_SCALER = f"{STORE}/release/ecumetric_scaler.npz"
CLAIMS_OUT = os.path.dirname(ARGS.out) if ARGS.out else os.path.join(HERE, "out")
WITHIN_CLAIM = f"{CLAIMS_OUT}/collection_within_coverage.json"
SCAN = f"{CLAIMS_OUT}/outside_coverage_scan.json"
ROWS = f"{CLAIMS_OUT}/outside_coverage_rows.npz"
OUT = ARGS.out or f"{CLAIMS_OUT}/outside_coverage.json"

RECORD_ROWS = 1694434380
TOP_FOLDERS = 15
TOP_DATASETS = 5
PERCENTILES = [50, 90, 99]
RATIO_EDGES = [1.0, 1.001, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0, 5.0, np.inf]
TAIL_RATIOS = [1.1, 1.5, 2.0, 5.0]
LARGE_ATOM_PERCENTILE = 99
FILTER_NAMES = ["passed", "zbl_rejected", "nan_features", "nomad_excluded"]
FPS_NOISE = 1e-3


def percent(part, whole):
    return 100.0 * part / whole if whole else None


def ratio_summary(ratio):
    if ratio.size == 0:
        return {"n": 0}

    summary = {"n": int(ratio.size)}
    for value, level in zip(np.percentile(ratio, PERCENTILES), PERCENTILES):
        summary[f"p{level}"] = float(value)

    summary["max"] = float(ratio.max())
    return summary


def ratio_bins(ratio):
    counts = np.histogram(ratio, bins=RATIO_EDGES)[0]
    bins = {}

    for low, high, count in zip(RATIO_EDGES[:-1], RATIO_EDGES[1:], counts):
        bins[f"({low}, {high}]"] = {"count": int(count), "percent": percent(int(count), ratio.size)}

    return bins


def histogram_percentile(histogram, level):
    cumulative = np.cumsum(histogram)
    return int(np.searchsorted(cumulative, level / 100.0 * cumulative[-1]))


def family_fps_radii():
    family_scale = float(pickle.load(open(FAMILY_SCALER, "rb"))["scale_"])
    release_scale = float(np.load(RELEASE_SCALER)["scale"])
    radii = {}

    for family in ["ANI-1", "ANI-2x", "Alexandria", "OC20-IS2RES", "OC20_S2EF_train_all", "ODAC", "OMat24", "OMol25", "Open_Molecular_Crystals_2025_OMC25", "gdb13"]:
        meta = json.load(open(f"{FAMILY_FPS}/{family}/fps.parquet.meta.json"))
        epsilon_sq = pq.read_table(f"{FAMILY_FPS}/{family}/fps.parquet", columns=["epsilon_sq"]).column("epsilon_sq").to_numpy()

        # the mean cancels in differences so only the scalar scale differs
        radius_family_metric = float(np.sqrt(epsilon_sq[-1]))
        radii[family] = {
            "picks": int(epsilon_sq.size),
            "fps_input_rows": meta["n_structures"],
            "fps_input_dropped": meta["n_dropped_nonfinite"] + meta["n_dropped_maxforce"] + meta["n_dropped_energy"],
            "radius_family_scaler": radius_family_metric,
            "radius_release_scaler": radius_family_metric * family_scale / release_scale,
        }

    return radii


def main():
    scan = json.load(open(SCAN))
    rows = np.load(ROWS)
    reported = json.load(open(WITHIN_CLAIM))

    radius = float(rows["radius"])
    folder_names = rows["folder_names"].astype(str)
    folder_index = rows["folder_index"]
    ratio = rows["distance"].astype(np.float64) / radius
    filter_code = rows["filter_code"]
    is_duplicate = rows["is_duplicate"]
    in_pool = rows["in_pool"]
    n_atoms = rows["n_atoms"]
    image_flag = rows["image_flag"]
    dataset_names = rows["dataset_names"].astype(str)
    dataset_index = rows["dataset_index"]
    atoms_all_rows = rows["atoms_all_rows"]
    folders = scan["folders"]

    n_rows = scan["rows"]
    n_outside = scan["outside_finite"]
    n_nonfinite = scan["nonfinite"]
    n_not_within = n_outside + n_nonfinite
    assert ratio.size == n_outside

    report = {"radius": radius, "radius_source": scan["radius_source"]}

    # item 1 totals
    report["totals"] = {
        "rows": n_rows,
        "within": scan["within"],
        "within_percent": percent(scan["within"], n_rows),
        "not_within": n_not_within,
        "not_within_percent": percent(n_not_within, n_rows),
        "outside_finite_distance": n_outside,
        "outside_finite_distance_percent": percent(n_outside, n_rows),
        "nonfinite_distance": n_nonfinite,
        "sensitivity_construction_log_radius": {
            "radius": reported["radius"]["construction_log_float32"],
            "within": reported["within"]["construction_log_float32"]["count"],
            "within_percent": reported["within"]["construction_log_float32"]["percent"],
        },
        "record_without_nonfinite": {
            "rows": RECORD_ROWS,
            "within_percent": percent(scan["within"], RECORD_ROWS),
            "outside_percent": percent(n_outside, RECORD_ROWS),
        },
    }
    assert n_rows - n_nonfinite == RECORD_ROWS

    # item 2 contributing folders and their dataset ids
    folder_table = []
    for name, folder in folders.items():
        not_within = folder["outside_finite"] + folder["nonfinite"]
        folder_table.append({
            "folder": name,
            "rows": folder["rows"],
            "outside_finite": folder["outside_finite"],
            "nonfinite": folder["nonfinite"],
            "not_within": not_within,
            "share_of_all_not_within_percent": percent(not_within, n_not_within),
            "share_of_all_outside_finite_percent": percent(folder["outside_finite"], n_outside),
            "folder_not_within_percent": percent(not_within, folder["rows"]),
        })

    folder_table.sort(key=lambda entry: entry["not_within"], reverse=True)

    cumulative = 0
    for entry in folder_table:
        cumulative += entry["not_within"]
        entry["cumulative_share_percent"] = percent(cumulative, n_not_within)

    position = {name: index for index, name in enumerate(folder_names)}
    for entry in folder_table[:TOP_FOLDERS]:
        in_folder = folder_index == position[entry["folder"]]
        entry["ratio"] = ratio_summary(ratio[in_folder])

        datasets = folders[entry["folder"]]["dataset_id_rows_outside_nonfinite"]
        dataset_table = [
            {
                "dataset_id": dataset,
                "rows": counts[0],
                "not_within": counts[1] + counts[2],
                "share_of_folder_not_within_percent": percent(counts[1] + counts[2], entry["not_within"]),
                "dataset_not_within_percent": percent(counts[1] + counts[2], counts[0]),
            }
            for dataset, counts in datasets.items()
        ]
        dataset_table.sort(key=lambda item: item["not_within"], reverse=True)
        entry["dataset_ids"] = dataset_table[:TOP_DATASETS]

    report["folders_top"] = folder_table[:TOP_FOLDERS]
    report["folders_with_any_not_within"] = sum(entry["not_within"] > 0 for entry in folder_table)
    report["folders_total"] = len(folder_table)

    # item 3 distance over radius among finite outside rows
    report["ratio"] = ratio_summary(ratio)
    report["ratio_without_image_flag_rows"] = ratio_summary(ratio[~image_flag])

    # same percentiles with the non-finite rows counted as infinitely far
    with_nonfinite = np.concatenate([ratio, np.full(n_nonfinite, np.inf)])
    report["ratio_nonfinite_as_infinite"] = {
        f"p{level}": float(np.percentile(with_nonfinite, level, method="lower"))
        for level in PERCENTILES
    }

    report["ratio_bins"] = ratio_bins(ratio)

    # item 4 flags duplicates pool membership and size
    names = scan["category_names"]
    category = filter_code.astype(np.int64) * 4 + is_duplicate.astype(np.int64) * 2 + in_pool.astype(np.int64)
    category_counts = np.bincount(category, minlength=len(names))
    category_all = np.sum([folder["categories_all"] for folder in folders.values()], axis=0)
    category_nonfinite = np.sum([folder["categories_nonfinite"] for folder in folders.values()], axis=0)

    report["categories"] = {
        name: {
            "rows": int(category_all[index]),
            "outside_finite": int(category_counts[index]),
            "nonfinite": int(category_nonfinite[index]),
            "outside_finite_percent_of_category": percent(int(category_counts[index]), int(category_all[index])),
            "ratio": ratio_summary(ratio[category == index]),
        }
        for index, name in enumerate(names)
        if category_all[index]
    }

    flagged = filter_code != 0
    report["flags"] = {
        "outside_flagged": int(flagged.sum()),
        "outside_flagged_percent": percent(int(flagged.sum()), n_outside),
        "outside_by_filter_code": {
            FILTER_NAMES[code]: int((filter_code == code).sum()) for code in range(4)
        },
        "outside_duplicate": int(is_duplicate.sum()),
        "outside_in_pool": int(in_pool.sum()),
        "outside_passed_unique_not_in_pool": int((~flagged & ~is_duplicate & ~in_pool).sum()),
        "outside_image_flag": int(image_flag.sum()),
        "nonfinite_image_flag": sum(folder["nonfinite_image_flag"] for folder in folders.values()),
        "ratio_flagged": ratio_summary(ratio[flagged]),
        "ratio_passed": ratio_summary(ratio[~flagged]),
    }

    atoms_total = atoms_all_rows.sum(axis=0)
    large_threshold = histogram_percentile(atoms_total, LARGE_ATOM_PERCENTILE)
    large = n_atoms > large_threshold

    report["atoms"] = {
        "all_rows_percentiles": {f"p{level}": histogram_percentile(atoms_total, level) for level in PERCENTILES},
        "all_rows_max": int(np.flatnonzero(atoms_total)[-1]),
        "outside_percentiles": {f"p{level}": float(value) for value, level in zip(np.percentile(n_atoms, PERCENTILES), PERCENTILES)},
        "outside_max": int(n_atoms.max()),
        "large_threshold_atoms": large_threshold,
        "large_definition": f"more atoms than the p{LARGE_ATOM_PERCENTILE} of all release rows",
        "all_rows_large_percent": percent(int(atoms_total[large_threshold + 1:].sum()), int(atoms_total.sum())),
        "outside_large_percent": percent(int(large.sum()), n_outside),
        "ratio_large": ratio_summary(ratio[large]),
        "ratio_not_large": ratio_summary(ratio[~large]),
    }

    # composition of the tail beyond each ratio
    tail = {}
    for threshold in TAIL_RATIOS:
        beyond = ratio > threshold
        n_beyond = int(beyond.sum())
        tail_folders = np.bincount(folder_index[beyond], minlength=len(folder_names))
        top = np.argsort(tail_folders)[::-1][:5]

        tail[f"ratio_above_{threshold}"] = {
            "count": n_beyond,
            "percent_of_outside": percent(n_beyond, n_outside),
            "flagged_percent": percent(int((beyond & flagged).sum()), n_beyond),
            "large_percent": percent(int((beyond & large).sum()), n_beyond),
            "flagged_or_large_percent": percent(int((beyond & (flagged | large)).sum()), n_beyond),
            "image_flag": int((beyond & image_flag).sum()),
            "top_folders": {folder_names[index]: int(tail_folders[index]) for index in top if tail_folders[index]},
        }

    report["tail"] = tail

    # bound from the per-family FPS prefix that entered the pool
    radii = family_fps_radii()
    bound_check = {}
    for family, family_radius in radii.items():
        in_family = (folder_index == position[family]) & ~flagged & ~in_pool
        bound = 1.0 + family_radius["radius_release_scaler"] / radius
        family_ratio = ratio[in_family]

        bound_check[family] = {
            **family_radius,
            "ratio_bound": bound,
            "outside_passed_not_in_pool": int(in_family.sum()),
            "ratio_max": float(family_ratio.max()) if family_ratio.size else None,
            "above_bound": int((family_ratio > bound * (1.0 + FPS_NOISE)).sum()),
        }

    report["family_fps_bound"] = bound_check

    family_positions = [position[family] for family in radii]
    big = np.isin(folder_index, family_positions)
    report["big_family_share"] = {
        "outside_finite": int(big.sum()),
        "percent_of_outside": percent(int(big.sum()), n_outside),
        "passed_not_in_pool": int((big & ~flagged & ~in_pool).sum()),
        "passed_not_in_pool_percent_of_outside": percent(int((big & ~flagged & ~in_pool).sum()), n_outside),
    }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
