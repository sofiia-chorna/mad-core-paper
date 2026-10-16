import json
import os
from multiprocessing import Pool

import numpy as np

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description="see the README")
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
parser.add_argument("--coverage-cert", required=True, help="the out/ folder of the coverage certificate with its per-folder population/ arrays (2.9 GB, development tree, not part of this release)")
parser.add_argument("workers", nargs="?", type=int, default=48)
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
COVERAGE_CERT = ARGS.coverage_cert
CERTIFICATE = f"{COVERAGE_CERT}/stageB/coverage_exact.json"
POPULATION = f"{COVERAGE_CERT}/population"
CLAIMS_OUT = os.path.dirname(ARGS.out) if ARGS.out else os.path.join(HERE, "out")
WITHIN_CLAIM = f"{CLAIMS_OUT}/collection_within_coverage.json"
SCAN_OUT = ARGS.out or f"{CLAIMS_OUT}/outside_coverage_scan.json"
ROWS_OUT = f"{CLAIMS_OUT}/outside_coverage_rows.npz"

LEVELS = 25
LEVEL = 24
BLOCK_ROWS = 1 << 22
RELEASED_ROWS = 1694502355
N_CATEGORIES = 16
FILTER_NAMES = ["passed", "zbl_rejected", "nan_features", "nomad_excluded"]

RADIUS = json.load(open(CERTIFICATE))["release"][str(LEVEL)]["cov_exact_direct"]


def category_names():
    names = []

    for code in range(4):
        for duplicate in ("unique", "duplicate"):
            for pool in ("not_in_pool", "in_pool"):
                names.append(f"{FILTER_NAMES[code]}|{duplicate}|{pool}")

    return names


def folder_rows(folder):
    if folder.endswith("mad-next-coreset"):
        return np.load(f"{folder}/configuration_id.npy", mmap_mode="r").shape[0]

    return int(np.load(f"{folder}/ns.npy")[0])


def add_counts(total, counts):
    if counts.size > total.size:
        total = np.pad(total, (0, counts.size - total.size))

    total[:counts.size] += counts
    return total


def add_dataset_counts(counts, column, values):
    if values.size == 0:
        return

    if (values == values[0]).all():
        counts.setdefault(values[0].decode(), [0, 0, 0])[column] += int(values.size)
        return

    keys, sizes = np.unique(values, return_counts=True)
    for key, size in zip(keys, sizes):
        counts.setdefault(key.decode(), [0, 0, 0])[column] += int(size)


def scan_folder(folder):
    name = os.path.basename(folder)
    is_anchor = name == "mad-next-coreset"
    n_rows = folder_rows(folder)

    distances = np.memmap(f"{folder}/voronoi_d.bin", dtype=np.float32, mode="r", shape=(n_rows, LEVELS))
    atom_offsets = np.load(f"{folder}/na.npy", mmap_mode="r")
    dataset_ids = np.load(f"{folder}/dataset_id.npy", mmap_mode="r")

    # the anchor tier carries no filter or duplicate files and is the pool by construction
    if is_anchor:
        filter_codes = np.zeros(n_rows, dtype=np.int8)
        duplicates = np.zeros(n_rows, dtype=bool)
        pool_rows = np.load(f"{POPULATION}/mad-next-coreset.npy")
    else:
        filter_codes = np.load(f"{folder}/filter_code.npy", mmap_mode="r")
        duplicates = np.load(f"{folder}/is_duplicate.npy", mmap_mode="r")
        pool_rows = np.load(f"{POPULATION}/collection__{name}.npy")

    in_pool = np.zeros(n_rows, dtype=bool)
    in_pool[pool_rows] = True

    image_flag_path = f"{folder}/image_flag_rows.npy"
    image_flag_rows = np.load(image_flag_path) if os.path.exists(image_flag_path) else np.zeros(0, dtype=np.int64)

    all_categories = np.zeros(N_CATEGORIES, dtype=np.int64)
    outside_categories = np.zeros(N_CATEGORIES, dtype=np.int64)
    nonfinite_categories = np.zeros(N_CATEGORIES, dtype=np.int64)
    all_atoms = np.zeros(0, dtype=np.int64)
    dataset_counts = {}
    nonfinite_image_flag = 0

    parts = {key: [] for key in ("row", "distance", "filter_code", "is_duplicate", "in_pool", "n_atoms", "dataset_id")}

    for start in range(0, n_rows, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n_rows)

        block_distance = np.asarray(distances[start:stop, LEVEL])
        block_code = np.asarray(filter_codes[start:stop])
        block_duplicate = np.asarray(duplicates[start:stop])
        block_pool = in_pool[start:stop]
        block_atoms = np.diff(np.asarray(atom_offsets[start:stop + 1]))
        block_dataset = np.asarray(dataset_ids[start:stop])

        finite = np.isfinite(block_distance)
        outside = finite & (block_distance.astype(np.float64) > RADIUS)

        category = (
            block_code.astype(np.int64) * 4
            + block_duplicate.astype(np.int64) * 2
            + block_pool.astype(np.int64)
        )

        all_categories += np.bincount(category, minlength=N_CATEGORIES)
        outside_categories += np.bincount(category[outside], minlength=N_CATEGORIES)
        nonfinite_categories += np.bincount(category[~finite], minlength=N_CATEGORIES)

        all_atoms = add_counts(all_atoms, np.bincount(block_atoms))

        add_dataset_counts(dataset_counts, 0, block_dataset)
        add_dataset_counts(dataset_counts, 1, block_dataset[outside])
        add_dataset_counts(dataset_counts, 2, block_dataset[~finite])

        nonfinite_rows = np.flatnonzero(~finite) + start
        nonfinite_image_flag += int(np.isin(nonfinite_rows, image_flag_rows).sum())

        parts["row"].append(np.flatnonzero(outside) + start)
        parts["distance"].append(block_distance[outside])
        parts["filter_code"].append(block_code[outside])
        parts["is_duplicate"].append(block_duplicate[outside])
        parts["in_pool"].append(block_pool[outside])
        parts["n_atoms"].append(block_atoms[outside].astype(np.int32))
        parts["dataset_id"].append(block_dataset[outside])

    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    arrays["image_flag"] = np.isin(arrays["row"], image_flag_rows)

    summary = {
        "rows": n_rows,
        "nonfinite": int(nonfinite_categories.sum()),
        "outside_finite": int(outside_categories.sum()),
        "nonfinite_image_flag": nonfinite_image_flag,
        "image_flag_rows": int(image_flag_rows.size),
        "pool_rows": int(pool_rows.size),
        "categories_all": all_categories.tolist(),
        "categories_outside": outside_categories.tolist(),
        "categories_nonfinite": nonfinite_categories.tolist(),
        "dataset_id_rows_outside_nonfinite": dataset_counts,
        "mean_atoms": float(atom_offsets[-1]) / n_rows,
    }

    print(f"{name} rows {n_rows} outside {summary['outside_finite']} nonfinite {summary['nonfinite']}", flush=True)
    return name, summary, all_atoms, arrays


def main():
    folders = [f"{RELEASE}/collection/{name}" for name in os.listdir(f"{RELEASE}/collection")]
    folders.append(f"{RELEASE}/mad-next-coreset")

    # largest folders first so the long ones start early
    folders.sort(key=folder_rows, reverse=True)

    workers = ARGS.workers
    with Pool(workers) as pool:
        results = list(pool.imap_unordered(scan_folder, folders, chunksize=1))

    results.sort(key=lambda result: result[0])
    folder_names = [result[0] for result in results]

    total_rows = sum(result[1]["rows"] for result in results)
    total_outside = sum(result[1]["outside_finite"] for result in results)
    total_nonfinite = sum(result[1]["nonfinite"] for result in results)
    total_within = total_rows - total_outside - total_nonfinite

    # must reproduce the reported coverage run
    reported = json.load(open(WITHIN_CLAIM))
    assert total_rows == RELEASED_ROWS, total_rows
    assert total_nonfinite == reported["n_nonfinite_distance"], total_nonfinite
    assert total_within == reported["within"]["certified_exact"]["count"], total_within

    atom_bins = max(result[2].size for result in results)
    atoms_all = np.zeros((len(results), atom_bins), dtype=np.int64)
    for index, result in enumerate(results):
        atoms_all[index, :result[2].size] = result[2]

    folder_index = np.concatenate([
        np.full(result[3]["row"].size, index, dtype=np.int16)
        for index, result in enumerate(results)
    ])

    def gather(key):
        return np.concatenate([result[3][key] for result in results])

    dataset_names, dataset_index = np.unique(gather("dataset_id"), return_inverse=True)

    np.savez(
        ROWS_OUT,
        radius=np.float64(RADIUS),
        folder_names=np.array(folder_names),
        folder_index=folder_index,
        row=gather("row"),
        distance=gather("distance"),
        filter_code=gather("filter_code"),
        is_duplicate=gather("is_duplicate"),
        in_pool=gather("in_pool"),
        n_atoms=gather("n_atoms"),
        image_flag=gather("image_flag"),
        dataset_names=dataset_names,
        dataset_index=dataset_index.astype(np.int32),
        atoms_all_rows=atoms_all,
    )

    scan = {
        "radius": RADIUS,
        "radius_source": "coverage_exact.json release level 24 cov_exact_direct",
        "rows": total_rows,
        "within": total_within,
        "outside_finite": total_outside,
        "nonfinite": total_nonfinite,
        "category_names": category_names(),
        "dataset_id_columns": ["rows", "outside_finite", "nonfinite"],
        "folders": {result[0]: result[1] for result in results},
    }

    with open(SCAN_OUT, "w") as f:
        json.dump(scan, f, indent=1)

    print(json.dumps({key: value for key, value in scan.items() if key != "folders"}, indent=1))

if __name__ == "__main__":
    main()
