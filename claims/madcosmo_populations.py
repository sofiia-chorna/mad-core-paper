import json
import os
import shutil
from multiprocessing import Pool

import numpy as np

import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description="see the README")
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
parser.add_argument("--certificate", default=CERTIFICATE_IN_RELEASE, help="coverage_exact.json of the coverage certificate")
parser.add_argument("--analysis", required=True, help="folder with the pre-release population arrays populations.npz and populations_level20.npy (voronoi/out/analysis of the development tree, not part of this release)")
parser.add_argument("--work", default=None, help="scratch folder for the duplicate matching (default: tmp_madcosmo_populations next to the output)")
parser.add_argument("workers", nargs="?", type=int, default=32)
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
OUT = ARGS.out or os.path.join(HERE, "out", "madcosmo_populations.json")
WORK = ARGS.work or os.path.join(os.path.dirname(OUT), "tmp_madcosmo_populations")
CERTIFICATE = ARGS.certificate
ANALYSIS = ARGS.analysis

ANCHOR = "mad-next-coreset"
LEVELS = 25
LEVEL = 24
N_CELLS = 1 << 24
CHUNK_ROWS = 1 << 24
ID_BLOCK_ROWS = 1 << 22
MAX_ATOMS = 300
SKETCHMAP_WGAMMA = 0.25

ZBL_FORCE = 200.0
ZBL_ENERGY_PER_ATOM = 50.0
ZBL_ATOMIC_ENERGY = 100.0

RELEASED_ROWS = 1694502355
COLLECTION_ROWS = 1693646343

# category is filter_code times two plus is_duplicate
CATEGORY_NAMES = [
    "passed",
    "passed_duplicate",
    "zbl_rejected",
    "zbl_rejected_duplicate",
    "nan_features",
    "nan_features_duplicate",
    "nomad_excluded",
    "nomad_excluded_duplicate",
    "anchor",
]
N_CATEGORIES = len(CATEGORY_NAMES)

HEX_VALUES = np.zeros(256, dtype=np.uint64)
HEX_VALUES[np.frombuffer(b"0123456789", dtype=np.uint8)] = np.arange(10, dtype=np.uint64)
HEX_VALUES[np.frombuffer(b"abcdef", dtype=np.uint8)] = np.arange(10, 16, dtype=np.uint64)


def folder_rows(path):
    if path.endswith(ANCHOR):
        return np.load(f"{path}/configuration_id.npy", mmap_mode="r").shape[0]

    return int(np.load(f"{path}/ns.npy")[0])


def tally_chunk(task):
    folder, start, stop, radius = task
    path = f"{RELEASE}/{folder}"
    n_rows = folder_rows(path)
    rows = slice(start, stop)

    if folder == ANCHOR:
        category = np.full(stop - start, 8, dtype=np.int64)
    else:
        codes = np.asarray(np.load(f"{path}/filter_code.npy", mmap_mode="r")[rows], dtype=np.int64)
        duplicate = np.asarray(np.load(f"{path}/is_duplicate.npy", mmap_mode="r")[rows], dtype=np.int64)
        category = codes * 2 + duplicate

    atom_offsets = np.asarray(np.load(f"{path}/na.npy", mmap_mode="r")[start:stop + 1], dtype=np.int64)
    natoms = np.diff(atom_offsets)

    force = np.asarray(np.load(f"{path}/zbl_max_force.npy", mmap_mode="r")[rows])
    energy_per_atom = np.asarray(np.load(f"{path}/zbl_energy_per_atom.npy", mmap_mode="r")[rows])
    atomic_energy = np.asarray(np.load(f"{path}/zbl_max_atomic_energy.npy", mmap_mode="r")[rows])

    with np.errstate(invalid="ignore"):
        zbl_rule = (
            (force >= ZBL_FORCE)
            | (energy_per_atom > ZBL_ENERGY_PER_ATOM)
            | (atomic_energy > ZBL_ATOMIC_ENERGY)
        )

    distances = np.memmap(f"{path}/voronoi_d.bin", dtype=np.float32, mode="r", shape=(n_rows, LEVELS))
    d24 = np.asarray(distances[rows, LEVEL], dtype=np.float64)
    finite = np.isfinite(d24)
    within = np.zeros(len(d24), dtype=bool)
    within[finite] = d24[finite] <= radius

    tallies = {
        "rows": np.bincount(category, minlength=N_CATEGORIES),
        "natoms_below_300": np.bincount(category[natoms < MAX_ATOMS], minlength=N_CATEGORIES),
        "natoms_above_1000": np.bincount(category[natoms > 1000], minlength=N_CATEGORIES),
        "zbl_rule": np.bincount(category[zbl_rule], minlength=N_CATEGORIES),
        "nonfinite_d24": np.bincount(category[~finite], minlength=N_CATEGORIES),
        "within_radius_d24": np.bincount(category[within], minlength=N_CATEGORIES),
    }

    return folder, {name: values.tolist() for name, values in tallies.items()}


def hash_keys(path, n_rows):
    ids = np.load(f"{path}/configuration_id.npy", mmap_mode="r")
    width = ids.dtype.itemsize

    high = np.empty(n_rows, dtype=np.uint64)
    low = np.empty(n_rows, dtype=np.uint64)

    for start in range(0, n_rows, ID_BLOCK_ROWS):
        block = np.ascontiguousarray(ids[start:start + ID_BLOCK_ROWS])
        raw = np.frombuffer(block.tobytes(), dtype=np.uint8).reshape(len(block), width)

        # every id in a folder is prefix plus 32 hex digits
        assert (raw[:, -1] != 0).all(), path
        assert (raw[:, -33] == ord(":")).all(), path

        nibbles = HEX_VALUES[raw[:, -32:]]
        block_high = np.zeros(len(block), dtype=np.uint64)
        block_low = np.zeros(len(block), dtype=np.uint64)

        for j in range(16):
            block_high = (block_high << np.uint64(4)) | nibbles[:, j]
            block_low = (block_low << np.uint64(4)) | nibbles[:, 16 + j]

        high[start:start + len(block)] = block_high
        low[start:start + len(block)] = block_low

    return high, low


def dataset_codes(path, n_rows):
    dataset_ids = np.load(f"{path}/dataset_id.npy", mmap_mode="r")
    first = dataset_ids[0]

    same_as_first = all(
        bool((np.asarray(dataset_ids[start:start + ID_BLOCK_ROWS]) == first).all())

        for start in range(0, n_rows, ID_BLOCK_ROWS)
    )

    if same_as_first:
        return np.zeros(n_rows, dtype=np.int32), 1

    vocabulary = {}
    codes = np.empty(n_rows, dtype=np.int32)

    for start in range(0, n_rows, ID_BLOCK_ROWS):
        values, inverse = np.unique(np.asarray(dataset_ids[start:start + ID_BLOCK_ROWS]), return_inverse=True)
        lookup = np.array([vocabulary.setdefault(value, len(vocabulary)) for value in values], dtype=np.int32)
        codes[start:start + len(inverse)] = lookup[inverse]

    return codes, len(vocabulary)


def duplicate_scope(task):
    index, folder = task
    path = f"{RELEASE}/{folder}"
    n_rows = folder_rows(path)

    duplicate = np.load(f"{path}/is_duplicate.npy")
    high, low = hash_keys(path, n_rows)
    codes, n_dataset_ids = dataset_codes(path, n_rows)

    original_rows = np.flatnonzero(~duplicate)
    order = np.argsort(high[original_rows], kind="stable")
    sorted_high = high[original_rows][order]
    sorted_rows = original_rows[order]
    del order

    high_collisions = int((sorted_high[1:] == sorted_high[:-1]).sum()) if len(sorted_high) else 0
    np.save(f"{WORK}/original_{index:04d}.npy", sorted_high)

    duplicate_rows = np.flatnonzero(duplicate)
    result = {
        "folder": folder,
        "rows": n_rows,
        "dataset_ids": n_dataset_ids,
        "duplicates": int(len(duplicate_rows)),
        "high_collisions_among_originals": high_collisions,
    }

    if not len(duplicate_rows):
        result.update(same_folder_same_dataset_id=0, same_folder_other_dataset_id=0, other_folder=0)

        return index, result

    if len(sorted_high):
        position = np.minimum(np.searchsorted(sorted_high, high[duplicate_rows]), len(sorted_high) - 1)
        twin_rows = sorted_rows[position]
        matched = (sorted_high[position] == high[duplicate_rows]) & (low[twin_rows] == low[duplicate_rows])
    else:
        twin_rows = np.zeros(len(duplicate_rows), dtype=np.int64)
        matched = np.zeros(len(duplicate_rows), dtype=bool)

    same_dataset_id = codes[twin_rows[matched]] == codes[duplicate_rows[matched]]

    result["same_folder_same_dataset_id"] = int(same_dataset_id.sum())
    result["same_folder_other_dataset_id"] = int((~same_dataset_id).sum())
    result["other_folder"] = int((~matched).sum())

    np.save(f"{WORK}/unmatched_{index:04d}.npy", high[duplicate_rows[~matched]])

    return index, result


def find_twin_folders(task):
    index, unmatched_path = task
    original = np.load(f"{WORK}/original_{index:04d}.npy")
    unmatched = np.load(unmatched_path, mmap_mode="r")

    if not len(original):
        return index, np.zeros(0, dtype=np.int64)

    position = np.minimum(np.searchsorted(original, unmatched), len(original) - 1)

    return index, np.flatnonzero(original[position] == unmatched)


def cell_statistics(population):
    population = np.asarray(population, dtype=np.int64)
    total = int(population.sum())
    descending = np.sort(population)[::-1]
    cumulative = np.cumsum(descending)

    return {
        "cells": int(len(population)),
        "sum": total,
        "largest": int(descending[0]),
        "largest_cell": int(np.argmax(population)),
        "median": float(np.median(population)),
        "cells_holding_half": int(np.searchsorted(cumulative, total / 2) + 1),
        "cells_holding_90_percent": int(np.searchsorted(cumulative, 0.9 * total) + 1),
        "empty_cells": int((population == 0).sum()),
    }


def population_arrays():
    report = {}

    release_population = np.load(f"{RELEASE}/mad-core/voronoi_population.npy")
    report["release_voronoi_population"] = cell_statistics(release_population)

    blob = np.load(f"{ANALYSIS}/populations.npz")
    level_offsets = blob["level_offsets"]
    pre_release = blob["populations"][level_offsets[24]:level_offsets[25]]
    report["pre_release_populations_npz_level24"] = cell_statistics(pre_release)
    report["pre_release_minus_release_per_cell"] = {
        "cells_differing": int((pre_release != release_population).sum()),
        "rows": int(pre_release.sum() - release_population.sum()),
    }

    level20 = np.load(f"{ANALYSIS}/populations_level20.npy").astype(np.float64)
    weights = level20 ** SKETCHMAP_WGAMMA
    weights /= weights.sum()
    report["sketchmap_weights_level20"] = {
        "cells": int(len(level20)),
        "sum": int(level20.sum()),
        "wgamma": SKETCHMAP_WGAMMA,
        "n_effective": float(1.0 / (weights ** 2).sum()),
    }

    # distinct structures with a within-cell ordering
    cell_fps = f"{RELEASE}/mad-core/cell_fps"
    center_offsets = np.load(f"{cell_fps}/center_offsets.npy")
    order_in_cell = np.load(f"{cell_fps}/fps_order_in_cell.npy")
    assert center_offsets[-1] == RELEASED_ROWS, center_offsets[-1]
    assert (np.diff(center_offsets) == release_population).all()

    starts = center_offsets[:-1]
    distinct_ordered = np.maximum.reduceat(order_in_cell, starts).astype(np.int64) + 1
    ordered_rows = np.add.reduceat((order_in_cell >= 0).astype(np.int64), starts)

    report["cell_fps_distinct_ordered"] = cell_statistics(distinct_ordered)
    report["cell_fps_rows_with_ordering"] = int(ordered_rows.sum())
    report["cell_fps_rows_without_ordering"] = int((order_in_cell < 0).sum())

    return report


def main():
    workers = ARGS.workers

    certificate = json.load(open(CERTIFICATE))
    radius = certificate["release"][str(LEVEL)]["cov_exact_direct"]

    collection = sorted(os.listdir(f"{RELEASE}/collection"))
    folders = [f"collection/{name}" for name in collection] + [ANCHOR]
    sizes = {folder: folder_rows(f"{RELEASE}/{folder}") for folder in folders}
    assert sum(sizes.values()) == RELEASED_ROWS

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)

    # pass one
    chunks = [
        (folder, start, min(start + CHUNK_ROWS, sizes[folder]), radius)

        for folder in sorted(folders, key=lambda name: -sizes[name])

        for start in range(0, sizes[folder], CHUNK_ROWS)
    ]

    per_folder = {folder: {} for folder in folders}

    with Pool(workers) as pool:
        for done, (folder, tallies) in enumerate(pool.imap_unordered(tally_chunk, chunks)):
            for name, values in tallies.items():
                current = per_folder[folder].setdefault(name, [0] * N_CATEGORIES)
                per_folder[folder][name] = [a + b for a, b in zip(current, values)]

            if (done + 1) % 100 == 0:
                print(f"tallies {done + 1}/{len(chunks)}", flush=True)

    totals = {}

    for tallies in per_folder.values():
        for name, values in tallies.items():
            current = totals.setdefault(name, [0] * N_CATEGORIES)
            totals[name] = [a + b for a, b in zip(current, values)]

    assert sum(totals["rows"]) == RELEASED_ROWS
    assert sum(totals["rows"][:8]) == COLLECTION_ROWS

    # pass two
    indexed = sorted(enumerate(folders[:-1]), key=lambda item: -sizes[item[1]])
    scope = {}

    with Pool(max(1, workers // 2)) as pool:
        for index, result in pool.imap_unordered(duplicate_scope, indexed):
            scope[index] = result
            print(f"scope {len(scope)}/{len(indexed)} {result['folder']}", flush=True)

    # pass three
    unmatched_parts = []

    for index in sorted(scope):
        path = f"{WORK}/unmatched_{index:04d}.npy"

        if os.path.exists(path):
            keys = np.load(path)
            unmatched_parts.append((keys, np.full(len(keys), index, dtype=np.int32)))

    unmatched_keys = np.concatenate([keys for keys, _ in unmatched_parts])
    unmatched_folder = np.concatenate([owners for _, owners in unmatched_parts])
    order = np.argsort(unmatched_keys, kind="stable")
    unmatched_keys = unmatched_keys[order]
    unmatched_folder = unmatched_folder[order]
    np.save(f"{WORK}/unmatched_all.npy", unmatched_keys)

    twin_folder = np.full(len(unmatched_keys), -1, dtype=np.int32)
    tasks = [(index, f"{WORK}/unmatched_all.npy") for index, _ in indexed]

    with Pool(max(1, workers // 2)) as pool:
        for index, hits in pool.imap_unordered(find_twin_folders, tasks):
            twin_folder[hits] = index

    # pair key from the two folder indices
    pairs, pair_counts = np.unique(
        unmatched_folder.astype(np.int64) * 10000 + twin_folder.astype(np.int64) + 1,
        return_counts=True,
    )
    pair_order = np.argsort(pair_counts)[::-1]
    top_pairs = [
        {
            "duplicate_folder": folders[int(pairs[i] // 10000)],
            "original_folder": folders[int(pairs[i] % 10000) - 1] if pairs[i] % 10000 else None,
            "rows": int(pair_counts[i]),
        }

        for i in pair_order[:40]
    ]

    other_folder_rows = int((twin_folder >= 0).sum())
    no_original_rows = int((twin_folder < 0).sum())

    shutil.rmtree(WORK, ignore_errors=True)

    # cell populations behind the voronoi claims and the maps
    populations = population_arrays()

    by_category = {
        name: {CATEGORY_NAMES[c]: totals[name][c] for c in range(N_CATEGORIES)}

        for name in totals
    }

    eligible = totals["rows"][0]
    draw_population = totals["natoms_below_300"][0] + totals["rows"][8]

    folder_table = {}

    for folder in folders:
        rows = per_folder[folder]["rows"]
        folder_table[folder] = {
            "rows": sum(rows),
            "filter_code": [rows[0] + rows[1], rows[2] + rows[3], rows[4] + rows[5], rows[6] + rows[7]],
            "duplicates": rows[1] + rows[3] + rows[5] + rows[7],
            "passed_not_duplicate": rows[0],
            "passed_not_duplicate_below_300_atoms": per_folder[folder]["natoms_below_300"][0],
        }

    scope_totals = {
        key: sum(result[key] for result in scope.values())

        for key in (
            "duplicates",
            "same_folder_same_dataset_id",
            "same_folder_other_dataset_id",
            "other_folder",
            "high_collisions_among_originals",
        )
    }

    report = {
        "claim": "populations behind the MAD-COSMO counts, filters, duplicates, coverage and cell statistics",
        "release": RELEASE,
        "folders": {"collection": len(collection), "with_anchor": len(folders)},
        "rows": {
            "released": sum(totals["rows"]),
            "collection": sum(totals["rows"][:8]),
            "anchor": totals["rows"][8],
        },
        "level24_radius_certified": radius,
        "by_category": by_category,
        "eligible_passed_not_duplicate": eligible,
        "eligible_plus_anchor": eligible + totals["rows"][8],
        "random_draw_population": draw_population,
        "collection_folders_without_passed_rows": [
            folder for folder in folders[:-1] if folder_table[folder]["filter_code"][0] == 0
        ],
        "duplicate_scope": {
            **scope_totals,
            "other_folder_original_found": other_folder_rows,
            "no_original_in_collection": no_original_rows,
            "folders_with_duplicates": sum(1 for result in scope.values() if result["duplicates"]),
            "folder_pairs": int(len(pairs)),
            "top_pairs": top_pairs,
            "per_folder": {result["folder"]: result for result in scope.values() if result["duplicates"]},
        },
        "populations": populations,
        "per_folder": folder_table,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)

    summary = {key: report[key] for key in report if key not in ("per_folder", "duplicate_scope")}
    summary["duplicate_scope"] = {key: value for key, value in report["duplicate_scope"].items() if key != "per_folder"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
