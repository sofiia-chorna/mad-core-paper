import json
from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
parser.add_argument("--certificate", default=CERTIFICATE_IN_RELEASE, help="coverage_exact.json of the coverage certificate")
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
CLAIMS_OUT = os.path.dirname(ARGS.out) if ARGS.out else os.path.join(HERE, "out")
ROWS = f"{CLAIMS_OUT}/outside_coverage_rows.npz"
SCAN = f"{CLAIMS_OUT}/outside_coverage_scan.json"
OUT = ARGS.out or f"{CLAIMS_OUT}/outside_coverage_maxima.json"
CERTIFICATE = ARGS.certificate

LEVELS = 25
LEVEL = 24
FEATURES = 512
PERCENTILES = [50, 90, 99]
FILTER_NAMES = ["passed", "zbl_rejected", "nan_features", "nomad_excluded"]


def folder_path(name):
    return f"{RELEASE}/mad-next-coreset" if name == "mad-next-coreset" else f"{RELEASE}/collection/{name}"


def formula(numbers):
    counts = Counter(chemical_symbols[number] for number in numbers)
    return "".join(f"{symbol}{count if count > 1 else ''}" for symbol, count in sorted(counts.items()))


def describe(name, row, rows, index):
    folder = folder_path(name)
    n_rows = int(np.load(f"{folder}/ns.npy")[0])

    atom_offsets = np.load(f"{folder}/na.npy", mmap_mode="r")
    start, stop = int(atom_offsets[row]), int(atom_offsets[row + 1])
    numbers = np.memmap(f"{folder}/a.bin", dtype=np.int32, mode="r")[start:stop]
    positions = np.memmap(f"{folder}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3)[start:stop]
    cell = np.memmap(f"{folder}/c.bin", dtype=np.float32, mode="r").reshape(n_rows, 3, 3)[row]
    pbc = np.load(f"{folder}/pbc.npy", mmap_mode="r")[row]

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    pairs = np.triu_indices(len(atoms), 1)
    min_distance_open = float(atoms.get_all_distances()[pairs].min()) if len(atoms) > 1 else None

    # minimum image only where the cell is periodic and not degenerate
    periodic = bool(pbc.any()) and abs(float(np.linalg.det(cell.astype(np.float64)))) > 1e-6
    min_distance_periodic = float(atoms.get_all_distances(mic=True)[pairs].min()) if periodic and len(atoms) > 1 else None

    # recompute the stored float32 distance in float64 from the released features
    scaler = np.load(f"{RELEASE}/ecumetric_scaler.npz")
    center = int(np.memmap(f"{folder}/voronoi_a.bin", dtype=np.int32, mode="r", shape=(n_rows, LEVELS))[row, LEVEL])
    features = np.memmap(f"{folder}/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(n_rows, FEATURES))[row]
    center_features = np.memmap(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32, mode="r").reshape(-1, FEATURES)[center]
    difference = (features.astype(np.float64) - center_features.astype(np.float64)) / float(scaler["scale"])

    filter_code = int(rows["filter_code"][index])
    zbl = {
        key: float(np.load(f"{folder}/{key}.npy", mmap_mode="r")[row])
        for key in ("zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy")
    }

    return {
        "folder": name,
        "row": row,
        "configuration_id": np.load(f"{folder}/configuration_id.npy", mmap_mode="r")[row].decode(),
        "dataset_id": np.load(f"{folder}/dataset_id.npy", mmap_mode="r")[row].decode(),
        "distance_stored_float32": float(rows["distance"][index]),
        "distance_float64_direct": float(np.sqrt((difference ** 2).sum())),
        "distance_over_radius": float(rows["distance"][index]) / float(rows["radius"]),
        "filter_code": filter_code,
        "filter": FILTER_NAMES[filter_code],
        "is_duplicate": bool(rows["is_duplicate"][index]),
        "in_selection_pool": bool(rows["in_pool"][index]),
        "image_flag_row": bool(rows["image_flag"][index]),
        "nearest_madcore_row": center,
        "nearest_madcore_configuration_id": np.load(f"{RELEASE}/mad-core/configuration_id.npy", mmap_mode="r")[center].decode(),
        "n_atoms": len(atoms),
        "formula": formula(numbers),
        "pbc": pbc.tolist(),
        "cell_lengths_angles": np.round(atoms.cell.cellpar(), 4).tolist(),
        "min_interatomic_distance_open": min_distance_open,
        "min_interatomic_distance_periodic": min_distance_periodic,
        **zbl,
    }


def group_maximum(rows, folder_names, mask):
    distance = rows["distance"]
    candidates = np.flatnonzero(mask)
    best = distance[candidates].max()
    tied = candidates[distance[candidates] == best]

    return [
        describe(folder_names[rows["folder_index"][index]], int(rows["row"][index]), rows, int(index))
        for index in tied
    ]


def main():
    rows = np.load(ROWS)
    scan = json.load(open(SCAN))

    folder_names = rows["folder_names"].astype(str)
    radius = float(rows["radius"])
    passed = rows["filter_code"] == 0
    eligible = passed & ~rows["is_duplicate"]

    unflagged = rows["distance"][passed].astype(np.float64)
    nonfinite = {
        name: folder["categories_nonfinite"]
        for name, folder in scan["folders"].items()
        if folder["nonfinite"]
    }

    report = {
        "radius": radius,
        "units": "scaled ecumetric distance, same units as the radius, not squared",
        "maximum_all_finite": group_maximum(rows, folder_names, np.ones(rows["distance"].size, dtype=bool)),
        "maximum_passed": group_maximum(rows, folder_names, passed),
        "maximum_passed_not_duplicate": group_maximum(rows, folder_names, eligible),
        "maximum_flagged": group_maximum(rows, folder_names, ~passed),
        "maximum_selection_pool": group_maximum(rows, folder_names, rows["in_pool"]),
        "passed_outside": {
            "n": int(unflagged.size),
            **{f"p{level}": float(value) for value, level in zip(np.percentile(unflagged, PERCENTILES), PERCENTILES)},
            "max": float(unflagged.max()),
        },
        "nonfinite": {
            "n": scan["nonfinite"],
            "category_names": scan["category_names"],
            "by_folder": nonfinite,
        },
    }

    # level-24 distance of the outside rows against the level-23 coverage radius
    radius_previous = json.load(open(CERTIFICATE))["release"][str(LEVEL - 1)]["cov_exact_direct"]
    within_previous = rows["distance"].astype(np.float64) <= radius_previous

    groups = {
        "all_outside_with_nonfinite": (np.ones(within_previous.size, dtype=bool), scan["nonfinite"]),
        "finite": (np.ones(within_previous.size, dtype=bool), 0),
        "passed": (passed, 0),
        "passed_not_duplicate": (eligible, 0),
        "flagged_with_nonfinite": (~passed, scan["nonfinite"]),
    }

    report["within_level_23_radius"] = {"radius_level_23": radius_previous}
    for name, (mask, extra) in groups.items():
        count = int((within_previous & mask).sum())
        total = int(mask.sum()) + extra
        report["within_level_23_radius"][name] = {
            "within": count,
            "total": total,
            "percent": 100.0 * count / total,
        }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
