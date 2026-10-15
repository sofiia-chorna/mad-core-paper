import json
import os
from multiprocessing import Pool

import numpy as np

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
parser.add_argument("--certificate", default=CERTIFICATE_IN_RELEASE, help="coverage_exact.json of the coverage certificate")
parser.add_argument("workers", nargs="?", type=int, default=32)
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
CERTIFICATE = ARGS.certificate
OUT = ARGS.out or os.path.join(HERE, "out", "collection_within_coverage.json")

LEVELS = 25
LEVEL = 24
BLOCK_ROWS = 1 << 22
RELEASED_ROWS = 1694502355

radii = {}


def folder_rows(folder):
    if folder.endswith("mad-next-coreset"):
        return np.load(f"{folder}/configuration_id.npy", mmap_mode="r").shape[0]

    return int(np.load(f"{folder}/ns.npy")[0])


def count_folder(folder):
    n_rows = folder_rows(folder)
    distances = np.memmap(f"{folder}/voronoi_d.bin", dtype=np.float32, mode="r", shape=(n_rows, LEVELS))

    within = {name: 0 for name in radii}
    n_nonfinite = 0
    largest = 0.0

    for start in range(0, n_rows, BLOCK_ROWS):
        block = np.asarray(distances[start:start + BLOCK_ROWS, LEVEL], dtype=np.float64)

        finite = np.isfinite(block)
        n_nonfinite += int((~finite).sum())
        block = block[finite]

        for name, radius in radii.items():
            within[name] += int((block <= radius).sum())

        if block.size:
            largest = max(largest, float(block.max()))

    return os.path.basename(folder), n_rows, within, n_nonfinite, largest


def main():
    certificate = json.load(open(CERTIFICATE))
    levels = json.load(open(f"{RELEASE}/mad-core/levels.json"))["levels"]
    stored = next(entry for entry in levels if entry["level"] == LEVEL)

    radii["certified_exact"] = certificate["release"][str(LEVEL)]["cov_exact_direct"]
    radii["construction_log_float32"] = stored["coverage_runtime_float32_construction_log"]

    folders = sorted(f"{RELEASE}/collection/{name}" for name in os.listdir(f"{RELEASE}/collection"))
    folders.append(f"{RELEASE}/mad-next-coreset")

    workers = ARGS.workers
    with Pool(workers) as pool:
        results = pool.map(count_folder, folders)

    total_rows = sum(result[1] for result in results)
    assert total_rows == RELEASED_ROWS, total_rows

    report = {
        "claim": "fraction of the released collection within the level-24 coverage radius of the candidate pool",
        "level": LEVEL,
        "size": 2 ** LEVEL,
        "n_folders": len(folders),
        "n_rows": total_rows,
        "n_nonfinite_distance": sum(result[3] for result in results),
        "largest_distance": max(result[4] for result in results),
        "radius": radii,
        "within": {},
    }

    for name in radii:
        count = sum(result[2][name] for result in results)
        report["within"][name] = {"count": count, "percent": 100.0 * count / total_rows}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
