import json
import os

import numpy as np

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description="see the README")
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
parser.add_argument("--populations", default=os.path.join(HERE, "out", "madcosmo_populations.json"), help="output of madcosmo_populations.py")
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
NOMAD = f"{RELEASE}/collection/nomad"
OUT = ARGS.out or os.path.join(HERE, "out", "nonfinite_descriptors.json")
POPULATIONS = ARGS.populations

LEVELS = 25
LEVEL = 24
FAR = 1e4
FILTER_NAMES = ["passed", "zbl_rejected", "nan_features", "nomad_excluded"]


def main():
    ns = int(np.load(f"{NOMAD}/ns.npy")[0])
    na = np.load(f"{NOMAD}/na.npy")
    filter_code = np.load(f"{NOMAD}/filter_code.npy")

    # a row whose descriptors are not finite has a non-finite distance to its level-24 centre
    distance = np.memmap(f"{NOMAD}/voronoi_d.bin", dtype=np.float32, mode="r").reshape(ns, LEVELS)[:, LEVEL]
    nonfinite = ~np.isfinite(np.asarray(distance))

    # rows that came from the source with unwrapped coordinates, original x is x + flags @ cell
    flag_rows = np.load(f"{NOMAD}/image_flag_rows.npy")
    flags = np.memmap(f"{NOMAD}/image_flags.bin", dtype=np.int64, mode="r").reshape(-1, 3)
    cells = np.memmap(f"{NOMAD}/image_flag_cells.bin", dtype=np.float64, mode="r").reshape(-1, 3, 3)
    x = np.memmap(f"{NOMAD}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3)

    natoms = na[flag_rows + 1] - na[flag_rows]
    flag_offsets = np.concatenate([[0], np.cumsum(natoms)])
    assert flag_offsets[-1] == flags.shape[0], (flag_offsets[-1], flags.shape)

    extent = np.empty(len(flag_rows))
    for i, row in enumerate(flag_rows):
        folded = np.asarray(x[na[row]:na[row + 1]], dtype=np.float64)
        original = folded + np.asarray(flags[flag_offsets[i]:flag_offsets[i + 1]]) @ np.asarray(cells[i])
        extent[i] = np.abs(original).max()

    far = np.zeros(ns, dtype=bool)
    far[flag_rows[extent > FAR]] = True
    flagged = np.zeros(ns, dtype=bool)
    flagged[flag_rows] = True

    nonfinite_extent = extent[nonfinite[flag_rows]]
    report = {
        "nomad_rows": ns,
        "nonfinite_descriptor_rows": int(nonfinite.sum()),
        "nonfinite_by_filter_code": {FILTER_NAMES[c]: int((nonfinite & (filter_code == c)).sum()) for c in range(4)},
        "rows_with_unwrapped_source_coordinates": int(len(flag_rows)),
        "rows_with_source_coordinates_beyond_1e4_A": int(far.sum()),
        "nonfinite_rows_with_unwrapped_coordinates": int((nonfinite & flagged).sum()),
        "nonfinite_rows_beyond_1e4_A": int((nonfinite & far).sum()),
        "nonfinite_rows_with_ordinary_coordinates": int((nonfinite & ~flagged).sum()),
        "source_extent_of_nonfinite_rows_A": {
            "min": float(nonfinite_extent.min()) if nonfinite_extent.size else None,
            "median": float(np.median(nonfinite_extent)) if nonfinite_extent.size else None,
            "max": float(nonfinite_extent.max()) if nonfinite_extent.size else None,
        },
        "far_rows_with_finite_descriptors": int((far & ~nonfinite).sum()),
    }

    # release-wide totals from the full scan, for the rows outside nomad
    populations = json.load(open(POPULATIONS))
    report["release_nonfinite_rows"] = int(sum(populations["by_category"]["nonfinite_d24"].values()))
    report["release_nonfinite_rows_outside_nomad"] = report["release_nonfinite_rows"] - report["nonfinite_descriptor_rows"]

    # outside nomad every non-finite row is ZBL-rejected (full scan), so only those rows are read
    outside = {}
    for folder, counts in populations["per_folder"].items():
        if folder == "collection/nomad" or counts["filter_code"][1] == 0:
            continue
        path = f"{RELEASE}/{folder}"
        rows = np.flatnonzero(np.load(f"{path}/filter_code.npy") == 1)
        n = int(np.load(f"{path}/ns.npy")[0])
        d = np.memmap(f"{path}/voronoi_d.bin", dtype=np.float32, mode="r").reshape(n, LEVELS)
        bad = rows[~np.isfinite(np.asarray(d[rows, LEVEL]))]
        if not bad.size:
            continue
        unwrapped = os.path.exists(f"{path}/image_flag_rows.npy")
        flag_rows_here = np.load(f"{path}/image_flag_rows.npy") if unwrapped else np.empty(0, dtype=np.int64)
        na_here = np.load(f"{path}/na.npy", mmap_mode="r")
        x_here = np.memmap(f"{path}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
        zbl_force = np.asarray(np.load(f"{path}/zbl_max_force.npy", mmap_mode="r")[bad])
        smallest = []
        for row in bad:
            positions = np.asarray(x_here[na_here[row]:na_here[row + 1]], dtype=np.float64)
            gaps = np.linalg.norm(positions[:, None] - positions[None], axis=-1)[np.triu_indices(len(positions), 1)]
            smallest.append(float(gaps.min()) if gaps.size else None)
        outside[folder] = {
            "nonfinite_rows": int(bad.size),
            "with_unwrapped_source_coordinates": int(np.isin(bad, flag_rows_here).sum()),
            "zbl_max_force_nonfinite": int((~np.isfinite(zbl_force)).sum()),
            "rows_with_coincident_atoms": int(sum(s is not None and s < 1e-6 for s in smallest)),
            "smallest_interatomic_distance_A": [min(s for s in smallest if s is not None), max(s for s in smallest if s is not None)],
        }
    report["nonfinite_outside_nomad_by_folder"] = outside
    assert sum(v["nonfinite_rows"] for v in outside.values()) == report["release_nonfinite_rows_outside_nomad"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
