import json
import os
from collections import Counter

import numpy as np

import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
CERTIFICATE_IN_RELEASE = os.path.join(HERE, os.pardir, "workflows", "coverage_certificate", "out", "stageB", "coverage_exact.json")

parser = argparse.ArgumentParser(description="see the README")
parser.add_argument("--release-root", required=True, help="the aa002 mad-core release directory (internal layout, see workflows/data_records/FORMAT.md)")
parser.add_argument("--out", default=None, help="output JSON (default: out/<script name>.json next to this script)")
ARGS = parser.parse_args()
RELEASE = ARGS.release_root
MAD_CORE = f"{RELEASE}/mad-core"
CELL_FPS = f"{MAD_CORE}/cell_fps"
OUT = ARGS.out or os.path.join(HERE, "out", "largest_voronoi_cell.json")

RELEASED_ROWS = 1694502355
N_CELLS = 1 << 24
TOP_CELLS = 117
ORGANIC = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
    "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No"
).split()


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def formula(numbers):
    counts = Counter(int(z) for z in numbers)
    order = [6, 1] if 6 in counts else []
    order += sorted(z for z in counts if z not in order)

    return "".join(SYMBOLS[z] + (str(counts[z]) if counts[z] > 1 else "") for z in order)


def structure(folder, row):
    path = f"{RELEASE}/{folder}"
    na = np.load(f"{path}/na.npy", mmap_mode="r")
    start, stop = int(na[row]), int(na[row + 1])
    numbers = np.memmap(f"{path}/a.bin", dtype=np.int32, mode="r")[start:stop]
    pbc = np.load(f"{path}/pbc.npy", mmap_mode="r")[row]

    record = {
        "folder": folder,
        "row": int(row),
        "configuration_id": text(np.load(f"{path}/configuration_id.npy", mmap_mode="r")[row]),
        "dataset_id": text(np.load(f"{path}/dataset_id.npy", mmap_mode="r")[row]),
        "natoms": stop - start,
        "formula": formula(numbers),
        "elements": sorted({SYMBOLS[int(z)] for z in numbers}),
        "organic_elements_only": bool(set(int(z) for z in numbers) <= ORGANIC),
        "pbc": [bool(p) for p in pbc],
    }

    if os.path.exists(f"{path}/source_reference_offsets.npy"):
        offsets = np.load(f"{path}/source_reference_offsets.npy", mmap_mode="r")
        reference = np.memmap(f"{path}/source_reference_bytes.bin", dtype=np.uint8, mode="r")
        record["source_reference"] = bytes(reference[int(offsets[row]):int(offsets[row + 1])]).decode()

    return record


def main():
    population = np.load(f"{MAD_CORE}/voronoi_population.npy")
    assert population.shape == (N_CELLS,), population.shape
    total = int(population.sum())
    assert total == RELEASED_ROWS, total

    # smallest number of cells holding half of the records
    descending = np.sort(population)[::-1]
    cumulative = np.cumsum(descending)
    half_cells = int(np.searchsorted(cumulative, total / 2) + 1)

    largest = int(np.argmax(population))
    names = [text(n) for n in np.load(f"{MAD_CORE}/release_directory_names.npy")]
    group = np.load(f"{MAD_CORE}/dataset_group.npy", mmap_mode="r")
    directory = np.load(f"{MAD_CORE}/release_directory.npy", mmap_mode="r")
    row_in_directory = np.load(f"{MAD_CORE}/release_row_in_directory.npy", mmap_mode="r")
    k_set = np.load(f"{MAD_CORE}/k_set.npy", mmap_mode="r")

    centre_folder = names[int(directory[largest])]
    centre = structure(centre_folder, int(row_in_directory[largest]))
    centre["mad_core_row"] = largest
    centre["mad_core_dataset_group"] = text(group[largest])
    centre["mad_core_configuration_id"] = text(np.load(f"{MAD_CORE}/configuration_id.npy", mmap_mode="r")[largest])
    centre["k_set"] = int(k_set[largest])
    centre["inside_mad_next_anchor_block"] = largest < 856012
    assert centre["mad_core_configuration_id"] == centre["configuration_id"]

    # members of the largest cell
    offsets = np.load(f"{CELL_FPS}/center_offsets.npy", mmap_mode="r")
    lo, hi = int(offsets[largest]), int(offsets[largest + 1])
    assert hi - lo == int(population[largest]), (hi - lo, int(population[largest]))
    assert text(np.load(f"{CELL_FPS}/center_configuration_id.npy", mmap_mode="r")[largest]) == centre["configuration_id"]

    # within-cell FPS
    order_in_cell = np.asarray(np.load(f"{CELL_FPS}/fps_order_in_cell.npy", mmap_mode="r")[lo:hi])
    ordered = order_in_cell[order_in_cell >= 0]
    distinct_ordered = int(np.unique(ordered).size)
    assert distinct_ordered == int(ordered.max()) + 1
    within_cell_fps = {
        "rows": hi - lo,
        "rows_without_ordering": int((order_in_cell < 0).sum()),
        "distinct_structures_ordered": distinct_ordered,
        "rows_repeating_a_twin_rank": int(ordered.size - distinct_ordered),
        "centre_has_rank_0": int((order_in_cell == 0).sum()) >= 1,
    }

    member_directory = np.asarray(np.load(f"{CELL_FPS}/release_directory.npy", mmap_mode="r")[lo:hi])
    folder_counts = np.bincount(member_directory, minlength=len(names))
    members_by_folder = {names[i]: int(c) for i, c in sorted(enumerate(folder_counts), key=lambda t: -t[1]) if c}

    # where the centres of the cells holding half of the records come from
    top = np.argsort(population)[::-1][:half_cells]
    top_groups = Counter(text(group[i]) for i in top)

    report = {
        "released_rows": total,
        "cells": N_CELLS,
        "cells_holding_half_of_the_records": half_cells,
        "records_in_those_cells": int(cumulative[half_cells - 1]),
        "records_in_the_first_116_cells": int(cumulative[half_cells - 2]),
        "largest_cell": {
            "cell": largest,
            "records": int(population[largest]),
            "centre": centre,
            "within_cell_fps": within_cell_fps,
            "members_by_release_folder": members_by_folder,
            "share_of_members_in_centre_folder": float(folder_counts[int(directory[largest])] / (hi - lo)),
        },
        "centre_groups_of_the_half_cells": dict(top_groups.most_common()),
        "text_claim": {
            "half_cells_is_117": half_cells == TOP_CELLS,
            "largest_rounds_to_115_million": round(int(population[largest]) / 1e6) == 115,
            "centre_has_26_atoms": centre["natoms"] == 26,
            "centre_is_gdb13": centre_folder == "collection/gdb13",
            "centre_is_organic": centre["organic_elements_only"],
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
