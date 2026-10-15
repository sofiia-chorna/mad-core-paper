import json
from collections import defaultdict

import numpy as np

from common import HERE, ROOT, STORE, load_frames

SPACE_DIR = f"{STORE}/mad-core/models/training/atomic_combined"
MAD_TIER_DIR = f"{STORE}/mad-core/release/mad-next-coreset"
RELEASE_IDS_TSV = f"{ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe/sets/random_test_6k/release_ids.tsv"
SELECTION = f"{HERE}/results_random_test/selection_random_seed0.npz"
OUT = f"{HERE}/results_random_test/overlap_audit.json"

# same composition and sorted interatomic distances within this tolerance count as the same geometry
TOLERANCE = 0.01
EXACT = 1e-4


def composition_key(numbers):
    return np.sort(np.asarray(numbers, dtype=np.int64)).tobytes()


# sorted pairwise distances, minimum image when periodic, invariant to atom order, translation, rotation
def sorted_distances(positions, cell, periodic):
    positions = np.asarray(positions, dtype=np.float64)
    delta = positions[:, None, :] - positions[None, :, :]

    if periodic:
        cell = np.asarray(cell, dtype=np.float64)
        fractional = delta @ np.linalg.inv(cell)
        fractional -= np.round(fractional)
        delta = fractional @ cell

    upper = np.triu_indices(len(positions), k=1)

    return np.sort(np.linalg.norm(delta[upper], axis=1))


def is_periodic(cell, pbc):
    return bool(np.any(pbc)) and abs(np.linalg.det(np.asarray(cell, dtype=np.float64))) > 1e-6


def load_reference(name):
    if name == "space":
        na = np.load(f"{SPACE_DIR}/na.npy")
        n = len(na) - 1
        cells = np.memmap(f"{SPACE_DIR}/c.bin", dtype=np.float32, mode="r", shape=(n, 3, 3))
        # the training arrays carry no pbc, a non singular cell means periodic
        pbc = np.array([abs(np.linalg.det(np.asarray(cells[i], dtype=np.float64))) > 1e-6 for i in range(n)])
        pbc = np.repeat(pbc[:, None], 3, axis=1)

        split = np.full(n, "train", dtype="U5")
        split[np.loadtxt(f"{SPACE_DIR}/val_indices_in_all.txt", dtype=np.int64)] = "val"
        split[np.loadtxt(f"{SPACE_DIR}/test_indices_in_all.txt", dtype=np.int64)] = "test"

        return {
            "na": na,
            "numbers": np.memmap(f"{SPACE_DIR}/a.bin", dtype=np.int32, mode="r"),
            "positions": np.memmap(f"{SPACE_DIR}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3),
            "cells": cells,
            "pbc": pbc,
            "split": split,
            "ids": None,
        }

    na = np.load(f"{MAD_TIER_DIR}/na.npy")
    n = len(na) - 1

    return {
        "na": na,
        "numbers": np.memmap(f"{MAD_TIER_DIR}/a.bin", dtype=np.int32, mode="r"),
        "positions": np.memmap(f"{MAD_TIER_DIR}/x.bin", dtype=np.float32, mode="r").reshape(-1, 3),
        "cells": np.memmap(f"{MAD_TIER_DIR}/c.bin", dtype=np.float32, mode="r").reshape(n, 3, 3),
        "pbc": np.load(f"{MAD_TIER_DIR}/pbc.npy"),
        "split": np.load(f"{MAD_TIER_DIR}/split.npy").astype(str),
        "ids": np.load(f"{MAD_TIER_DIR}/configuration_id.npy").astype(str),
    }


def match_reference(name, frames, wanted_keys):
    reference = load_reference(name)
    na = reference["na"]
    n = len(na) - 1

    candidates = defaultdict(list)
    for i in range(n):
        key = composition_key(reference["numbers"][na[i]:na[i + 1]])
        if key in wanted_keys:
            candidates[key].append(i)

    n_candidates = sum(len(v) for v in candidates.values())
    print(f"{name} has {n} structures, {n_candidates} share a composition with a random structure", flush=True)

    matches = defaultdict(list)
    cache = {}
    for frame_index, atoms in frames.items():
        key = composition_key(atoms.get_atomic_numbers())
        if key not in candidates:
            continue

        mine = sorted_distances(atoms.get_positions(), atoms.get_cell().array, is_periodic(atoms.cell.array, atoms.pbc))
        mine_lengths = np.sort(atoms.cell.lengths())

        for i in candidates[key]:
            if i not in cache:
                start, end = na[i], na[i + 1]
                periodic = is_periodic(reference["cells"][i], reference["pbc"][i])
                cache[i] = (
                    sorted_distances(reference["positions"][start:end], reference["cells"][i], periodic),
                    periodic,
                    np.sort(np.linalg.norm(np.asarray(reference["cells"][i], dtype=np.float64), axis=1)),
                )
            theirs, their_periodic, their_lengths = cache[i]

            difference = float(np.abs(mine - theirs).max()) if len(mine) else 0.0
            if difference > TOLERANCE:
                continue

            matches[frame_index].append({
                "reference_index": int(i),
                "reference_split": str(reference["split"][i]) or "none",
                "reference_id": None if reference["ids"] is None else str(reference["ids"][i]),
                "max_distance_difference": difference,
                "exact": difference < EXACT,
                "same_periodicity": bool(their_periodic == is_periodic(atoms.cell.array, atoms.pbc)),
                "same_cell_lengths": bool(np.allclose(mine_lengths, their_lengths, atol=TOLERANCE)),
            })

    return matches, reference


def summarize(label, subset, matches, splits, id_hits):
    summary = {"n_structures": len(subset), "id_in_mad_next_tier": int(sum(i in id_hits for i in subset))}

    for reference, found in matches.items():
        hit = [i for i in subset if i in found]
        summary[f"geometry_match_{reference}"] = len(hit)
        summary[f"exact_geometry_match_{reference}"] = int(sum(any(m["exact"] for m in found[i]) for i in hit))
        for split in splits[reference]:
            summary[f"geometry_match_{reference}_{split}"] = int(sum(
                any(m["reference_split"] == split for m in found[i]) for i in hit
            ))

    print(label, summary, flush=True)

    return summary


def main():
    frames_list, meta = load_frames()
    selection = np.load(SELECTION)
    usable = [int(i) for i in selection["kept"]]
    test = [int(i) for i in selection["kept"][selection["test_idx"]]]
    frames = {i: frames_list[i] for i in usable}
    wanted = {composition_key(atoms.get_atomic_numbers()) for atoms in frames.values()}

    release_of = {}
    with open(RELEASE_IDS_TSV) as handle:
        next(handle)
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            release_of[columns[0]] = columns[1]

    mad_ids = set(np.load(f"{MAD_TIER_DIR}/configuration_id.npy").astype(str))
    configuration_ids = meta["configuration_id"].astype(str)
    id_hits = {i for i in usable if release_of.get(configuration_ids[i], configuration_ids[i]) in mad_ids}

    matches = {}
    splits = {}
    for name in ["space", "mad_next_tier"]:
        matches[name], reference = match_reference(name, frames, wanted)
        splits[name] = sorted({str(s) or "none" for s in np.unique(reference["split"])})

    details = []
    for reference, found in matches.items():
        for frame_index, hits in found.items():
            details.append({
                "reference": reference,
                "frame": frame_index,
                "configuration_id": configuration_ids[frame_index],
                "dataset_group": str(meta["dataset_group"][frame_index]),
                "in_reconstruction_test": frame_index in set(test),
                "matches": hits[:5],
                "n_matches": len(hits),
            })

    result = {
        "tolerance_angstrom": TOLERANCE,
        "usable": summarize("usable", usable, matches, splits, id_hits),
        "reconstruction_test": summarize("reconstruction test", test, matches, splits, id_hits),
        "id_hits": [configuration_ids[i] for i in sorted(id_hits)],
        "details": details,
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
