import json

import numpy as np
from ase.data import chemical_symbols

from common import FEATURE_DIR, HERE, ROOT, STORE

RELEASE = f"{STORE}/mad-core/release"
SETS = f"{ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe/sets/random_test_6k"
SELECTION = f"{HERE}/results_random_test/selection_random_seed0.npz"
OUT = f"{HERE}/results_random_test/feature_audit.json"
MODELS = ["pet-mad", "mace-omat", "uma-omat", "dpa-omat", "dpa-omol", "orb-v3-omat", "phace-atomic"]
DPA_OMOL_ELEMENTS = 83


def read_tsv(path):
    with open(path) as handle:
        header = next(handle).rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def pool_mean_std(rows):
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)])


def main():
    meta = np.load(f"{FEATURE_DIR}/frames_meta.npz")
    natoms = meta["natoms"]
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    configuration_ids = meta["configuration_id"].astype(str)
    usable = np.load(SELECTION)["kept"]
    result = {}

    # every frame must be either fully finite or fully NaN in every model
    patterns = {}
    for model in MODELS:
        features = np.load(f"{FEATURE_DIR}/{model}_ll.npy", mmap_mode="r")
        finite_rows = np.isfinite(features).all(axis=1)
        per_frame = np.add.reduceat(finite_rows.astype(np.int64), offsets[:-1])
        patterns[model] = {
            "rows": int(features.shape[0]),
            "frames_all_finite": int((per_frame == natoms).sum()),
            "frames_all_nan": int((per_frame == 0).sum()),
            "frames_partially_nan": int(((per_frame > 0) & (per_frame < natoms)).sum()),
        }
    result["nan_pattern"] = patterns
    print(json.dumps(patterns, indent=1), flush=True)

    # the excluded DPA3-OMol frames must be exactly those with an element beyond its first 83
    numbers_by_frame = []
    for name in ["fff", "ttt", "surfaces"]:
        with open(f"{SETS}/{name}.extxyz") as handle:
            while True:
                header = handle.readline()
                if not header:
                    break
                count = int(header)
                handle.readline()
                numbers_by_frame.append([chemical_symbols.index(handle.readline().split()[0]) for _ in range(count)])
    beyond = np.array([max(numbers) > DPA_OMOL_ELEMENTS for numbers in numbers_by_frame])
    with open(f"{FEATURE_DIR}/dpa-omol_exclusions.json") as handle:
        excluded = set(json.load(handle)["excluded"].get("unsupported_element", []))
    excluded_mask = np.array([i in excluded for i in range(len(beyond))])
    elements = sorted({z for i in np.where(beyond)[0] for z in numbers_by_frame[i] if z > DPA_OMOL_ELEMENTS})
    result["dpa_omol_exclusions"] = {
        "excluded": int(excluded_mask.sum()),
        "frames_with_element_beyond_83": int(beyond.sum()),
        "identical_sets": bool(np.array_equal(excluded_mask, beyond)),
        "elements_beyond_83": [chemical_symbols[z] for z in elements],
    }
    print(result["dpa_omol_exclusions"], flush=True)

    # ecumetric descriptors pooled here against the release ecumetric_features.bin of the same rows
    release_of = {row["configuration_id"]: row["release_configuration_id"] for row in read_tsv(f"{SETS}/release_ids.tsv")}
    location = {row["configuration_id"]: (row["dataset_group"], int(row["row_in_group"])) for row in read_tsv(f"{SETS}/selection.tsv")}
    features = np.load(f"{FEATURE_DIR}/phace-atomic_ll.npy", mmap_mode="r")

    differences = []
    id_mismatch = 0
    for i in usable:
        group, row = location[configuration_ids[i]]
        folder = f"{RELEASE}/mad-next-coreset" if group == "mad-next-coreset" else f"{RELEASE}/collection/{group}"
        release_id = np.load(f"{folder}/configuration_id.npy", mmap_mode="r")[row].decode()
        if release_id != release_of[configuration_ids[i]]:
            id_mismatch += 1
            continue

        stored = np.memmap(f"{folder}/ecumetric_features.bin", dtype=np.float32, mode="r").reshape(-1, 512)[row]
        mine = pool_mean_std(np.asarray(features[offsets[i]:offsets[i + 1]], dtype=np.float64))
        differences.append(float(np.abs(mine - stored).max() / max(np.abs(stored).max(), 1e-12)))

    differences = np.array(differences)
    result["ecumetric_vs_release"] = {
        "compared": int(len(differences)),
        "release_row_id_mismatch": id_mismatch,
        "max_relative_difference": float(differences.max()),
        "median_relative_difference": float(np.median(differences)),
        "above_1e-3": int((differences > 1e-3).sum()),
    }
    print(result["ecumetric_vs_release"], flush=True)

    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
