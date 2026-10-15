"""
Extract DPA-3.1 per-atom features from the MAD-1.5 dataset.
Handles both periodic and non-periodic structures (cells=None for molecules).

Usage:
    python3 scripts/feats/mad15/mad15_dpa31_feats.py <TASK> [LL|BB]
    e.g. python3 scripts/feats/mad15/mad15_dpa31_feats.py Omat24 LL
"""

import os
import sys
from collections import defaultdict

import ase.io
import numpy as np
import torch
from deepmd.infer import DeepPot
from tqdm import tqdm

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"
OUTPUT_DIR = "data/mad-1.5/dpa/dpa-3.1"

TASKS = ["Omat24", "Mptraj", "OC20M", "ODAC23", "SPICE2"]

if len(sys.argv) < 2:
    print("Usage: python3 mad15_dpa31_feats.py <TASK> [LL|BB]")
    print(f"  Available tasks: {TASKS}")
    sys.exit(1)

task = sys.argv[1]
if task not in TASKS:
    print(f"Error: unknown task '{task}'. Available: {TASKS}")
    sys.exit(1)

VARIANT = sys.argv[2].upper() if len(sys.argv) > 2 else "LL"
if VARIANT not in ["LL", "BB"]:
    print(f"Error: unknown variant '{VARIANT}'. Use LL or BB")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = f"${OTHER_WORKSPACE}/models/dpa3.1_{task}.pth"
OUTPUT_PATH = f"{OUTPUT_DIR}/dpa3.1-{task}_{VARIANT.lower()}.npy"
IDS_PATH = f"{OUTPUT_DIR}/dpa3.1-{task}_ids.npz"

print(f"Task:    {task}")
print(f"Variant: {VARIANT}")
print(f"Model:   {MODEL_PATH}")
print(f"Output:  {OUTPUT_PATH}")

BATCH_SIZE = 64


def filter_structures(structures, type_map):
    type_map_set = set(type_map)
    kept, kept_ids, unsupported_ids = [], [], []

    for i, s in enumerate(structures):
        if not set(s.get_chemical_symbols()).issubset(type_map_set):
            unsupported_ids.append(i)
            continue
        kept.append(s)
        kept_ids.append(i)

    print(f"filtered out {len(unsupported_ids)} structures with unsupported elements")
    return kept, np.array(kept_ids, dtype=np.int64)


def extract_features(structures, dp, variant):
    type_map = dp.get_type_map()
    groups = defaultdict(list)
    for i, s in enumerate(structures):
        groups[(len(s), bool(s.get_pbc().all()))].append(i)
    print(f"unique (natoms, periodic) groups: {len(groups)}")

    all_atom_features = [None] * len(structures)
    eval_fn = (
        dp.eval_fitting_backbone_layer if variant == "BB" else dp.eval_fitting_last_layer
    )

    for (_, is_periodic), indices in tqdm(groups.items(), desc="natom groups"):
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[batch_start : batch_start + BATCH_SIZE]
            batch = [structures[i] for i in batch_idx]
            nframes = len(batch)

            coords = np.stack([s.get_positions() for s in batch]).reshape(nframes, -1)
            cells = (
                np.stack([s.get_cell().array for s in batch]).reshape(nframes, -1)
                if is_periodic
                else None
            )
            atom_types = np.array(
                [[type_map.index(sym) for sym in s.get_chemical_symbols()] for s in batch],
                dtype=np.int32,
            )

            fit_atomic = eval_fn(coords, cells, atom_types, mixed_type=True)
            for local_i, orig_i in enumerate(batch_idx):
                all_atom_features[orig_i] = torch.from_numpy(fit_atomic[local_i])

    return all_atom_features


if __name__ == "__main__":
    print("DP_VARIANT:", os.environ.get("DP_VARIANT"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dp = DeepPot(MODEL_PATH)
    type_map = dp.get_type_map()
    print(f"type map ({len(type_map)} elements): {type_map[:10]}...")

    structures = ase.io.read(DATASET_PATH, ":")
    print(f"loaded {len(structures)} structures")

    if os.path.exists(IDS_PATH):
        print(f"loading cached filter indices from {IDS_PATH}")
        kept_ids = np.load(IDS_PATH)["kept"]
        structures = [structures[i] for i in kept_ids]
        print(f"using {len(structures)} structures (cached)")
    else:
        structures, kept_ids = filter_structures(structures, type_map)
        print(f"using {len(structures)} structures after filtering")

    atom_features = extract_features(structures, dp, VARIANT)

    nan_local = [i for i, f in enumerate(atom_features) if torch.isnan(f).any()]
    if nan_local:
        print(f"WARNING: {len(nan_local)} structures produced NaN — dropping")
        nan_set = set(nan_local)
        atom_features = [f for i, f in enumerate(atom_features) if i not in nan_set]
        kept_ids = np.array([kept_ids[i] for i in range(len(kept_ids)) if i not in nan_set], dtype=np.int64)
    else:
        print("no NaN features")

    np.savez(IDS_PATH, kept=kept_ids)
    print(f"saved ids → {IDS_PATH}")

    atom_feats = torch.cat(atom_features, dim=0).cpu().numpy().astype(np.float32)
    print(f"per-atom features shape: {atom_feats.shape}")
    np.save(OUTPUT_PATH, atom_feats)
    print(f"saved features → {OUTPUT_PATH}")
