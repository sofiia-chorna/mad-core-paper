import os
import sys

import ase.io
import numpy as np
import torch
from deepmd.infer import DeepPot
from tqdm import tqdm

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"
OUTPUT_DIR = "data/mad-1.5/dpa"

# one model per training dataset — pick the one that matches your data domain
TASKS = ["Mptraj", "Omat24", "OC20M", "SPICE2", "ODAC23", "Domains_Drug", "H2O_H2O_PD"]

if len(sys.argv) < 2:
    print("Usage: python3 mad15_dpa_feats.py <TASK> [LL|BB]")
    print(f"  Available tasks: {TASKS}")
    print("  Variant defaults to LL if not specified")
    sys.exit(1)

task = sys.argv[1]
if task not in TASKS:
    print(f"Error: unknown task '{task}'. Available: {TASKS}")
    sys.exit(1)

VARIANT = sys.argv[2].upper() if len(sys.argv) > 2 else "LL"
if VARIANT not in ["LL", "BB"]:
    print(f"Error: unknown variant '{VARIANT}'. Available: ['LL', 'BB']")
    sys.exit(1)

# optional pass start/end indices to split work across multiple gpu jobs, then merge outputs
START = int(sys.argv[3]) if len(sys.argv) > 3 else None
END = int(sys.argv[4]) if len(sys.argv) > 4 else None

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = f"${OTHER_WORKSPACE}/models/dpa3.1_{task}.pth"
slice_suffix = f"_{START}_{END}" if START is not None else ""
OUTPUT_PATH = f"{OUTPUT_DIR}/dpa-{task}_{VARIANT.lower()}{slice_suffix}"

print(f"Task: {task}")
print(f"Variant: {VARIANT}")
print(f"Slice: {START}:{END}")
print(f"Dataset: {DATASET_PATH}")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_PATH}.npy")


BATCH_SIZE = 64


def extract_deepmd_features(structures, model_path, variant, device="cpu"):
    from collections import defaultdict

    dp = DeepPot(model_path)
    # we must use these indices, not local per-structure ones
    type_map = dp.get_type_map()
    print(f"model type map ({len(type_map)} elements): {type_map[:10]}...")

    # group by (natoms, is_periodic) — deepmd can't batch structures of different sizes
    groups = defaultdict(list)
    for i, s in enumerate(structures):
        groups[(len(s), s.get_pbc().all())].append(i)
    print(
        f"unique (natoms, pbc) groups: {len(groups)}, sizes: min={min(len(v) for v in groups.values())} max={max(len(v) for v in groups.values())}"
    )

    # pre-allocate in original order so we can write results back by index
    all_atom_features = [None] * len(structures)

    eval_fn = (
        dp.eval_fitting_backbone_layer
        if variant == "BB"
        else dp.eval_fitting_last_layer
    )

    for group_key, indices in tqdm(groups.items(), desc="natom groups"):
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[batch_start : batch_start + BATCH_SIZE]
            batch = [structures[i] for i in batch_idx]
            nframes = len(batch)

            coords = np.stack([s.get_positions() for s in batch]).reshape(nframes, -1)
            cells = np.stack([s.get_cell().array for s in batch]).reshape(nframes, -1)
            # mixed_type True each row of atom_types is the type indices for one frame
            atom_types = np.array(
                [
                    [type_map.index(sym) for sym in s.get_chemical_symbols()]
                    for s in batch
                ],
                dtype=np.int32,
            )

            fit_atomic = eval_fn(coords, cells, atom_types, mixed_type=True)
            # output is (nframes, natoms, feat_dim) — write each frame back to its original position
            for local_i, orig_i in enumerate(batch_idx):
                all_atom_features[orig_i] = torch.from_numpy(fit_atomic[local_i])

    return all_atom_features


if __name__ == "__main__":
    print("DP_VARIANT:", os.environ.get("DP_VARIANT"))
    print("CUDAToolkit_ROOT:", os.environ.get("CUDAToolkit_ROOT"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    structures = ase.io.read(DATASET_PATH, ":")
    print(f"Loaded {len(structures)} structures total")
    if START is not None:
        structures = structures[START:END]
        print(f"Using slice [{START}:{END}] → {len(structures)} structures")

    atom_features = extract_deepmd_features(structures, MODEL_PATH, VARIANT, device)
    atom_feats = torch.cat(atom_features, dim=0).cpu().numpy().astype(np.float32)

    print(f"Per-atom features shape: {atom_feats.shape}")
    np.save(OUTPUT_PATH, atom_feats)
    print(f"Saved to {OUTPUT_PATH}.npy")
