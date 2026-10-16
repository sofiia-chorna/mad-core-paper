import os

import ase.io
import numpy as np
import torch
from deepmd.infer import DeepPot
from tqdm import tqdm

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"
OUTPUT_DIR = "data/mad-1.5/dpa-omol"
MODEL_PATH = "${OTHER_WORKSPACE}/models/DPA3-Omol-Large.pt"
OUTPUT_PATH = f"{OUTPUT_DIR}/dpa-omol_ll.npy"
IDS_PATH = f"{OUTPUT_DIR}/dpa-omol_ids.npz"

BATCH_SIZE = 128


def filter_structures(structures, type_map):
    supported_elements = set(type_map[:83])
    kept, kept_ids = [], []
    for i, s in enumerate(structures):
        if set(s.get_chemical_symbols()).issubset(supported_elements):
            kept.append(s)
            kept_ids.append(i)
    filtered_out = len(structures) - len(kept)
    if filtered_out > 0:
        print(f"Filtered out {filtered_out} structures with unsupported elements")
    return kept, np.array(kept_ids, dtype=np.int64)


def extract_features(structures, dp):
    from collections import defaultdict

    type_map = dp.get_type_map()

    # group by (natoms, is_periodic) — batches must have uniform natoms AND uniform
    groups = defaultdict(list)
    for i, s in enumerate(structures):
        is_periodic = bool(s.get_pbc().all())
        groups[(len(s), is_periodic)].append(i)
    print(
        f"unique (natoms, periodic) groups: {len(groups)}, "
        f"sizes: min={min(len(v) for v in groups.values())} max={max(len(v) for v in groups.values())}"
    )

    all_atom_features = [None] * len(structures)

    for (_, is_periodic), indices in tqdm(groups.items(), desc="natom groups"):
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[batch_start : batch_start + BATCH_SIZE]
            batch = [structures[i] for i in batch_idx]
            nframes = len(batch)

            coords = np.stack([s.get_positions() for s in batch]).reshape(nframes, -1)

            if is_periodic:
                cells = np.stack([s.get_cell().array for s in batch]).reshape(nframes, -1)
            else:
                cells = None  # deepmd uses non-PBC neighbor search

            atom_types = np.array(
                [
                    [type_map.index(sym) for sym in s.get_chemical_symbols()]
                    for s in batch
                ],
                dtype=np.int32,
            )

            fit_atomic = dp.eval_fitting_last_layer(
                coords, cells, atom_types, mixed_type=True
            )
            for local_i, orig_i in enumerate(batch_idx):
                all_atom_features[orig_i] = torch.from_numpy(fit_atomic[local_i])

    return all_atom_features


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dp = DeepPot(MODEL_PATH)
    type_map = dp.get_type_map()
    print(f"model type map ({len(type_map)} elements): {type_map[:10]}...")

    structures = ase.io.read(DATASET_PATH, ":")
    print(f"Loaded {len(structures)} structures total")

    structures, kept_ids = filter_structures(structures, type_map)
    print(f"Using {len(structures)} structures after filtering")

    atom_features = extract_features(structures, dp)

    valid_features = [atom_features[i] for i in range(len(structures))]
    atom_feats = torch.cat(valid_features, dim=0).cpu().numpy().astype(np.float32)

    print(f"Per-atom features shape: {atom_feats.shape}")
    np.save(OUTPUT_PATH, atom_feats)
    print(f"Saved to {OUTPUT_PATH}")

    np.savez(IDS_PATH, kept=kept_ids)
    print(f"Saved structure indices to {IDS_PATH}")
