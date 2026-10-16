import os
import sys

import numpy as np
import torch
from ase.io import read
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from tqdm import tqdm

DATASET_PATH = "data/mad-1.5/mad-1.5-r2scan-all.xyz"
OUTPUT_DIR = "data/mad-1.5/uma"
MODEL_NAME = "uma-s-1p2"

AVAILABLE_TASKS = ["oc20", "omat", "omol", "odac", "omc"]

if len(sys.argv) < 2:
    print("Usage: python3 mad15_uma_feats2.py <TASK> [LL|BB]")
    print(f"  Available tasks: {AVAILABLE_TASKS}")
    sys.exit(1)

TASK = sys.argv[1].lower()
VARIANT = sys.argv[2].upper() if len(sys.argv) > 2 else "BOTH"

if TASK not in AVAILABLE_TASKS:
    print(f"Error: unknown task '{TASK}'. Available: {AVAILABLE_TASKS}")
    sys.exit(1)

if VARIANT not in ["LL", "BB", "BOTH"]:
    print(f"Error: unknown variant '{VARIANT}'. Available: ['LL', 'BB', 'BOTH']")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 128

print(f"Task: {TASK}")
print(f"Variant: {VARIANT}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output directory: {OUTPUT_DIR}")


def get_uma_features_batched(data_list, task, predictor, batch_size=BATCH_SIZE):
    # UMA backbone asserts pbc must be all-True or all-False within each batch
    periodic_local = [i for i, d in enumerate(data_list) if d.pbc.all().item()]
    nonperiodic_local = [i for i, d in enumerate(data_list) if not d.pbc.all().item()]
    print(f"periodic: {len(periodic_local)}, non-periodic: {len(nonperiodic_local)}")

    all_atom_features_bb = [None] * len(data_list)
    all_atom_features_ll = [None] * len(data_list)

    for group_name, local_indices in [("periodic", periodic_local), ("non-periodic", nonperiodic_local)]:
        if not local_indices:
            continue

        print(f"processing {len(local_indices)} {group_name} structures in batches of {batch_size}...")
        for i in tqdm(range(0, len(local_indices), batch_size), desc=f"{group_name} inference"):
            batch_local = local_indices[i : i + batch_size]
            batch_data = [data_list[j] for j in batch_local]
            batch = atomicdata_list_to_batch(batch_data)
            batch.dataset = [task] * len(batch_data)

            with torch.no_grad():
                output = predictor.predict(batch, expose_feat=True)

            bb_feat = output["bb_feat"].to(torch.float32).cpu()
            ll_feat = output["ll_feat"].to(torch.float32).cpu()

            natoms_list = [d.natoms.item() for d in batch_data]
            bb_splits = torch.split(bb_feat, natoms_list)
            ll_splits = torch.split(ll_feat, natoms_list)

            for local_j, orig_local in enumerate(batch_local):
                all_atom_features_bb[orig_local] = bb_splits[local_j]
                all_atom_features_ll[orig_local] = ll_splits[local_j]

    return all_atom_features_bb, all_atom_features_ll


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("loading uma model...")
    predictor = pretrained_mlip.get_predict_unit(
        MODEL_NAME, device=device, expose_feat=True
    )

    try:
        max_num_elements = predictor.model.module.backbone.max_num_elements
    except AttributeError:
        max_num_elements = 100
    print(f"model max_num_elements: {max_num_elements}")
    max_atomic_number = max_num_elements - 1

    structures = read(DATASET_PATH, ":")
    print(f"Loaded {len(structures)} structures")

    # only filter what causes a hard crash

    # unsupported elements embedding index out of bounds, O(N) to check
    unsupported_ids = []
    surviving = []
    for i, s in enumerate(structures):
        atomic_nums = s.get_atomic_numbers()
        if np.any(atomic_nums > max_atomic_number) or np.any(atomic_nums < 1):
            unsupported_ids.append(i)
        else:
            surviving.append(i)
    print(f"filtered out {len(unsupported_ids)} structures with unsupported elements")

    # convert to AtomicData with try-except
    data_list = []
    converted_ids = []    # original structure indices that converted successfully
    mixed_pbc_ids = []    # original structure indices with mixed pbc (e.g. [T,T,F] slabs)
    fromase_error_ids = []  # original structure indices that failed from_ase
    for idx in tqdm(surviving, desc="converting to AtomicData"):
        s = structures[idx]
        pbc = s.pbc

        if not (pbc.all() or (~pbc).all()):
            mixed_pbc_ids.append(idx)
            continue

        atoms_clean = s.copy()
        atoms_clean.calc = None
        for key in ["energy", "forces", "stress", "REF_energy", "REF_forces", "REF_stress"]:
            atoms_clean.info.pop(key, None)
            atoms_clean.arrays.pop(key, None)

        try:
            data = AtomicData.from_ase(atoms_clean, r_edges=False, task_name=TASK)
            data_list.append(data)
            converted_ids.append(idx)
        except Exception as e:
            fromase_error_ids.append(idx)
            print(f"WARNING: from_ase failed for structure {idx}: {e}")

    print(f"filtered out {len(mixed_pbc_ids)} structures with mixed pbc")
    print(f"converted {len(data_list)} structures ({len(fromase_error_ids)} from_ase errors)")

    atom_bb, atom_ll = get_uma_features_batched(
        data_list, task=TASK, predictor=predictor
    )

    # post-inference NaN detection — check both BB and LL
    nan_ids_local = [
        i for i in range(len(data_list))
        if torch.isnan(atom_ll[i]).any() or torch.isnan(atom_bb[i]).any()
    ]
    nan_ids = [converted_ids[i] for i in nan_ids_local]
    nan_ids_set = set(nan_ids)
    kept_ids = [i for i in converted_ids if i not in nan_ids_set]

    if nan_ids:
        print(f"WARNING: {len(nan_ids)} structures produced NaN features: {nan_ids[:20]}{'...' if len(nan_ids) > 20 else ''}")
    else:
        print("No NaN features detected.")

    ids_path = f"{OUTPUT_DIR}/{MODEL_NAME}-{TASK}_ids.npz"
    np.savez(ids_path, kept=np.array(kept_ids, dtype=np.int64))
    print(f"saved indices to {ids_path}")

    valid_local = [i for i in range(len(data_list)) if i not in set(nan_ids_local)]

    if VARIANT in ["LL", "BOTH"]:
        ll_concat = torch.cat([atom_ll[i] for i in valid_local], dim=0).numpy()
        ll_path = f"{OUTPUT_DIR}/{MODEL_NAME}-{TASK}_ll.npy"
        np.save(ll_path, ll_concat)
        print(f"Per-atom LL features shape: {ll_concat.shape}, saved to {ll_path}")

    if VARIANT in ["BB", "BOTH"]:
        bb_concat = torch.cat([atom_bb[i] for i in valid_local], dim=0).numpy()
        bb_path = f"{OUTPUT_DIR}/{MODEL_NAME}-{TASK}_bb.npy"
        np.save(bb_path, bb_concat)
        print(f"Per-atom BB features shape: {bb_concat.shape}, saved to {bb_path}")
