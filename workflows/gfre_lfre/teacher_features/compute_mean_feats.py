"""
Compute mean-pooled (per-structure) features from per-atom features.

Usage:
    python compute_mean_feats.py <DATASET_PATH> <FEATS_NPY> <OUTPUT_NPY> [--ids IDS_NPZ] [--splits SPLITS_JSON]

Examples:
    # Full dataset
    python compute_mean_feats.py data/matpes/matpes_pbe.xyz data/matpes/mace/mace_ll.npy data/matpes/mace/mace_ll_mean.npy

    # With filtered IDs (from MACE)
    python compute_mean_feats.py data/matpes/matpes_pbe.xyz data/matpes/mace/mace_ll.npy data/matpes/mace/mace_ll_mean.npy --ids data/matpes/mace/mace_ids.npz

    # Validation split only
    python compute_mean_feats.py data/matpes/matpes_pbe.xyz data/matpes/mace/mace_ll_val.npy data/matpes/mace/mace_ll_val_mean.npy --splits data/matpes/MatPES-PBE-split.json.gz
"""

import os
import sys
import gzip
import json

import numpy as np


def read_natoms_from_xyz(path):
    natoms = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                n = int(line)
                natoms.append(n)
                next(f)
                for _ in range(n):
                    next(f)
            except ValueError:
                continue
    return np.array(natoms, dtype=np.int64)


def mean_pool(atom_feats_mmap, natoms, starts=None):
    n_structs = len(natoms)
    feat_dim = atom_feats_mmap.shape[1]
    out = np.empty((n_structs, feat_dim), dtype=np.float32)
    if starts is None:
        offset = 0
        for i, n in enumerate(natoms):
            out[i] = atom_feats_mmap[offset : offset + n].mean(axis=0)
            offset += n
    else:
        for i, (s, n) in enumerate(zip(starts, natoms)):
            out[i] = atom_feats_mmap[s : s + n].mean(axis=0)
    return out


def main():
    args = sys.argv[1:]
    ids_path = None
    splits_path = None

    if "--ids" in args:
        idx = args.index("--ids")
        ids_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--splits" in args:
        idx = args.index("--splits")
        splits_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if len(args) < 3:
        print("Usage: python compute_mean_feats.py <DATASET_PATH> <FEATS_NPY> <OUTPUT_NPY> [--ids IDS_NPZ] [--splits SPLITS_JSON]")
        sys.exit(1)

    dataset_path = args[0]
    feats_path = args[1]
    output_path = args[2]

    print(f"Dataset: {dataset_path}")
    print(f"Features: {feats_path}")
    print(f"Output: {output_path}")
    if ids_path:
        print(f"IDs file: {ids_path}")
    if splits_path:
        print(f"Splits: {splits_path} (validation only)")

    print("\nReading natoms from XYZ...")
    natoms_all = read_natoms_from_xyz(dataset_path)
    n_structs_total = len(natoms_all)
    print(f"  {n_structs_total} structures, {natoms_all.sum()} atoms total")

    if splits_path:
        # Use validation split
        with gzip.open(splits_path, 'rt') as f:
            splits = json.load(f)
        kept = np.array(splits["valid"], dtype=np.int64)
        print(f"  Using validation split: {len(kept)} structures")
    elif ids_path:
        # Use filtered IDs (e.g., from MACE)
        kept = np.load(ids_path)["kept"]
        print(f"  Using filtered IDs: {len(kept)} structures")
    else:
        kept = np.arange(n_structs_total, dtype=np.int64)

    natoms_kept = natoms_all[kept]
    total_atoms = int(natoms_all.sum())
    kept_atoms = int(natoms_kept.sum())

    if not os.path.exists(feats_path):
        print(f"ERROR: features file not found: {feats_path}")
        sys.exit(1)

    atom_feats = np.load(feats_path, mmap_mode="r")
    nrows = atom_feats.shape[0]
    print(f"\nPer-atom features shape: {atom_feats.shape}")

    print("Mean-pooling per-atom → per-structure...")
    if nrows == total_atoms:
        # full per-atom file: index by kept (works for any kept ordering)
        cum_all = np.concatenate(([0], np.cumsum(natoms_all)))
        starts = cum_all[kept]
        struct_feats = mean_pool(atom_feats, natoms_kept, starts)
    elif nrows == kept_atoms:
        # pre-filtered file with rows already in `kept` order
        struct_feats = mean_pool(atom_feats, natoms_kept)
    else:
        print(f"ERROR: atom count mismatch: file has {nrows} rows; "
              f"expected {total_atoms} (full) or {kept_atoms} (pre-filtered)")
        sys.exit(1)
    print(f"Per-structure features shape: {struct_feats.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, struct_feats)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
