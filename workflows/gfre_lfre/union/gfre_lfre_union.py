from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement

import numpy as np
from skmatter.linear_model import Ridge2FoldCV
from skmatter.metrics import global_reconstruction_error, local_reconstruction_error

ROOT = "${DEV_ROOT}"
ATOMIC_COMBINED = f"{ROOT}/phace-atomic/data/atomic_combined"
UNION_TEST = f"{ROOT}/phace-atomic/data/union_test"
MAD15 = f"{ROOT}/phace-atomic/data/mad15"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = f"{HERE}/results_union"

N_UNION = 677454
BLOCKS = [
    ("mad15", 0, 216803, f"{ROOT}/data/mad-1.5/mad-1.5-r2scan-all.xyz"),
    ("mad_next", 216803, 638443, f"{ROOT}/phace-atomic/data/mad_next/mad_next.xyz"),
    ("mad_sol", 638443, 677454, f"{ROOT}/phace-atomic/data/mad_sol/mad_sol.xyz"),
]
MAD15_END = 216803
SUBSETS_CACHE = f"{UNION_TEST}/union_subsets.npz"

TEACHERS = {
    "pet-mad": ("pet.bin", 512),
    "mace-omat": ("mace.bin", 144),
    "uma-omat": ("uma.bin", 128),
    "dpa-omat": ("dpa_omat.bin", 240),
    "dpa-omol": ("dpa_omol.bin", 240),
}
SPLIT_MODELS = {
    "orb-v3-omat": (f"{MAD15}/orb/orb-v3-omat_ll.npy",
                    f"{UNION_TEST}/orb-v3-omat_nextsol_ll.npy",
                    f"{UNION_TEST}/orb-v3-omat_nextsol_meta.npz"),
    "phace-atomic": (f"{MAD15}/phace_atomic/phace_atomic_ep92_ll.npy",
                     f"{UNION_TEST}/phace_atomic_ep92_nextsol_ll.npy",
                     f"{UNION_TEST}/phace_atomic_ep92_nextsol_meta.npz"),
}
ORDER = ["pet-mad", "mace-omat", "uma-omat", "dpa-omat", "dpa-omol",
         "orb-v3-omat", "phace-atomic"]

ESTIMATOR_KWARGS = dict(alphas=np.geomspace(1e-5, 0.9, 20), alpha_type="relative",
                        regularization_method="cutoff", random_state=0x5F3759DF,
                        shuffle=True, scoring="neg_root_mean_squared_error", n_jobs=1)


def read_subsets(path):
    subsets = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                n = int(stripped)
            except ValueError:
                continue
            comment = next(f)
            match = re.search(r"subset=(\S+)", comment)
            subsets.append(match.group(1) if match else "unknown")
            for _ in range(n):
                next(f)
    return np.array(subsets)


def union_subsets():
    if os.path.exists(SUBSETS_CACHE):
        return np.load(SUBSETS_CACHE, allow_pickle=False)["subsets"]

    parts = []
    for name, block_start, block_end, xyz in BLOCKS:
        labels = read_subsets(xyz)
        assert len(labels) == block_end - block_start, \
            f"{name}: {len(labels)} comments != {block_end - block_start}"
        parts.append(labels)
        print(f"  {name}: {len(labels):,} subset labels ({len(np.unique(labels))} unique)", flush=True)

    subsets = np.concatenate(parts)
    os.makedirs(UNION_TEST, exist_ok=True)
    np.savez(SUBSETS_CACHE, subsets=subsets)
    return subsets


def stratified(labels, n, seed=0):
    unique, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(seed)

    chosen = []
    for label, fraction in zip(unique, counts / counts.sum()):
        members = np.where(labels == label)[0]
        k = min(len(members), max(1, round(fraction * n)))
        chosen.append(rng.choice(members, k, replace=False))

    picks = np.sort(np.concatenate(chosen))
    if len(picks) > n:
        picks = np.sort(rng.choice(picks, n, replace=False))
    return picks


def stratified_target(pool_labels, weights, n, seed=0):
    """Draw n items from a pool, allocating each subset a share of n proportional
    to `weights` (the candidate-pool counts) renormalized over the subsets that
    survive the descriptor filters, capped by availability, remainders largest-first."""
    rng = np.random.default_rng(seed)
    labels = np.unique(pool_labels)
    members = {label: np.where(pool_labels == label)[0] for label in labels}
    capacity = {label: len(members[label]) for label in labels}
    quota = {label: 0 for label in labels}

    remaining = min(n, sum(capacity.values()))
    open_labels = list(labels)
    while remaining > 0 and open_labels:
        total_weight = sum(weights[label] for label in open_labels)
        shares = {label: remaining * weights[label] / total_weight for label in open_labels}
        capped = [label for label in open_labels
                  if shares[label] >= capacity[label] - quota[label]]
        if capped:
            for label in capped:
                remaining -= capacity[label] - quota[label]
                quota[label] = capacity[label]
                open_labels.remove(label)
            continue
        floors = {label: int(shares[label]) for label in open_labels}
        for label in open_labels:
            quota[label] += floors[label]
        remaining -= sum(floors.values())
        for label in sorted(open_labels, key=lambda l: shares[l] - floors[l],
                            reverse=True)[:remaining]:
            quota[label] += 1
        remaining = 0

    picks = [rng.choice(members[label], quota[label], replace=False)
             for label in labels if quota[label]]
    return np.sort(np.concatenate(picks))


def pool_mean_std(rows):
    valid = rows[np.isfinite(rows).all(axis=1)]
    if len(valid) == 0:
        return np.full(2 * rows.shape[1], np.nan, dtype=np.float64)
    return np.concatenate([valid.mean(axis=0), valid.std(axis=0)])


def build_descriptors(global_ids, atom_offsets):
    n_atoms = int(atom_offsets[-1])

    sources = {}
    for name, (bin_name, dim) in TEACHERS.items():
        sources[name] = np.memmap(f"{ATOMIC_COMBINED}/{bin_name}", dtype="float32",
                                  mode="r", shape=(n_atoms, dim))
    for name, (mad15_npy, nextsol_npy, nextsol_meta) in SPLIT_MODELS.items():
        mad15_features = np.load(mad15_npy, mmap_mode="r")
        nextsol_features = np.load(nextsol_npy, mmap_mode="r")
        meta = np.load(nextsol_meta)
        nextsol_offsets = np.concatenate([[0], np.cumsum(meta["natoms"])])
        lookup = {int(gid): (int(nextsol_offsets[i]), int(meta["natoms"][i]))
                  for i, gid in enumerate(meta["gids"])}
        sources[name] = (mad15_features, nextsol_features, lookup)

    descriptors = {}
    for name, source in sources.items():
        pooled = []
        for gid in global_ids:
            row_start, row_end = int(atom_offsets[gid]), int(atom_offsets[gid + 1])
            if isinstance(source, tuple):
                mad15_features, nextsol_features, lookup = source
                if gid < MAD15_END:
                    rows = np.asarray(mad15_features[row_start:row_end], dtype=np.float64)
                elif int(gid) in lookup:
                    offset, count = lookup[int(gid)]
                    rows = np.asarray(nextsol_features[offset:offset + count], dtype=np.float64)
                else:
                    rows = np.full((row_end - row_start, mad15_features.shape[1]), np.nan)
            else:
                rows = np.asarray(source[row_start:row_end], dtype=np.float64)
            pooled.append(pool_mean_std(rows))
        descriptors[name] = np.vstack(pooled)
        print(f"  {name}: pooled {descriptors[name].shape}", flush=True)
    return descriptors


def fre_one(X, Y, kind, test_idx, n_jobs):
    estimator = Ridge2FoldCV(**ESTIMATOR_KWARGS)
    if kind == "gfre":
        return global_reconstruction_error(X, Y, estimator=estimator, test_idx=test_idx)
    k = max(X.shape[1], Y.shape[1])
    return local_reconstruction_error(X, Y, n_local_points=k, n_jobs=max(1, n_jobs),
                                      estimator=estimator, test_idx=test_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=15000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--max-norm-ratio", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gfre", action="store_true")
    parser.add_argument("--with-lfre", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--lfre-n-jobs", type=int, default=6)
    parser.add_argument("--pair-workers", type=int, default=7)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    atom_offsets = np.load(f"{ATOMIC_COMBINED}/na.npy")
    assert len(atom_offsets) == N_UNION + 1, \
        f"na.npy has {len(atom_offsets)} entries, expected {N_UNION + 1}"
    test_ids = np.loadtxt(f"{ATOMIC_COMBINED}/test_indices_in_all.txt", dtype=np.int64)
    print(f"union test split: {len(test_ids):,} structures", flush=True)

    print("subset labels ...", flush=True)
    labels = union_subsets()[test_ids]

    candidates_local = stratified(labels, args.n_total * 3, seed=args.seed)
    global_ids = test_ids[candidates_local]
    candidate_labels = labels[candidates_local]
    per_block = {name: int(((global_ids >= start) & (global_ids < end)).sum())
                 for name, start, end, _ in BLOCKS}
    print(f"candidates: {len(global_ids):,} stratified over "
          f"{len(np.unique(labels))} subsets; by block: {per_block}", flush=True)

    descriptor_cache = f"{OUT_DIR}/descriptors_seed{args.seed}.npz"
    if os.path.exists(descriptor_cache):
        cached = np.load(descriptor_cache)
        assert np.array_equal(cached["gids"], global_ids), \
            f"{descriptor_cache} was built for a different candidate set"
        descriptors = {name: cached[name] for name in ORDER}
        print(f"loaded pooled descriptors from {descriptor_cache}", flush=True)
    else:
        print("pooling [mean, std] descriptors ...", flush=True)
        descriptors = build_descriptors(global_ids, atom_offsets)
        np.savez(descriptor_cache, gids=global_ids, **descriptors)
        print(f"saved pooled descriptors => {descriptor_cache}", flush=True)

    finite = np.ones(len(global_ids), bool)
    for values in descriptors.values():
        finite &= np.isfinite(values).all(axis=1)
    print(f"finite in all models: {int(finite.sum()):,}/{len(global_ids):,}", flush=True)

    for name, values in descriptors.items():
        norms = np.linalg.norm(values, axis=1)
        threshold = args.max_norm_ratio * np.median(norms[finite])
        blown_up = norms > threshold
        if blown_up[finite].any():
            print(f"  norm-filter ({name}): drops {int(blown_up[finite].sum())}", flush=True)
        finite &= ~blown_up

    keep = np.where(finite)[0]
    rng = np.random.default_rng(args.seed)
    if len(keep) > args.n_total:
        weights = dict(zip(*np.unique(candidate_labels, return_counts=True)))
        keep = keep[stratified_target(candidate_labels[keep], weights,
                                      args.n_total, seed=args.seed)]
    kept_labels, kept_counts = np.unique(candidate_labels[keep], return_counts=True)
    print("  subset composition (candidates -> usable -> selected, %):", flush=True)
    for label, count in zip(kept_labels, kept_counts):
        in_pool = int((candidate_labels == label).sum())
        usable = int((candidate_labels[finite] == label).sum())
        print(f"    {label:24s} {100 * in_pool / len(candidate_labels):6.2f}"
              f" {100 * usable / max(int(finite.sum()), 1):6.2f}"
              f" {100 * count / len(keep):6.2f}   ({count:,}/{usable:,})", flush=True)
    dropped = sorted(str(label) for label in
                     set(np.unique(candidate_labels)) - set(kept_labels))
    if dropped:
        print(f"  subsets with no usable structures: {dropped}", flush=True)
    aligned = {name: np.ascontiguousarray(values[keep])
               for name, values in descriptors.items()}

    n_kept = len(keep)
    n_test = min(args.n_test, n_kept // 2)
    test_idx = np.sort(rng.choice(n_kept, n_test, replace=False))
    n_jobs = args.n_jobs if args.n_jobs != -1 else (os.cpu_count() or 8)
    kept_blocks = {name: int(((global_ids[keep] >= start) & (global_ids[keep] < end)).sum())
                   for name, start, end, _ in BLOCKS}
    print(f"usable structures={n_kept:,}  ridge-test={n_test:,}  by block: {kept_blocks}\n", flush=True)
    np.savez(f"{OUT_DIR}/selection_seed{args.seed}.npz",
             gids=global_ids[keep], test_idx=test_idx)

    def run_kind(kind, jobs, pair_workers):
        path = f"{OUT_DIR}/union_{kind}_seed{args.seed}.json"
        results = json.load(open(path)) if os.path.exists(path) else {}
        pending = [(a, b) for a, b in combinations_with_replacement(ORDER, 2)
                   if f"{a}__{b}" not in results]

        def compute(pair):
            a, b = pair
            ab = float(fre_one(aligned[a], aligned[b], kind, test_idx, jobs))
            ba = float(fre_one(aligned[b], aligned[a], kind, test_idx, jobs)) if a != b else ab
            return a, b, ab, ba

        with ThreadPoolExecutor(max_workers=pair_workers) as pool:
            futures = [pool.submit(compute, pair) for pair in pending]
            for future in as_completed(futures):
                a, b, ab, ba = future.result()
                results[f"{a}__{b}"] = {"A": a, "B": b, "A_to_B": ab, "B_to_A": ba}
                json.dump(results, open(path, "w"), indent=2)
                print(f"  {kind}  {a} <-> {b}: {ab:.3f}/{ba:.3f}", flush=True)
        print(f"saved => {path}", flush=True)

    if not args.no_gfre:
        print("GFRE ...", flush=True)
        run_kind("gfre", n_jobs, args.pair_workers)
    if args.with_lfre:
        print("LFRE ...", flush=True)
        run_kind("lfre", args.lfre_n_jobs, args.pair_workers)


if __name__ == "__main__":
    sys.exit(main())
