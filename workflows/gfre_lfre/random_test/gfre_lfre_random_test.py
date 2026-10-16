import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement

import numpy as np
import skmatter
from skmatter.linear_model import _ridge
from skmatter.metrics import _reconstruction_measures
from skmatter.preprocessing import _data

from common import FEATURE_DIR, HERE, ROOT

PARENT = os.path.dirname(HERE)
OUT_DIR = f"{HERE}/results_random_test"
VALID = f"{ROOT}/phace-atomic/mad-core-valid"
CONVERGED_TSV = f"{VALID}/experiments/voronoi_weighted/pbe_20k/manifests/test_random.tsv"
RELEASE_IDS_TSV = f"{VALID}/dft/lda_pbe/sets/random_test_6k/release_ids.tsv"

spec = importlib.util.spec_from_file_location("gfre_lfre_union", f"{PARENT}/gfre_lfre_union.py")
union = importlib.util.module_from_spec(spec)
spec.loader.exec_module(union)
ORDER = union.ORDER


# plain tab split, SMILES ids contain the hash sign that numpy reads as a comment
def read_columns(path, *columns):
    with open(path) as handle:
        next(handle)
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]

    if len(columns) == 1:
        return [row[columns[0]] for row in rows]

    return [tuple(row[c] for c in columns) for row in rows]


def pooled_descriptors():
    meta = np.load(f"{FEATURE_DIR}/frames_meta.npz")
    natoms = meta["natoms"]
    offsets = np.concatenate([[0], np.cumsum(natoms)])
    descriptors = {}
    exclusions = {}

    for name in ORDER:
        features = np.load(f"{FEATURE_DIR}/{name}_ll.npy", mmap_mode="r")
        assert features.shape[0] == offsets[-1], f"{name} has {features.shape[0]} rows for {offsets[-1]} atoms"

        descriptors[name] = np.vstack([
            union.pool_mean_std(np.asarray(features[offsets[i]:offsets[i + 1]], dtype=np.float64))
            for i in range(len(natoms))
        ])
        with open(f"{FEATURE_DIR}/{name}_exclusions.json") as handle:
            exclusions[name] = json.load(handle)["excluded"]

        counts = {reason: len(frames) for reason, frames in exclusions[name].items()}
        print(f"  {name} pooled {descriptors[name].shape}, excluded {counts}", flush=True)

    return meta, descriptors, exclusions


def usable_mask(descriptors, max_norm_ratio, log):
    finite = np.ones(len(next(iter(descriptors.values()))), bool)
    finite_per_model = {}
    for name, values in descriptors.items():
        ok = np.isfinite(values).all(axis=1)
        finite_per_model[name] = int(ok.sum())
        finite &= ok
    finite_in_all = int(finite.sum())

    norm_drops = {}
    for name, values in descriptors.items():
        norms = np.linalg.norm(values, axis=1)
        blown_up = norms > max_norm_ratio * np.median(norms[finite])
        norm_drops[name] = int(blown_up[finite].sum())
        finite &= ~blown_up

    log["finite_per_model"] = finite_per_model
    log["finite_in_all"] = finite_in_all
    log["norm_filter_drops"] = norm_drops
    log["usable"] = int(finite.sum())

    return finite


def digest(payload):
    if isinstance(payload, np.ndarray):
        payload = np.ascontiguousarray(payload).tobytes()
    if isinstance(payload, str):
        payload = payload.encode()

    return hashlib.sha256(payload).hexdigest()


def fingerprint(aligned, keep, test_idx, args):
    estimator = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in union.ESTIMATOR_KWARGS.items()
    }
    code = "".join([
        inspect.getsource(union.pool_mean_std),
        inspect.getsource(union.fre_one),
        inspect.getsource(_reconstruction_measures),
        inspect.getsource(_ridge),
        inspect.getsource(_data),
    ])

    return {
        "descriptors": {name: digest(values) for name, values in aligned.items()},
        "kept": digest(np.asarray(keep, dtype=np.int64)),
        "test_idx": digest(np.asarray(test_idx, dtype=np.int64)),
        "settings": {
            "max_norm_ratio": args.max_norm_ratio,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "lfre_neighbours": "larger of the two descriptor dimensions",
        },
        "estimator": estimator,
        "code": digest(code),
        "skmatter_version": skmatter.__version__,
    }


def check_provenance(current, outputs, path):
    existing = [output for output in outputs if os.path.exists(output)]
    if not existing:
        return

    stored = None
    if os.path.exists(path):
        with open(path) as handle:
            stored = json.load(handle)
    if stored == current:
        print("existing results match the current fingerprint, resuming", flush=True)
        return

    if stored is None:
        reason = "no fingerprint"
    else:
        reason = "fingerprint differs in " + ", ".join(sorted(key for key in current if stored.get(key) != current[key]))
    archive = f"{OUT_DIR}/superseded/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(archive, exist_ok=True)
    for output in existing + ([path] if os.path.exists(path) else []):
        shutil.move(output, archive)
    print(f"{reason}, moved {len(existing)} existing outputs to {archive} and recomputing", flush=True)


def run_arm(tag, aligned, test_idx, kinds, args):
    for kind in kinds:
        jobs = args.n_jobs if kind == "gfre" else args.lfre_n_jobs
        path = f"{OUT_DIR}/{tag}_{kind}_seed{args.seed}.json"
        results = {}
        if os.path.exists(path):
            with open(path) as handle:
                results = json.load(handle)
        pending = [(a, b) for a, b in combinations_with_replacement(ORDER, 2) if f"{a}__{b}" not in results]

        def compute(pair):
            a, b = pair
            ab = float(union.fre_one(aligned[a], aligned[b], kind, test_idx, jobs))
            ba = ab
            if a != b:
                ba = float(union.fre_one(aligned[b], aligned[a], kind, test_idx, jobs))

            return a, b, ab, ba

        with ThreadPoolExecutor(max_workers=args.pair_workers) as pool:
            for future in as_completed([pool.submit(compute, pair) for pair in pending]):
                a, b, ab, ba = future.result()
                results[f"{a}__{b}"] = {"A": a, "B": b, "A_to_B": ab, "B_to_A": ba}
                with open(path, "w") as handle:
                    json.dump(results, handle, indent=2)
                print(f"  {tag} {kind} {a} and {b} {ab:.3f} {ba:.3f}", flush=True)

        print(f"saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-norm-ratio", type=float, default=50.0)
    parser.add_argument("--test-fraction", type=float, default=4000 / 15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gfre", action="store_true")
    parser.add_argument("--with-lfre", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--lfre-n-jobs", type=int, default=32)
    parser.add_argument("--pair-workers", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.n_jobs == -1:
        args.n_jobs = os.cpu_count() or 8
    kinds = ([] if args.no_gfre else ["gfre"]) + (["lfre"] if args.with_lfre else [])
    summary = {"seed": args.seed, "max_norm_ratio": args.max_norm_ratio, "models": ORDER}

    # the random MAD-COSMO test selection
    print("pooling the random test selection", flush=True)
    meta, descriptors, exclusions = pooled_descriptors()
    log = {
        "n_structures": int(len(meta["natoms"])),
        "n_atoms": int(meta["natoms"].sum()),
        "exclusions": exclusions,
    }
    keep = np.where(usable_mask(descriptors, args.max_norm_ratio, log))[0]

    # the frames carry selection time ids, the labeled test set is keyed by release ids
    configuration_ids = meta["configuration_id"].astype(str)
    release_of = dict(read_columns(RELEASE_IDS_TSV, 0, 1))
    converged = set(read_columns(CONVERGED_TSV, 1))
    in_converged = np.array([release_of.get(cid, cid) in converged for cid in configuration_ids])
    log["n_in_converged_5847"] = int(in_converged.sum())
    log["usable_in_converged_5847"] = int(in_converged[keep].sum())

    split_files, counts = np.unique(meta["split_file"][keep], return_counts=True)
    log["usable_by_split_file"] = {str(name): int(count) for name, count in zip(split_files, counts)}

    n = len(keep)
    n_test = int(round(args.test_fraction * n))
    rng = np.random.default_rng(args.seed)
    test_idx = np.sort(rng.choice(n, n_test, replace=False))
    log["n_fit"] = n - n_test
    log["n_test"] = n_test
    summary["random"] = log
    print(f"random usable {n}, fit {n - n_test}, test {n_test}", flush=True)

    aligned = {name: np.ascontiguousarray(values[keep]) for name, values in descriptors.items()}
    current = fingerprint(aligned, keep, test_idx, args)
    fingerprint_path = f"{OUT_DIR}/fingerprint_seed{args.seed}.json"
    outputs = [
        f"{OUT_DIR}/selection_random_seed{args.seed}.npz",
        f"{OUT_DIR}/summary_seed{args.seed}.json",
        f"{OUT_DIR}/random_gfre_seed{args.seed}.json",
        f"{OUT_DIR}/random_lfre_seed{args.seed}.json",
    ]
    check_provenance(current, outputs, fingerprint_path)
    with open(fingerprint_path, "w") as handle:
        json.dump(current, handle, indent=1)

    np.savez(
        f"{OUT_DIR}/selection_random_seed{args.seed}.npz",
        kept=keep,
        test_idx=test_idx,
        configuration_id=configuration_ids[keep],
    )
    with open(f"{OUT_DIR}/summary_seed{args.seed}.json", "w") as handle:
        json.dump(summary, handle, indent=1)

    run_arm("random", aligned, test_idx, kinds, args)
    print("done", flush=True)


if __name__ == "__main__":
    main()
