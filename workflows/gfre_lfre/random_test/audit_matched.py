import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement

import numpy as np

from common import HERE

PARENT = os.path.dirname(HERE)
UNION = f"{PARENT}/results_union"
AUDIT = f"{HERE}/results_random_test/audit_matched"
ORIGINAL = f"{AUDIT}/original"
SEED = 0
N = 5905
N_TEST = 1575
LFRE_JOBS = 32
PAIR_WORKERS = 7

spec = importlib.util.spec_from_file_location("gfre_lfre_union", f"{PARENT}/gfre_lfre_union.py")
union = importlib.util.module_from_spec(spec)
spec.loader.exec_module(union)


def main():
    cache = np.load(f"{UNION}/descriptors_seed{SEED}.npz")
    selection = np.load(f"{UNION}/selection_seed{SEED}.npz")
    gids = selection["gids"]
    row_of = {int(gid): i for i, gid in enumerate(cache["gids"])}
    rows = np.array([row_of[int(gid)] for gid in gids])

    # the same draws as the removed control arm, a uniform subsample of the 15000 and its test split
    rng = np.random.default_rng(SEED)
    subsample = np.sort(rng.choice(len(gids), N, replace=False))
    test_idx = np.sort(rng.choice(N, N_TEST, replace=False))

    original = np.load(f"{ORIGINAL}/selection_madnext_seed{SEED}.npz")
    checks = {
        "same_gids_as_original": bool(np.array_equal(original["gids"], gids[subsample])),
        "same_test_idx_as_original": bool(np.array_equal(original["test_idx"], test_idx)),
        "subset_of_main_selection": bool(np.isin(gids[subsample], gids).all()),
        "n": int(N),
        "n_fit": int(N - N_TEST),
        "n_test": int(N_TEST),
        "main_test_idx_overlap_with_subsample_test": int(len(
            set(gids[selection["test_idx"]].tolist()) & set(gids[subsample][test_idx].tolist())
        )),
    }
    print(checks, flush=True)

    aligned = {name: np.ascontiguousarray(cache[name][rows[subsample]]) for name in union.ORDER}
    assert all(np.isfinite(values).all() for values in aligned.values())

    for kind in ["gfre", "lfre"]:
        jobs = os.cpu_count() if kind == "gfre" else LFRE_JOBS
        pairs = list(combinations_with_replacement(union.ORDER, 2))

        def compute(pair):
            a, b = pair
            ab = float(union.fre_one(aligned[a], aligned[b], kind, test_idx, jobs))
            ba = ab
            if a != b:
                ba = float(union.fre_one(aligned[b], aligned[a], kind, test_idx, jobs))

            return a, b, ab, ba

        results = {}
        with ThreadPoolExecutor(max_workers=PAIR_WORKERS) as pool:
            for future in as_completed([pool.submit(compute, pair) for pair in pairs]):
                a, b, ab, ba = future.result()
                results[f"{a}__{b}"] = {"A": a, "B": b, "A_to_B": ab, "B_to_A": ba}

        with open(f"{ORIGINAL}/madnext_{kind}_seed{SEED}.json") as handle:
            stored = json.load(handle)
        difference = max(
            max(abs(results[key]["A_to_B"] - stored[key]["A_to_B"]), abs(results[key]["B_to_A"] - stored[key]["B_to_A"]))
            for key in results
        )
        checks[f"{kind}_max_difference_to_original"] = float(difference)
        print(f"{kind} recomputed, max difference to the original {difference:.3e}", flush=True)

        with open(f"{AUDIT}/madnext_{kind}_seed{SEED}.json", "w") as handle:
            json.dump(results, handle, indent=2)

    with open(f"{AUDIT}/checks_seed{SEED}.json", "w") as handle:
        json.dump(checks, handle, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
