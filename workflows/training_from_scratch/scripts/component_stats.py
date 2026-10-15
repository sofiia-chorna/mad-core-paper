import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
from common_bd import XYZ, load_predictions, read_truth  # noqa: E402

S = Path(__file__).resolve().parent.parent
SELECTED = S / "reports" / "selected_checkpoints"
PREDS = SELECTED / "preds"
CACHE_MTT = SELECTED / "cache_mtt"
OUT = SELECTED / "cache_stats"
TABLE = SELECTED / "metrics_mtt.tsv"
PREFIX = "scratchv3-s"
TESTS = ("test_random", "test_madcore")


# per structure sums of the absolute force component errors and their powers, from which pooled metrics of any
# subset of structures follow exactly
def structure_sums(per_structure):
    absolute = [np.abs(d).ravel() for d in per_structure]
    return {
        "n_components": np.array([a.size for a in absolute]),
        "f_sum_abs": np.array([a.sum() for a in absolute]),
        "f_sum_sq": np.array([(a ** 2).sum() for a in absolute]),
        "f_sum_e8": np.array([(a ** 8).sum() for a in absolute]),
        "f_max_abs": np.array([a.max() for a in absolute]),
    }


def pooled(de, sums):
    e = np.abs(de)
    n = sums["n_components"].sum()
    return {
        "E_mae": e.mean(),
        "E_rmse": np.sqrt((e ** 2).mean()),
        "E_p8": ((e ** 8).mean()) ** (1 / 8),
        "E_sup": e.max(),
        "F_mae": sums["f_sum_abs"].sum() / n,
        "F_rmse": np.sqrt(sums["f_sum_sq"].sum() / n),
        "F_p8": (sums["f_sum_e8"].sum() / n) ** (1 / 8),
        "F_sup": sums["f_max_abs"].max(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(TABLE) as fh:
        table = {(r["test"], r["model"]): r for r in csv.DictReader(fh, delimiter="\t")}
    truths = {test: read_truth(XYZ / f"{test}.xyz") for test in TESTS}
    runs = sorted(p.parent.name for p in (S / "runs").glob(f"{PREFIX}--*/options.yaml"))
    assert len(runs) == 8, runs

    worst = 0.0
    for run in runs:
        for test in TESTS:
            natoms, energy, forces, cids, _ = truths[test]
            e_pred, f_pred = load_predictions(PREDS / f"{run}@{test}.xyz", cids, truths[test])
            de = (e_pred - energy) / natoms * 1000
            per_structure = [(f - t) * 1000 for f, t in zip(f_pred, forces)]
            sums = structure_sums(per_structure)

            # the existing per structure cache and the verified table must be reproduced
            old = np.load(CACHE_MTT / f"{run}@{test}.npz", allow_pickle=True)
            assert (old["configuration_id"].astype(str) == np.array(cids)).all()
            assert np.allclose(old["de"], de, rtol=0, atol=1e-9)
            assert np.allclose(old["df_comp"], sums["f_sum_abs"] / sums["n_components"], rtol=1e-12, atol=1e-9)
            metrics = pooled(de, sums)
            row = table[(test, run)]
            for key, value in metrics.items():
                difference = abs(round(float(value), 3) - float(row[key]))
                worst = max(worst, difference)
                assert difference < 2e-3, (run, test, key, value, row[key])

            np.savez(OUT / f"{run}@{test}.npz", configuration_id=np.array(cids), natoms=natoms, de=de, **sums)
            print(f"{run} {test}: {len(cids)} structures, table reproduced", flush=True)

    print(f"largest difference to metrics_mtt.tsv after rounding to 3 decimals: {worst:.4f}")
    print("COMPONENT STATS DONE")


if __name__ == "__main__":
    main()
