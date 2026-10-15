import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
from common_bd import XYZ, load_predictions, read_truth  # noqa: E402

# test metrics with metatrain's definitions, from the prediction files already written by the evaluations (no new
# inference): energy error per structure = |dE| / atoms (meV/atom), statistics over structures; force errors pooled over
# all force components (x, y, z of every atom) of the test set (meV/A), as metatrain's force MAE and RMSE; p8 and SUP use
# the same pooling; per structure caches for the maps hold the mean absolute force component error of each structure
# usage: metatrain_metrics.py --round DIR --prefix scratch-s
parser = argparse.ArgumentParser()
parser.add_argument("--round", required=True)
parser.add_argument("--prefix", required=True)
ARGS = parser.parse_args()
ROUND = Path(ARGS.round).resolve()
TESTS = ("test_random", "test_madcore")
KINDS = {"selected": (ROUND / "reports" / "selected_checkpoints" / "preds", ROUND / "reports" / "selected_checkpoints" / "metrics_mtt.tsv", ROUND / "reports" / "selected_checkpoints" / "cache_mtt"),
         "final": (ROUND / "reports" / "final_checkpoints", ROUND / "reports" / "final_checkpoints" / "metrics_final_mtt.tsv", ROUND / "reports" / "final_checkpoints" / "cache_mtt")}
COLUMNS = ["test", "model", "checkpoint", "n", "atoms", "force_components", "E_mae", "E_rmse", "E_p8", "E_sup", "F_mae", "F_rmse", "F_p8", "F_sup"]


def stats(values):
    a = np.abs(np.asarray(values, dtype=float))
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())), "p8": float((a ** 8).mean() ** (1 / 8)), "sup": float(a.max())}


def main():
    truths = {test: read_truth(XYZ / f"{test}.xyz") for test in TESTS}
    for kind, (pred_dir, table, cache) in KINDS.items():
        cache.mkdir(parents=True, exist_ok=True)
        rows = []
        for run_dir in sorted(p.parent for p in (ROUND / "runs").glob(f"{ARGS.prefix}--*/options.yaml")):
            for test in TESTS:
                pred = pred_dir / f"{run_dir.name}@{test}.xyz"
                if not pred.exists():
                    continue
                truth = truths[test]
                natoms, energy, forces, cids, _ = truth
                e_pred, f_pred = load_predictions(pred, cids, truth)
                de = (e_pred - energy) / natoms * 1000
                per_structure = [(f - t) * 1000 for f, t in zip(f_pred, forces)]
                components = np.concatenate([d.ravel() for d in per_structure])
                e, f = stats(de), stats(components)
                df_comp = np.array([np.abs(d).mean() for d in per_structure])
                np.savez(cache / f"{run_dir.name}@{test}.npz", configuration_id=np.array(cids), de=de, df_comp=df_comp, natoms=natoms)
                rows.append({"test": test, "model": run_dir.name, "checkpoint": kind, "n": len(cids), "atoms": int(natoms.sum()), "force_components": int(components.size),
                             **{f"E_{k}": round(v, 3) for k, v in e.items()}, **{f"F_{k}": round(v, 3) for k, v in f.items()}})
                print(f"{kind:8s} {run_dir.name} {test}: E MAE {e['mae']:.1f} | F MAE {f['mae']:.1f} RMSE {f['rmse']:.1f} p8 {f['p8']:.1f} SUP {f['sup']:.1f} (meV/A, components)", flush=True)
        if rows:
            with open(table, "w") as fh:
                fh.write("\t".join(COLUMNS) + "\n")
                for r in rows:
                    fh.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
            print("wrote", table)
    print("MTT METRICS DONE")


if __name__ == "__main__":
    main()
