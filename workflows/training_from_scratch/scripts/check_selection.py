import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import metatomic.torch  # noqa: F401
from metatrain.utils.io import model_from_checkpoint

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
from common_bd import EVAL_OPTIONS, load_predictions, read_truth  # noqa: E402

# recomputes metatrain's native validation metrics for a saved snapshot on the combined validation file and compares
# them with the logged values of the same epoch, then checks that the stored best epoch is the argmin of the logged
# energy MAE (per atom) times forces MAE
MTT = "${HOME}/venv-metatrain-fix/bin/mtt"
S = Path(__file__).resolve().parent.parent
VAL = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v2/manifests/xyz/val_combined_1250.xyz")
LINE = re.compile(r"Epoch: +(\d+) \|.*validation energy MAE \(per atom\): +([0-9.]+) meV.*validation forces MAE: +([0-9.]+) meV/A")
TOL_E, TOL_F = 0.3, 0.3


def native_metrics(e_pred, f_pred, truth):
    natoms, energy, forces, cids, _ = truth
    e_term = float(np.mean(np.abs(e_pred - energy) / natoms)) * 1000
    components = np.concatenate([(f - t).ravel() for f, t in zip(f_pred, forces)])
    f_term = float(np.mean(np.abs(components))) * 1000
    return e_term, f_term


def main():
    run_dir = Path(sys.argv[1])
    epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = run_dir / "selection_check"
    out.mkdir(exist_ok=True)
    logged = {int(m.group(1)): (float(m.group(2)), float(m.group(3))) for m in LINE.finditer((run_dir / "train.log").read_text(errors="replace"))}
    snapshot = next(run_dir.glob(f"outputs/*/*/model_{epoch}.ckpt"))
    exported = out / f"model_{epoch}.pt"
    model_from_checkpoint(torch.load(snapshot, map_location="cpu", weights_only=False), context="restart").export().save(str(exported))
    (out / "eval.yaml").write_text(EVAL_OPTIONS.format(xyz=VAL))
    pred = out / f"val_pred_{epoch}.xyz"
    result = subprocess.run([MTT, "eval", str(exported), str(out / "eval.yaml"), "-o", pred.name, "-b", "16", "--no-warm-up"], cwd=out, capture_output=True, text=True)
    (out / f"eval_{epoch}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and pred.exists(), "mtt eval failed"
    truth = read_truth(VAL)
    natoms, energy, forces, cids, _ = truth
    e_pred, f_pred = load_predictions(pred, cids, truth)
    e_all, f_all = native_metrics(e_pred, f_pred, truth)
    subsets = {}
    for name, sl in (("random_1000", slice(0, 1000)), ("madcore_250", slice(1000, 1250))):
        sub = (natoms[sl], energy[sl], forces[sl], cids[sl], None)
        subsets[name] = dict(zip(("energy_mae_per_atom_meV", "forces_mae_meV_per_A"), native_metrics(e_pred[sl], f_pred[sl], sub)))
    e_log, f_log = logged[epoch]
    best = torch.load(run_dir / "model.ckpt", map_location="cpu", weights_only=False)["best_epoch"]
    products = {e: v[0] * v[1] for e, v in logged.items()}
    argmin = min(products, key=products.get)
    report = {"snapshot": str(snapshot), "epoch": epoch, "logged": {"energy_mae_per_atom_meV": e_log, "forces_mae_meV_per_A": f_log, "mae_prod": e_log * f_log},
              "recomputed_combined": {"energy_mae_per_atom_meV": e_all, "forces_mae_meV_per_A": f_all, "mae_prod": e_all * f_all},
              "recomputed_subsets": subsets, "structures": {"total": len(cids), "random": 1000, "madcore": 250},
              "best_epoch_in_checkpoint": best, "argmin_of_logged_mae_prod": argmin, "logged_mae_prod_by_epoch": products,
              "tolerance_meV": {"energy": TOL_E, "forces": TOL_F},
              "formula": "energy term = mean over structures of |dE|/natoms; force term = mean over all force components of |dF|; selection = argmin over epochs of their product"}
    report["passed"] = abs(e_all - e_log) <= TOL_E and abs(f_all - f_log) <= TOL_F and best == argmin
    json.dump(report, open(run_dir / "selection_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "logged_mae_prod_by_epoch"}, indent=1))
    print("SELECTION CHECK PASSED" if report["passed"] else "SELECTION CHECK FAILED")


if __name__ == "__main__":
    main()
