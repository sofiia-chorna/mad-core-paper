import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import metatomic.torch  # noqa: F401

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
import common_bd  # noqa: E402
from common_bd import TESTS, XYZ, errors, load_predictions, metrics, read_truth, run_mtt_eval, sha256  # noqa: E402

# the shared helper defaults to the original environment, every evaluation here must use the corrected one
common_bd.MTT = "${HOME}/venv-metatrain-fix/bin/mtt"
from metatrain.utils.io import model_from_checkpoint  # noqa: E402

# diagnostic evaluation of the FINAL checkpoint of every finished run (last epoch, not the selected one)
S = Path(__file__).resolve().parent.parent
RUNS = S / "runs"
OUT = S / "reports" / "final_checkpoints"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(p.parent for p in RUNS.glob("scratchv3-s--*/options.yaml")):
        log_path = run_dir / "train.log"
        if not log_path.exists() or "Training finished!" not in log_path.read_text(errors="replace"):
            continue
        final_pt = run_dir / "model_final.pt"
        if not final_pt.exists():
            ckpt = torch.load(run_dir / "model.ckpt", map_location="cpu", weights_only=False)
            model_from_checkpoint(ckpt, context="restart").export().save(str(final_pt))
            log(f"{run_dir.name}: exported final checkpoint (epoch {ckpt['epoch']})")
        for test in TESTS:
            truth = read_truth(XYZ / f"{test}.xyz")
            natoms, energy, forces, cids, _ = truth
            tag = f"{run_dir.name}@{test}"
            pred = OUT / f"{tag}.xyz"
            if not pred.exists():
                run_mtt_eval(final_pt, XYZ / f"{test}.xyz", pred, 16, "cuda", OUT / f"{tag}.log")
            e, f = load_predictions(pred, cids, truth)
            de, df = errors(e, f, truth)
            np.savez(OUT / f"{tag}.npz", configuration_id=np.array(cids), de=de, df_mean=df, natoms=natoms, provenance=json.dumps({"model_final_pt_sha256": sha256(final_pt), "batch_size": 16}))
            m = metrics(de, df)
            rows.append({"test": test, "model": run_dir.name, "checkpoint": "final", **{k: round(v, 3) for k, v in m.items()}})
            log(f"{tag} final: E MAE {m['E_mae']:.2f} F MAE {m['F_mae']:.1f}")
    if rows:
        with open(OUT / "metrics_final.tsv", "w") as fh:
            fh.write("\t".join(rows[0]) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[k]) for k in rows[0]) + "\n")
    log("FINAL DONE")


if __name__ == "__main__":
    main()
