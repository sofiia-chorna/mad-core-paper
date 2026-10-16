import json
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import metatomic.torch  # noqa: F401
from metatrain.utils.io import model_from_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
from common_bd import EVAL_OPTIONS, load_predictions, read_truth  # noqa: E402

# and a stride of snapshots nothing here enters checkpoint selection
MTT = "${HOME}/venv-metatrain-fix/bin/mtt"
S = Path(__file__).resolve().parent.parent
RUNS = S / "runs"
OUT = S / "reports" / "validation_subsets"
VAL = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v2/manifests/xyz/val_combined_1250.xyz")
LINE = re.compile(r"Epoch: +(\d+) \|.*validation energy MAE \(per atom\): +([0-9.]+) meV.*validation forces MAE: +([0-9.]+) meV/A")
BEST = re.compile(r"Using best model from epoch (\d+)")
SUBSETS = {"combined": slice(0, 1250), "random_1000": slice(0, 1000), "madcore_250": slice(1000, 1250)}
MAX_POINTS = 60
COLOR = {"combined": "#000000", "random_1000": "#0072B2", "madcore_250": "#E69F00"}
COLUMNS = ["run", "kind", "epoch", "optimizer_updates"] + [f"{k}_{m}" for k in SUBSETS for m in ("E_mae_per_atom", "F_mae", "mae_prod")] + ["logged_E_mae_per_atom", "logged_F_mae"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def native(e_pred, f_pred, natoms, energy, forces):
    e_term = float(np.mean(np.abs(e_pred - energy) / natoms)) * 1000
    f_term = float(np.mean(np.abs(np.concatenate([(f - t).ravel() for f, t in zip(f_pred, forces)])))) * 1000
    return e_term, f_term, e_term * f_term


def evaluate(model_pt, tag, truth):
    pred = OUT / "cache" / f"{tag}.xyz"
    if not pred.exists():
        opts = OUT / "cache" / f"{tag}.yaml"
        opts.write_text(EVAL_OPTIONS.format(xyz=VAL))
        result = subprocess.run([MTT, "eval", str(model_pt), str(opts), "-o", pred.name, "-b", "16", "--no-warm-up"], cwd=OUT / "cache", capture_output=True, text=True)
        (OUT / "cache" / f"{tag}.log").write_text(result.stdout + result.stderr)
        if result.returncode != 0 or not pred.exists():
            raise RuntimeError(f"mtt eval failed for {tag}")
    natoms, energy, forces, cids, _ = truth
    e_pred, f_pred = load_predictions(pred, cids, truth)
    values = {}
    for name, sl in SUBSETS.items():
        values[name] = native(e_pred[sl], f_pred[sl], natoms[sl], energy[sl], forces[sl])
    return values


def main():
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    truth = read_truth(VAL)
    prov = json.load(open(RUNS / "PROVENANCE.json"))["runs"]
    rows = []
    for run_dir in sorted(p.parent for p in RUNS.glob("scratchv3-s--*/options.yaml")):
        text = (run_dir / "train.log").read_text(errors="replace") if (run_dir / "train.log").exists() else ""
        if "Training finished!" not in text:
            continue
        steps = prov[run_dir.name]["steps_per_epoch"]
        logged = {int(m.group(1)): (float(m.group(2)), float(m.group(3))) for m in LINE.finditer(text)}
        best_epoch = int(BEST.findall(text)[-1])
        snapshots = {int(p.stem.split("_")[1]): p for p in run_dir.glob("outputs/*/*/model_*.ckpt")}
        stride = max(5, int(np.ceil(len(snapshots) / MAX_POINTS)) * 5)
        chosen = sorted(e for e in snapshots if e % stride == 0)
        jobs = [("selected", best_epoch, run_dir / "model_best.pt"), ("final", max(logged), run_dir / "model_final.pt")] + [("snapshot", e, snapshots[e]) for e in chosen]
        for kind, epoch, path in jobs:
            if not path.exists():
                log(f"{run_dir.name}: {kind} model {path.name} missing, skipped")
                continue
            if path.suffix == ".ckpt":
                exported = OUT / "cache" / f"{run_dir.name}__snapshot_{epoch}.pt"
                if not exported.exists():
                    model_from_checkpoint(torch.load(path, map_location="cpu", weights_only=False), context="restart").export().save(str(exported))
                path = exported
            values = evaluate(path, f"{run_dir.name}__{kind}_{epoch}", truth)
            row = {"run": run_dir.name, "kind": kind, "epoch": epoch, "optimizer_updates": (epoch + 1) * steps}
            for name, (e, f, p) in values.items():
                row.update({f"{name}_E_mae_per_atom": round(e, 3), f"{name}_F_mae": round(f, 3), f"{name}_mae_prod": round(p, 3)})
            row["logged_E_mae_per_atom"], row["logged_F_mae"] = logged.get(epoch, (float("nan"), float("nan")))
            rows.append(row)
            log(f"{run_dir.name} {kind} epoch {epoch}: combined E {values['combined'][0]:.2f} F {values['combined'][1]:.1f} | random E {values['random_1000'][0]:.2f} F {values['random_1000'][1]:.1f} | madcore E {values['madcore_250'][0]:.2f} F {values['madcore_250'][1]:.1f}")
    if not rows:
        log("no finished run")
        return
    with open(OUT / "metrics_val_subsets.tsv", "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in COLUMNS) + "\n")
    figure(rows)
    log("SUBSETS DONE")


def figure(rows):
    runs = sorted({r["run"] for r in rows})
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 2.6 * len(runs)), squeeze=False)
    for i, run in enumerate(runs):
        snaps = sorted((r for r in rows if r["run"] == run and r["kind"] == "snapshot"), key=lambda r: r["epoch"])
        selected = [r for r in rows if r["run"] == run and r["kind"] == "selected"]
        for j, metric in enumerate(("E_mae_per_atom", "F_mae")):
            ax = axes[i, j]
            for name in SUBSETS:
                ax.plot([r["optimizer_updates"] for r in snaps], [r[f"{name}_{metric}"] for r in snaps], "-", lw=2, color=COLOR[name], label=name.replace("_", " "))
                if selected:
                    ax.plot([selected[0]["optimizer_updates"]], [selected[0][f"{name}_{metric}"]], "o", ms=8, mfc="white", mec=COLOR[name], mew=2)
            ax.set_yscale("log")
            ax.set_ylabel("E MAE per atom (meV)" if metric == "E_mae_per_atom" else "F MAE, component (meV/A)")
            ax.grid(alpha=0.25)
            ax.set_title(run.split("--")[1].replace("madcore", "MAD-CORE").replace("_", " "), fontsize=10, loc="left")
            if i == len(runs) - 1:
                ax.set_xlabel("optimizer updates")
    axes[0, 0].legend(fontsize=8, title="validation view (open circle = selected epoch)", title_fontsize=8)
    fig.suptitle("validation metrics recomputed on the combined file and its two subsets, inference only, snapshots every stride; selection used the combined values", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "validation_subsets.png", dpi=150)
    fig.savefig(OUT / "validation_subsets.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
