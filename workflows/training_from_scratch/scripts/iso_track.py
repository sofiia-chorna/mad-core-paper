import csv
import json
import re
import time
from pathlib import Path

import torch
import metatomic.torch  # noqa: F401
from metatrain.utils.io import model_from_checkpoint

from isolated_atoms import ELEMENTS, isolated_systems, predict_isolated, summary

# training diagnostic (not a selection criterion): predicted energies of the 102 isolated atom references for every
# stored snapshot of every run, appended incrementally to one tsv per run
S = Path(__file__).resolve().parent.parent
RUNS = S / "runs"
OUT = S / "reports" / "isolated_atoms"
COLUMNS = ["run", "epoch", "optimizer_updates", "median_abs_eV", "max_abs_eV", "max_element", "H_eV", "Ne_eV", "n_above_1eV"] + ELEMENTS
BEST = re.compile(r"Using best model from epoch (\d+)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov = json.load(open(RUNS / "PROVENANCE.json"))["runs"]
    for run_dir in sorted(p.parent for p in RUNS.glob("scratchv3-s--*/options.yaml")):
        steps = prov[run_dir.name]["steps_per_epoch"]
        path = OUT / f"{run_dir.name}.tsv"
        done = {(-1 if str(r["epoch"]).startswith("final") else int(r["epoch"])) for r in csv.DictReader(open(path), delimiter="\t")} if path.exists() else set()
        snapshots = {int(p.stem.split("_")[1]): p for p in run_dir.glob("outputs/*/*/model_*.ckpt")}
        text = (run_dir / "train.log").read_text(errors="replace") if (run_dir / "train.log").exists() else ""
        if "Training finished!" in text and (run_dir / "model.ckpt").exists():
            snapshots[-1] = run_dir / "model.ckpt"
        new = sorted(e for e in snapshots if e not in done)
        if not new:
            continue
        systems = None
        with open(path, "a") as fh:
            if not done:
                fh.write("\t".join(COLUMNS) + "\n")
            for epoch in new:
                try:
                    ckpt = torch.load(snapshots[epoch], map_location="cpu", weights_only=False)
                    model = model_from_checkpoint(ckpt, context="restart").to(dtype=torch.float32).eval()
                except Exception as error:
                    print(f"{run_dir.name} epoch {epoch}: could not load ({error!r}), skipped", flush=True)
                    continue
                if systems is None:
                    systems = isolated_systems(model)
                pred = predict_isolated(model, systems)
                s = summary(pred)
                label = epoch if epoch >= 0 else int(ckpt.get("epoch", -1))
                row = {"run": run_dir.name, "epoch": label if epoch >= 0 else f"final:{label}", "optimizer_updates": (label + 1) * steps, **s, **{e: f"{pred[e]:.4f}" for e in ELEMENTS}}
                fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
                fh.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {run_dir.name} epoch {row['epoch']}: median {s['median_abs_eV']:.3f} max {s['max_abs_eV']:.3f} ({s['max_element']}) H {s['H_eV']:.3f} Ne {s['Ne_eV']:.3f}", flush=True)
    print("ISO TRACK DONE")


if __name__ == "__main__":
    main()
