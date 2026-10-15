import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/scripts")
from common_bd import XYZ, load_predictions, read_truth  # noqa: E402
from check_eval_provenance import exported_tensors, sha256, state_match  # noqa: E402

D = Path(__file__).resolve().parent.parent
E = D.parent
S3 = E / "scratch_scaling_v3"
PBE20K = E / "voronoi_weighted" / "pbe_20k"
OUT = D / "reports" / "comparison"
POPULATIONS = S3 / "reports" / "populations"
MANIFEST = POPULATIONS / "manifests" / "balanced_4000.tsv"
EARLIER = POPULATIONS / "finetune_vs_scratch" / "metrics.tsv"
TESTS = ("test_random", "test_madcore")
SUBSETS = {"balanced_4000": TESTS, "random_2000": ("test_random",), "madcore_2000": ("test_madcore",)}
METRICS = ("mae", "rmse", "p8", "sup")
SCRATCH, PREVIOUS, ALIGNED = "scratchv3-s--madcore_20k--s42", "omat-s--madcore_20k--s42", "omat-s--madcore_20k_aligned--s42"
MODELS = {
    "scratch": {"label": "PET-S, trained from scratch", "run": S3 / "runs" / SCRATCH,
                "preds": {t: S3 / "reports" / "selected_checkpoints" / "preds" / f"{SCRATCH}@{t}.xyz" for t in TESTS}},
    "finetune_previous": {"label": "PET-OMat-S, fine-tuned, previous", "run": PBE20K / "runs" / "pbe" / "corrected_20k" / PREVIOUS,
                          "preds": {t: PBE20K / "reports" / "corrected" / "preds" / f"{PREVIOUS}@{t}.xyz" for t in TESTS},
                          "provenance": {t: PBE20K / "reports" / "corrected" / "cache" / f"{PREVIOUS}@{t}.provenance.json" for t in TESTS}},
    "finetune_aligned": {"label": "PET-OMat-S, fine-tuned, aligned", "run": D / "runs" / ALIGNED,
                         "preds": {t: D / "reports" / "preds" / f"{ALIGNED}@{t}.xyz" for t in TESTS},
                         "provenance": {t: D / "reports" / "preds" / f"{ALIGNED}@{t}.provenance.json" for t in TESTS}},
}


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def stats(values):
    a = np.abs(np.asarray(values, dtype=float))
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())), "p8": float(((a ** 8).mean()) ** 0.125), "sup": float(a.max())}


# checkpoint, exported model and predictions of one model, checked against the recorded provenance
def identity(name, model, checks):
    run = model["run"]
    ckpt = torch.load(run / "model.ckpt", map_location="cpu", weights_only=False)
    record = {"run": str(run), "checkpoint_sha256": sha256(run / "model.ckpt"), "exported_sha256": sha256(run / "model_best.pt"),
              "epochs": ckpt["epoch"] + 1, "best_epoch_index": ckpt["best_epoch"], "predictions": {t: sha256(p) for t, p in model["preds"].items()}}
    n, equal = state_match(exported_tensors(run / "model_best.pt"), ckpt["best_model_state_dict"])
    record["exported_equals_selected_state"] = f"{equal}/{n}"
    checks.append((f"{name}: exported model carries the selected checkpoint state ({equal}/{n})", n > 0 and equal == n))
    if name == "scratch":
        for r in read_tsv(S3 / "reports" / "eval_provenance.tsv"):
            if r["checkpoint"] == "selected" and r["run"] == SCRATCH:
                checks.append((f"scratch {r['test']}: checkpoint and exported model match eval_provenance.tsv",
                               r["checkpoint_sha256"] == record["checkpoint_sha256"] and r["exported_sha256"] == record["exported_sha256"]))
    else:
        for t in TESTS:
            p = json.loads(model["provenance"][t].read_text())
            checks.append((f"{name} {t}: predictions, exported model and test file match the provenance record",
                           p["predictions_sha256"] == record["predictions"][t] and p["model_best_pt_sha256"] == record["exported_sha256"]
                           and p["test_xyz_sha256"] == sha256(XYZ / f"{t}.xyz")))
    steps = read_tsv(run / "grad_clip_epochs.tsv")
    record["steps_per_epoch"] = int(steps[-1]["steps"])
    record["updates"] = sum(int(r["steps"]) for r in steps)
    return record


def latex(rows, by_subset):
    lead = "l l " if by_subset else "l "
    first = "model & test & " if by_subset else "model & "
    blank = " & & " if by_subset else " & "
    offset = 2 if by_subset else 1
    text = (f"\\begin{{tabular}}{{{lead}rrrr rrrr}}\n\\toprule\n{blank}\\multicolumn{{4}}{{c}}{{energy (meV/atom)}} & \\multicolumn{{4}}{{c}}{{force (meV/\\AA{{}})}} \\\\\n"
            f"\\cmidrule(lr){{{offset + 1}-{offset + 4}}} \\cmidrule(lr){{{offset + 5}-{offset + 8}}}\n{first}MAE & RMSE & $p_8$ & max & MAE & RMSE & $p_8$ & max \\\\\n\\midrule\n")
    previous = None
    for r in rows:
        if by_subset and previous is not None and r["population"] != previous:
            text += "\\midrule\n"
        previous = r["population"]
        test = {"random_2000": "random (2,000)", "madcore_2000": "MAD-CORE (2,000)"}.get(r["population"], "")
        cells = (f"{r['E_mae']:.1f} & {r['E_rmse']:.1f} & {r['E_p8']:.0f} & {r['E_sup']:.0f} & "
                 f"{r['F_mae']:.0f} & {r['F_rmse']:.0f} & {r['F_p8']:.0f} & {r['F_sup']:.0f}")
        text += f"{r['label']} & {test + ' & ' if by_subset else ''}{cells} \\\\\n"
    return text + "\\bottomrule\n\\end{tabular}\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    frozen = json.loads((POPULATIONS / "POPULATIONS.json").read_text())["manifests"]["balanced_4000"]["sha256"]
    checks.append(("balanced_4000 manifest matches its frozen checksum", sha256(MANIFEST) == frozen))
    manifest = read_tsv(MANIFEST)
    records = {name: identity(name, model, checks) for name, model in MODELS.items()}

    # per structure errors, every prediction aligned to its test file by atom count and positions
    truths = {t: read_truth(XYZ / f"{t}.xyz") for t in TESTS}
    selection = {}
    for t in TESTS:
        index = {c: i for i, c in enumerate(truths[t][3])}
        rows = [r for r in manifest if r["source_test"] == t]
        selection[t] = np.array([index[r["configuration_id"]] for r in rows])
        checks.append((f"{t}: all {len(rows)} population rows found with matching atom counts",
                       all(int(r["natoms"]) == int(truths[t][0][index[r["configuration_id"]]]) for r in rows)))
    results = []
    for name, model in MODELS.items():
        errors = {}
        for t in TESTS:
            natoms, energy, forces, cids, _ = truths[t]
            e_pred, f_pred = load_predictions(model["preds"][t], cids, truths[t])
            errors[t] = {"de": (e_pred - energy) / natoms * 1000, "df": [(f - r) * 1000 for f, r in zip(f_pred, forces)], "natoms": natoms}
        checks.append((f"{name}: predictions aligned with both test files", True))
        for subset, tests in SUBSETS.items():
            de = np.concatenate([errors[t]["de"][selection[t]] for t in tests])
            df = np.concatenate([errors[t]["df"][i].ravel() for t in tests for i in selection[t]])
            e, f = stats(de), stats(df)
            results.append({"model": name, "label": model["label"], "population": subset, "structures": len(de),
                            "atoms": int(sum(errors[t]["natoms"][selection[t]].sum() for t in tests)), "force_components": df.size,
                            **{f"E_{k}": e[k] for k in METRICS}, **{f"F_{k}": f[k] for k in METRICS}})

    # the two existing models must reproduce the comparison saved earlier
    earlier = {(r["model"], r["population"]): r for r in read_tsv(EARLIER)}
    gap = 0.0
    for r in results:
        old = earlier.get(({"finetune_previous": "finetune"}.get(r["model"], r["model"]), r["population"]))
        if old is not None:
            gap = max(gap, *(abs(r[f"E_{k}"] - float(old[f"E_{k}_meV_per_atom"])) for k in METRICS),
                      *(abs(r[f"F_{k}"] - 1000 * float(old[f"F_{k}_eV_per_A"])) for k in METRICS))
    checks.append((f"scratch and previous fine-tune reproduce finetune_vs_scratch/metrics.tsv (largest gap {gap:.1e})", gap < 1e-2))

    columns = ["model", "label", "population", "structures", "atoms", "force_components"] + [f"E_{k}" for k in METRICS] + [f"F_{k}" for k in METRICS]
    with open(OUT / "metrics.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\t# E in meV/atom, F in meV/A\n")
        for r in results:
            fh.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in columns) + "\n")
    order = list(MODELS)
    pooled = [r for r in results if r["population"] == "balanced_4000"]
    parts = sorted([r for r in results if r["population"] != "balanced_4000"], key=lambda r: (r["population"] != "random_2000", order.index(r["model"])))
    (OUT / "table_pooled.tex").write_text(latex(pooled, False))
    (OUT / "table_by_subset.tex").write_text(latex(parts, True))
    passed = all(ok for _, ok in checks)
    (OUT / "provenance.json").write_text(json.dumps({"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "models": records,
                                                      "checks": [{"check": c, "passed": ok} for c, ok in checks]}, indent=1) + "\n")
    for r in results:
        print(f"{r['model']:18s} {r['population']:13s} E {r['E_mae']:.1f} / {r['E_rmse']:.1f} / {r['E_p8']:.0f} / {r['E_sup']:.0f} meV/atom"
              f" | F {r['F_mae']:.0f} / {r['F_rmse']:.0f} / {r['F_p8']:.0f} / {r['F_sup']:.0f} meV/A")
    for name, rec in records.items():
        print(name, {k: rec[k] for k in ("epochs", "best_epoch_index", "steps_per_epoch", "updates", "exported_equals_selected_state")})
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    print("COMPARISON", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
