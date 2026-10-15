import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/audit/batchdep")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_bd import XYZ, load_predictions, read_truth  # noqa: E402
from check_eval_provenance import exported_tensors, sha256, state_match  # noqa: E402

S = Path(__file__).resolve().parent.parent
E = S.parent
POPULATIONS = S / "reports" / "populations"
OUT = POPULATIONS / "finetune_vs_scratch"
MANIFEST = POPULATIONS / "manifests" / "balanced_4000.tsv"
TESTS = ("test_random", "test_madcore")
SUBSETS = {"balanced_4000": TESTS, "random_2000": ("test_random",), "madcore_2000": ("test_madcore",)}
SCRATCH = "scratchv3-s--madcore_20k--s42"
FINETUNE = "omat-s--madcore_20k--s42"
PBE20K = E / "voronoi_weighted" / "pbe_20k"
MODELS = {
    "scratch": {
        "label": "PET-S from scratch",
        "run": S / "runs" / SCRATCH,
        "preds": {t: S / "reports" / "selected_checkpoints" / "preds" / f"{SCRATCH}@{t}.xyz" for t in TESTS},
    },
    "finetune": {
        "label": "PET-OMat-S fine-tune",
        "run": PBE20K / "runs" / "pbe" / "corrected_20k" / FINETUNE,
        "preds": {t: PBE20K / "reports" / "corrected" / "preds" / f"{FINETUNE}@{t}.xyz" for t in TESTS},
    },
}
METRICS = ("mae", "rmse", "p8", "sup")


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def stats(values):
    a = np.abs(np.asarray(values, dtype=float))
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())), "p8": float(((a ** 8).mean()) ** 0.125), "sup": float(a.max())}


# the recorded hashes of each model's checkpoint, exported model, test files and predictions, checked against the files
def check_identity(name, model, checks):
    run = model["run"]
    ckpt = torch.load(run / "model.ckpt", map_location="cpu", weights_only=False)
    exported = run / "model_best.pt"
    record = {"checkpoint": str(run / "model.ckpt"), "checkpoint_sha256": sha256(run / "model.ckpt"), "exported": str(exported),
              "exported_sha256": sha256(exported), "epochs": ckpt["epoch"] + 1, "best_epoch_index": ckpt["best_epoch"], "predictions": {}}
    n, equal = state_match(exported_tensors(exported), ckpt["best_model_state_dict"])
    record["exported_equals_selected_state"] = f"{equal}/{n}"
    checks.append((f"{name}: exported model carries the selected checkpoint state ({equal}/{n} tensors)", n > 0 and equal == n))

    if name == "scratch":
        rows = [r for r in read_tsv(S / "reports" / "eval_provenance.tsv") if r["checkpoint"] == "selected" and r["run"] == SCRATCH]
        for r in rows:
            checks.append((f"scratch {r['test']}: checkpoint, exported model and test file match eval_provenance.tsv",
                           r["checkpoint_sha256"] == record["checkpoint_sha256"] and r["exported_sha256"] == record["exported_sha256"]
                           and r["test_sha256"] == sha256(XYZ / f"{r['test']}.xyz") and Path(r["predictions"]) == model["preds"][r["test"]]))
    else:
        for t in TESTS:
            p = json.loads((PBE20K / "reports" / "corrected" / "cache" / f"{FINETUNE}@{t}.provenance.json").read_text())
            checks.append((f"finetune {t}: predictions, exported model and test file match the provenance record",
                           p["predictions_sha256"] == sha256(model["preds"][t]) and p["model_best_pt_sha256"] == record["exported_sha256"]
                           and p["test_xyz_sha256"] == sha256(XYZ / f"{t}.xyz")))
    for t in TESTS:
        record["predictions"][t] = {"path": str(model["preds"][t]), "sha256": sha256(model["preds"][t])}
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    # the frozen population, checked against its recorded checksum
    frozen = json.loads((POPULATIONS / "POPULATIONS.json").read_text())["manifests"]["balanced_4000"]["sha256"]
    checks.append(("balanced_4000 manifest matches its frozen checksum", sha256(MANIFEST) == frozen))
    manifest = read_tsv(MANIFEST)

    identities = {name: check_identity(name, model, checks) for name, model in MODELS.items()}

    # per structure errors of both models, with every prediction aligned to its test file by atom count and positions
    truths = {t: read_truth(XYZ / f"{t}.xyz") for t in TESTS}
    errors = {}
    for name, model in MODELS.items():
        errors[name] = {}
        for t in TESTS:
            natoms, energy, forces, cids, _ = truths[t]
            e_pred, f_pred = load_predictions(model["preds"][t], cids, truths[t])
            errors[name][t] = {"cids": cids, "natoms": natoms, "de": (e_pred - energy) / natoms * 1000, "df": [(f - r) * 1000 for f, r in zip(f_pred, forces)]}
        checks.append((f"{name}: predictions aligned with both test files (atom counts and positions within 1e-4 A)", True))

    # rows of the frozen population, matched by configuration id with the atom count checked
    selection = {}
    for t in TESTS:
        index = {c: i for i, c in enumerate(truths[t][3])}
        rows = [r for r in manifest if r["source_test"] == t]
        selection[t] = np.array([index[r["configuration_id"]] for r in rows])
        checks.append((f"{t}: {len(rows)} population rows found in the test file with matching atom counts",
                       all(int(r["natoms"]) == int(truths[t][0][index[r["configuration_id"]]]) for r in rows)))

    # metrics, energy per atom over structures in meV/atom, forces over all cartesian components in eV/A
    results = []
    for name in MODELS:
        for subset, tests in SUBSETS.items():
            de = np.concatenate([errors[name][t]["de"][selection[t]] for t in tests])
            df = np.concatenate([errors[name][t]["df"][i].ravel() for t in tests for i in selection[t]])
            atoms = int(sum(errors[name][t]["natoms"][selection[t]].sum() for t in tests))
            e, f = stats(de), stats(df / 1000)
            results.append({"model": name, "label": MODELS[name]["label"], "population": subset, "structures": len(de), "atoms": atoms, "force_components": df.size,
                            **{f"E_{k}_meV_per_atom": e[k] for k in METRICS}, **{f"F_{k}_eV_per_A": f[k] for k in METRICS}})

    # cross checks against the numbers already reported
    by = {(r["model"], r["population"]): r for r in results}
    old = {r["model"]: r for r in read_tsv(POPULATIONS / "balanced_4000" / "tables" / "metrics_balanced_4000.tsv")}[SCRATCH]
    gap = max(abs(by[("scratch", "balanced_4000")][f"{q}_{k}_{u}"] - float(old[f"{q}_{k}_{u}"])) for q, u in (("E", "meV_per_atom"), ("F", "eV_per_A")) for k in METRICS)
    checks.append((f"scratch on balanced_4000 reproduces metrics_balanced_4000.tsv (largest gap {gap:.1e})", gap < 1e-5))
    full = {(r["test"], r["model"]): r for r in read_tsv(S / "reports" / "selected_checkpoints" / "metrics_mtt.tsv")}
    corrected = {(r["test"], r["model"]): r for r in read_tsv(PBE20K / "reports" / "corrected" / "metrics.tsv")}
    for t in TESTS:
        scratch_full = stats(errors["scratch"][t]["de"])["mae"]
        finetune_full = stats(errors["finetune"][t]["de"])["mae"]
        checks.append((f"scratch full {t} energy MAE {scratch_full:.3f} reproduces metrics_mtt.tsv", abs(scratch_full - float(full[(t, SCRATCH)]["E_mae"])) < 2e-3))
        checks.append((f"finetune full {t} energy MAE {finetune_full:.3f} reproduces the corrected rerun table", abs(finetune_full - float(corrected[(t, FINETUNE)]["E_mae"])) < 2e-3))

    # the two training sets
    scratch_ids = {r["configuration_id"] for r in read_tsv(S / "manifests" / "madcore_20k_ordinary.tsv")}
    finetune_ids = {r["configuration_id"] for r in read_tsv(PBE20K / "manifests" / "madcore_20k.tsv")}
    val250 = {r["configuration_id"] for r in read_tsv(E / "scratch_scaling_v2" / "manifests" / "val_madcore_250.tsv")}
    training = {"scratch_ordinary": len(scratch_ids), "finetune": len(finetune_ids), "shared": len(scratch_ids & finetune_ids),
                "finetune_only_in_scratch_validation": len((finetune_ids - scratch_ids) & val250), "scratch_isolated_atom_references": 102}

    columns = list(results[0].keys())
    with open(OUT / "metrics.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in results:
            fh.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in columns) + "\n")
    passed = all(ok for _, ok in checks)
    (OUT / "provenance.json").write_text(json.dumps({"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "models": identities,
                                                      "training_sets": training, "checks": [{"check": c, "passed": ok} for c, ok in checks]}, indent=1) + "\n")
    for r in results:
        print(f"{r['model']:8s} {r['population']:13s} n={r['structures']:4d} E {r['E_mae_meV_per_atom']:.1f} / {r['E_rmse_meV_per_atom']:.1f} / {r['E_p8_meV_per_atom']:.1f} / {r['E_sup_meV_per_atom']:.1f} meV/atom"
              f" | F {r['F_mae_eV_per_A']:.3f} / {r['F_rmse_eV_per_A']:.3f} / {r['F_p8_eV_per_A']:.2f} / {r['F_sup_eV_per_A']:.2f} eV/A")
    print(json.dumps(training))
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    print("COMPARISON", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
