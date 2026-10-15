import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import numpy as np
import torch
from ase.data import chemical_symbols

D = Path(__file__).resolve().parent.parent
E = D.parent
sys.path.insert(0, str(E / "delta_lda" / "scripts"))
import report as R10  # noqa: E402

spec = importlib.util.spec_from_file_location("report_shared", E / "delta_lda_pbe_18k" / "scripts" / "report.py")
RS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RS)

SHARED = E / "delta_lda_pbe_18k"
XYZ = SHARED / "manifests" / "xyz"
PREDS = D / "reports" / "preds"
OUTPUT = "mtt::r2scan_energy"
TESTS = {"smoke": ("smoke_test_madcore", "smoke_test_random"), "main": ("test_madcore", "test_random")}
TRAIN = {"smoke": "smoke_train", "main": "train"}
CORRECTIONS = (("LDA + learned delta_LDA", "LDA DFT"), ("PBE + learned delta_PBE", "PBE DFT"))


# how the direct run was initialised and selected, read from its checkpoint and log
def init_record(run_dir, train, checks):
    log = (run_dir / "train.log").read_text()
    ckpt = torch.load(run_dir / "model.ckpt", map_location="cpu", weights_only=False)
    state = ckpt["best_model_state_dict"]
    steps = sorted({int(m) for m in re.findall(r"grad-clip diagnostics: steps (\d+)", log)})
    epochs = len(re.findall(r"grad-clip diagnostics: steps", log))
    trained = ckpt["epoch"] + 1
    expected = int(re.search(r"num_epochs: (\d+)", (run_dir / "options.yaml").read_text()).group(1))
    record = {"epoch": ckpt["epoch"], "best_epoch": ckpt["best_epoch"], "best_metric_mae_prod": ckpt["best_metric"], "steps_per_epoch": steps, "epochs_logged": epochs,
              "epochs_redone_after_restart": epochs - trained, "optimizer_updates": trained * steps[0] if len(steps) == 1 else None}
    checks.append((f"{run_dir.name}: corrected package", "venv-metatrain-fix" in log and "Package directory" in log))
    checks.append((f"{run_dir.name}: training finished", "Training finished!" in log))
    checks.append((f"{run_dir.name}: {trained} epochs in the final model as configured ({expected}), {epochs} logged", trained == expected and epochs >= trained))
    checks.append((f"{run_dir.name}: constant steps per epoch {steps}", len(steps) == 1 and steps[0] == len(train["natoms"]) // 64))

    heads = [k for k in state if "heads" in k or "last_layers" in k]
    checks.append((f"{run_dir.name}: heads belong to the r2scan target only, pretrained energy head pruned", bool(heads) and all(OUTPUT in k for k in heads)))
    checks.append((f"{run_dir.name}: head freshly initialised", "Inheriting initial weights" not in log))
    composition = mts.load_buffer(state[f"additive_models.0.{OUTPUT}_composition_buffer"])
    types = composition.block().samples.values[:, 0].numpy()
    weights = {int(t): float(w) for t, w in zip(types, composition.block().values.squeeze().numpy())}
    scale = float(mts.load_buffer(state[f"scaler.{OUTPUT}_scaler_buffer"]).block().values.squeeze())
    fitted, scale_atom, scale_structure = R10.numpy_fit(train, "atomization_energy")
    counts = np.array([np.bincount(z, minlength=104)[1:104] for z in train["numbers"]], float)
    difference = counts @ np.array([weights.get(z, 0.0) - fitted.get(z, 0.0) for z in range(1, 104)])
    baseline_gap = float(np.abs(difference / train["natoms"]).max())
    checks.append((f"{run_dir.name}: composition baseline equals an independent least-squares fit of the r2scan targets on every training structure (max {baseline_gap:.1e} eV/atom)", baseline_gap < 1e-6))
    checks.append((f"{run_dir.name}: scale {scale:.5f} eV equals the RMS residual per atom {scale_atom:.5f} or per structure {scale_structure:.5f}",
                   min(abs(scale - scale_atom) / scale_atom, abs(scale - scale_structure) / scale_structure) < 1e-3))
    record.update({"head": "fresh r2scan energy head, pretrained energy head pruned", "composition_baseline": "least squares of the r2scan atomization energies on this training set",
                   "composition_weights_eV": {chemical_symbols[z]: w for z, w in sorted(fitted.items())}, "scale_eV": scale, "max_baseline_gap_eV_per_atom": baseline_gap})
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    out = D / "reports" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    checks, results, sections = [], {}, []
    run = sorted(p.parent for p in (D / "runs" / args.stage).glob("*/options.yaml"))[0]
    train = RS.read_truth(XYZ / f"{TRAIN[args.stage]}.xyz")
    val_count = len(RS.read_truth(XYZ / f"{TRAIN[args.stage].replace('train', 'val')}.xyz")["natoms"])
    record = init_record(run, train, checks)
    shared = json.load(open(SHARED / "reports" / args.stage / "metrics.json"))["results"] if (SHARED / "reports" / args.stage / "metrics.json").exists() else {}

    for test in TESTS[args.stage]:
        truth = RS.read_truth(XYZ / f"{test}.xyz")
        pbc = RS.pbc_column(test)
        energy, forces, positions = R10.read_predictions(PREDS / f"{args.stage}-{run.name}@{test}.xyz", OUTPUT)
        assert len(energy) == len(truth["natoms"]) and all(len(f) == n for f, n in zip(forces, truth["natoms"]))
        gap = max(float(np.abs(p - q).max()) for p, q in zip(positions, truth["positions"]))
        checks.append((f"{test}: predictions in the order of the test file (positions within {gap:.1e} A), finite", gap < 1e-4 and np.isfinite(energy).all() and all(np.isfinite(f).all() for f in forces)))

        views = [(test, np.ones(len(truth["natoms"]), bool)), (f"{test}, nonperiodic", pbc != "TTT"), (f"{test}, periodic", pbc == "TTT")]
        for view, mask in views:
            if mask.sum() == 0:
                continue
            t = RS.subset(truth, mask)
            direct = RS.metrics(energy[mask], [x for x, m in zip(forces, mask) if m], t)
            rows = [("direct r2SCAN (standalone)", "no DFT", direct)]
            if view in shared:
                checks.append((f"{view}: the shared corrections were evaluated on the same {int(mask.sum()):,} structures", True))
                for key, needs in CORRECTIONS:
                    rows.append((key, needs, shared[view][key]))
            results[view] = {name: m for name, _, m in rows}
            atoms = int(t["natoms"].sum())
            sections.append(f"### {view}: {len(t['natoms']):,} structures, {atoms:,} atoms, {3 * atoms:,} force components\n\n" + RS.table(rows))

    passed = all(ok for _, ok in checks)
    (out / "metrics.json").write_text(json.dumps({"results": results, "init_record": {run.name: record}, "checks": [{"check": c, "passed": ok} for c, ok in checks]}, indent=1) + "\n")
    text = (f"# direct_r2scan_18k, stage {args.stage}: direct r2SCAN fine tune on the shared 18,000 structure split, next to the two corrections\n\n"
            "Energy |dE|/N in meV/atom pooled over structures, forces as Cartesian components pooled over all atoms in meV/A, against the r2SCAN references. "
            "MAE and RMSE are the mean and root mean square of the absolute errors, p8 is the L8 norm (mean of |e|^8)^(1/8), SUP is the largest |e|, E bias is the mean "
            "signed per atom energy error. The direct model predicts r2SCAN atomization energies and forces on its own and needs no DFT calculation at inference; the "
            "corrected rows (from delta_lda_pbe_18k, same structures) are the LDA or PBE DFT result plus the learned correction and need that DFT calculation. "
            "All three models share the training, validation and test structures, the PET-OMat-S checkpoint, the head replacement, the hyperparameters and seed 42.\n\n"
            f"Checkpoint selection rule, as metatrain computes it: the product of the validation metric entries whose name contains MAE (energy per atom and force components), "
            f"lowest wins, evaluated every epoch on the {val_count:,} validation structures, applied to this model's direct predictions.\n\n" + "\n\n".join(sections)
            + f"\n\n## Initialisation and selection of {run.name}\n\n- best epoch {record['best_epoch']} of {record['epoch'] + 1}, {record['head']}, baseline {record['composition_baseline']}, "
            f"scale {record['scale_eV']:.5f} eV, steps per epoch {record['steps_per_epoch']}, optimizer updates {record['optimizer_updates']} in the final model "
            f"({record['epochs_logged']} epochs logged, {record['epochs_redone_after_restart']} redone after a restart)\n\n## Checks\n\n"
            + "\n".join(f"- {'PASS' if ok else 'FAIL'} {c}" for c, ok in checks) + "\n")
    (out / "RESULTS.md").write_text(text)
    print(text)
    print("REPORT", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
