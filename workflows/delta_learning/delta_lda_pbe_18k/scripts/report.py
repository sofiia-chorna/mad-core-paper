import argparse
import json
import re
import sys
from pathlib import Path

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import numpy as np
import torch
from ase.data import atomic_numbers, chemical_symbols

D = Path(__file__).resolve().parent.parent
E = D.parent
sys.path.insert(0, str(E / "delta_lda" / "scripts"))
import report as R10  # noqa: E402

XYZ = D / "manifests" / "xyz"
PREDS = D / "reports" / "preds"
TESTS = {"smoke": ("smoke_test_madcore", "smoke_test_random"), "main": ("test_madcore", "test_random")}
TRAIN = {"smoke": "smoke_train", "main": "train"}
SCALARS = ("atomization_energy", "atomization_energy_lda", "atomization_energy_pbe", "delta_energy_lda", "delta_energy_pbe")
ARRAYS = ("forces", "forces_lda", "forces_pbe", "delta_forces_lda", "delta_forces_pbe")
FUNCTIONAL = {"delta_lda": "lda", "delta_pbe": "pbe"}


# the reference file written by build_sets_lda.py
def read_truth(path):
    data = {k: [] for k in (*SCALARS, *ARRAYS, "numbers", "positions", "configuration_id")}
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line.strip():
                break
            n = int(line)
            header = fh.readline()
            rows = [fh.readline().split() for _ in range(n)]
            for key in SCALARS:
                data[key].append(R10.value(header, key))
            data["configuration_id"].append(re.search(r'configuration_id="([^"]+)"', header).group(1))
            data["numbers"].append(np.array([atomic_numbers[r[0]] for r in rows]))
            values = np.array([[float(v) for v in r[1:]] for r in rows])
            data["positions"].append(values[:, 0:3])
            for i, key in enumerate(ARRAYS):
                data[key].append(values[:, 3 + 3 * i:6 + 3 * i])
    for key in SCALARS:
        data[key] = np.array(data[key])
    data["natoms"] = np.array([len(z) for z in data["numbers"]])
    return data


def subset(truth, mask):
    out = {}
    for k, v in truth.items():
        out[k] = v[mask] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(mask) else [x for x, m in zip(v, mask) if m]
    return out


def metrics(e_pred, f_pred, truth):
    de = (e_pred - truth["atomization_energy"]) / truth["natoms"] * 1000
    df = np.concatenate([((f - t) * 1000).ravel() for f, t in zip(f_pred, truth["forces"])])
    return {"E": dict(R10.stats(de), bias=float(de.mean())), "F": R10.stats(df)}


def init_record(run_dir, train, checks, functional):
    log = (run_dir / "train.log").read_text()
    ckpt = torch.load(run_dir / "model.ckpt", map_location="cpu", weights_only=False)
    state = ckpt["best_model_state_dict"]
    steps = sorted({int(m) for m in re.findall(r"grad-clip diagnostics: steps (\d+)", log)})
    epochs = len(re.findall(r"grad-clip diagnostics: steps", log))
    trained = ckpt["epoch"] + 1
    record = {"epoch": ckpt["epoch"], "best_epoch": ckpt["best_epoch"], "best_metric_mae_prod": ckpt["best_metric"],
              "steps_per_epoch": steps, "epochs_logged": epochs, "epochs_redone_after_restart": epochs - trained,
              "optimizer_updates": trained * steps[0] if len(steps) == 1 else None,
              "optimizer_updates_logged_including_redone": epochs * steps[0] if len(steps) == 1 else None}
    checks.append((f"{run_dir.name}: corrected package", "venv-metatrain-fix" in log and "Package directory" in log))
    checks.append((f"{run_dir.name}: training finished", "Training finished!" in log))
    expected = int(re.search(r"num_epochs: (\d+)", (run_dir / "options.yaml").read_text()).group(1))
    checks.append((f"{run_dir.name}: {trained} epochs in the final model as configured ({expected}), {epochs} logged", trained == expected and epochs >= trained))
    checks.append((f"{run_dir.name}: constant steps per epoch {steps}", len(steps) == 1 and steps[0] == len(train["natoms"]) // 64))

    heads = [k for k in state if "heads" in k or "last_layers" in k]
    checks.append((f"{run_dir.name}: heads belong to the delta target only", all("mtt::delta_energy" in k for k in heads)))
    checks.append((f"{run_dir.name}: delta head freshly initialised", "Inheriting initial weights" not in log))
    composition = mts.load_buffer(state["additive_models.0.mtt::delta_energy_composition_buffer"])
    types = composition.block().samples.values[:, 0].numpy()
    weights = {int(t): float(w) for t, w in zip(types, composition.block().values.squeeze().numpy())}
    scale = float(mts.load_buffer(state["scaler.mtt::delta_energy_scaler_buffer"]).block().values.squeeze())
    fitted, scale_atom, scale_structure = R10.numpy_fit(train, f"delta_energy_{functional}")
    counts = np.array([np.bincount(z, minlength=104)[1:104] for z in train["numbers"]], float)
    difference = counts @ np.array([weights.get(z, 0.0) - fitted.get(z, 0.0) for z in range(1, 104)])
    baseline_gap = float(np.abs(difference / train["natoms"]).max())
    checks.append((f"{run_dir.name}: composition baseline equals an independent least-squares fit on every training structure (max {baseline_gap:.1e} eV/atom)", baseline_gap < 1e-6))
    checks.append((f"{run_dir.name}: scale {scale:.5f} eV equals the RMS residual per atom {scale_atom:.5f} or per structure {scale_structure:.5f}",
                   min(abs(scale - scale_atom) / scale_atom, abs(scale - scale_structure) / scale_structure) < 1e-3))
    record.update({"head": "fresh delta head, pretrained energy head pruned", "composition_baseline": "least squares on this training set",
                   "composition_weights_eV": {chemical_symbols[z]: w for z, w in sorted(fitted.items())}, "scale_eV": scale, "max_baseline_gap_eV_per_atom": baseline_gap})
    return record


def table(rows):
    lines = ["| model | inference needs | E MAE / RMSE | E bias | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |", "|---|---|---|---|---|---|---|---|---|"]
    for name, needs, m in rows:
        e, f = m["E"], m["F"]
        lines.append(f"| {name} | {needs} | {e['mae']:.1f} / {e['rmse']:.1f} | {e['bias']:+.1f} | {f['mae']:.0f} / {f['rmse']:.0f} | {e['p8']:.1f} | {f['p8']:.0f} | {e['sup']:.1f} | {f['sup']:.0f} |")
    return "\n".join(lines)


def pbc_column(name):
    rows = [r.split("\t") for r in (D / "manifests" / f"{name}.tsv").read_text().strip().split("\n")]
    return np.array([r[rows[0].index("pbc")] for r in rows[1:]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    out = D / "reports" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    checks, results, sections, records = [], {}, [], {}

    runs = sorted(p.parent for p in (D / "runs" / args.stage).glob("*/options.yaml"))
    assert len(runs) == 2, runs
    train = read_truth(XYZ / f"{TRAIN[args.stage]}.xyz")
    val_count = len(read_truth(XYZ / f"{TRAIN[args.stage].replace('train', 'val')}.xyz")["natoms"])
    for name in (TRAIN[args.stage], *TESTS[args.stage]):
        t = read_truth(XYZ / f"{name}.xyz")
        for functional in ("lda", "pbe"):
            e_gap = float(np.abs(t[f"atomization_energy_{functional}"] + t[f"delta_energy_{functional}"] - t["atomization_energy"]).max())
            f_gap = max(float(np.abs(fb + d - f).max()) for fb, d, f in zip(t[f"forces_{functional}"], t[f"delta_forces_{functional}"], t["forces"]))
            checks.append((f"{name}: {functional} + delta target reproduces r2scan (E {e_gap:.1e} eV, F {f_gap:.1e} eV/A)", e_gap < 1e-8 and f_gap < 1e-8))
    for run in runs:
        records[run.name] = init_record(run, train, checks, FUNCTIONAL[run.name.split("--")[0]])
    # the two runs must share every split, checked on the options files
    same = all(open(runs[0] / "options.yaml").read().count(path) == 1 and open(runs[1] / "options.yaml").read().count(path) == 1
               for path in (str(XYZ / f"{TRAIN[args.stage]}.xyz"), str(XYZ / f"{TRAIN[args.stage].replace('train', 'val')}.xyz")))
    checks.append(("both runs read the same training and validation files", same))

    for test in TESTS[args.stage]:
        truth = read_truth(XYZ / f"{test}.xyz")
        pbc = pbc_column(test)
        preds = {}
        for run in runs:
            functional = FUNCTIONAL[run.name.split("--")[0]]
            energy, forces, positions = R10.read_predictions(PREDS / f"{args.stage}-{run.name}@{test}.xyz", "mtt::delta_energy")
            assert len(energy) == len(truth["natoms"]) and all(len(f) == n for f, n in zip(forces, truth["natoms"])), (run.name, test)
            gap = max(float(np.abs(p - q).max()) for p, q in zip(positions, truth["positions"]))
            checks.append((f"{test} {run.name}: predictions in the order of the test file (positions within {gap:.1e} A), finite",
                           gap < 1e-4 and np.isfinite(energy).all() and all(np.isfinite(f).all() for f in forces)))
            preds[functional] = (run.name, energy, forces)

        # the whole test, then its nonperiodic and periodic parts as energy diagnostics
        views = [(test, np.ones(len(truth["natoms"]), bool)), (f"{test}, nonperiodic", pbc != "TTT"), (f"{test}, periodic", pbc == "TTT")]
        for view, mask in views:
            if mask.sum() == 0:
                continue
            t = subset(truth, mask)
            rows = []
            for functional, label in (("lda", "LDA"), ("pbe", "PBE")):
                rows.append((f"{label}, no correction", f"{label} DFT", metrics(t[f"atomization_energy_{functional}"], t[f"forces_{functional}"], t)))
                name, energy, forces = preds[functional]
                e, f = energy[mask], [x for x, m in zip(forces, mask) if m]
                reconstructed = metrics(t[f"atomization_energy_{functional}"] + e, [fb + x for fb, x in zip(t[f"forces_{functional}"], f)], t)
                target_e = (e - t[f"delta_energy_{functional}"]) / t["natoms"] * 1000
                target_f = np.concatenate([((x - d) * 1000).ravel() for x, d in zip(f, t[f"delta_forces_{functional}"])])
                identity = max(abs(reconstructed["E"]["mae"] - R10.stats(target_e)["mae"]), abs(reconstructed["F"]["mae"] - R10.stats(target_f)["mae"]))
                checks.append((f"{view} {name}: reconstructed r2SCAN error equals the delta target error (gap {identity:.1e})", identity < 1e-6))
                rows.append((f"{label} + learned delta_{label}", f"{label} DFT", reconstructed))
            results[view] = {name: m for name, _, m in rows}
            atoms = int(t["natoms"].sum())
            sections.append(f"### {view}: {len(t['natoms']):,} structures, {atoms:,} atoms, {3 * atoms:,} force components\n\n" + table(rows))

    passed = all(ok for _, ok in checks)
    (out / "metrics.json").write_text(json.dumps({"results": results, "init_records": records, "checks": [{"check": c, "passed": ok} for c, ok in checks]}, indent=1) + "\n")
    init = "\n".join(f"- {name}: best epoch {r['best_epoch']} of {r['epoch'] + 1}, {r['head']}, baseline {r['composition_baseline']}, scale {r['scale_eV']:.5f} eV, "
                      f"steps per epoch {r['steps_per_epoch']}, optimizer updates {r['optimizer_updates']} in the final model ({r['epochs_logged']} epochs logged, "
                      f"{r['epochs_redone_after_restart']} redone after a restart)" for name, r in records.items())
    text = (f"# delta_lda_pbe_18k, stage {args.stage}: r2SCAN-LDA and r2SCAN-PBE corrections on one shared split\n\n"
            "Energy |dE|/N in meV/atom pooled over structures, forces as Cartesian components pooled over all atoms in meV/A, against the r2SCAN references. "
            "MAE and RMSE are the mean and root mean square of the absolute errors, p8 is the L8 norm (mean of |e|^8)^(1/8), SUP is the largest |e|, "
            "E bias is the mean signed per atom energy error (prediction minus reference). Corrected rows are reconstructed as the DFT energy and forces of "
            "the respective functional plus the learned correction, so each corrected prediction requires that functional's DFT calculation at inference. "
            "Both corrections are evaluated on identical structures. No test offset is subtracted and no checkpoint is chosen on test data.\n\n"
            f"Checkpoint selection rule, as metatrain computes it: at every epoch the validation metric entries whose name contains MAE (energy per atom and force "
            f"components) are multiplied, the epoch with the lowest product is kept (best_model_metric mae_prod), applied separately to each model on the same "
            f"{val_count:,} validation structures.\n\n" + "\n\n".join(sections)
            + "\n\n## Initialisation and selection\n\n" + init + "\n\n## Checks\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'} {c}" for c, ok in checks) + "\n")
    (out / "RESULTS.md").write_text(text)
    print(text)
    print("REPORT", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
