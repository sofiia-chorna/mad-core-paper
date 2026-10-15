import json
import re
import sys
from pathlib import Path

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import numpy as np
import torch

D = Path(__file__).resolve().parent.parent
OUT = D / "reports" / "main"

EPOCH = re.compile(r"Epoch: +(\d+) \| learning rate: ([0-9.e+-]+) \| training loss: ([0-9.e+-]+) .*?validation loss: ([0-9.e+-]+) \| validation mtt::r2scan_energy RMSE \(per atom\): +([0-9.]+) meV \| "
                   r"validation mtt::r2scan_energy MAE \(per atom\): +([0-9.]+) meV \| validation forces\[mtt::r2scan_energy\] RMSE: +([0-9.]+) meV/A \| validation forces\[mtt::r2scan_energy\] MAE: +([0-9.]+) meV/A")


def parse(text):
    rows = {}
    for m in EPOCH.finditer(text):
        epoch = int(m.group(1))
        rows[epoch] = {"lr": float(m.group(2)), "train_loss": float(m.group(3)), "val_loss": float(m.group(4)), "val_e_mae": float(m.group(6)), "val_f_mae": float(m.group(8))}
        rows[epoch]["mae_prod"] = rows[epoch]["val_e_mae"] * rows[epoch]["val_f_mae"]
    return rows


def buffers(state):
    composition = mts.load_buffer(state["additive_models.0.mtt::r2scan_energy_composition_buffer"]).block().values.numpy().ravel()
    scale = mts.load_buffer(state["scaler.mtt::r2scan_energy_scaler_buffer"]).block().values.numpy().ravel()
    return composition, scale


# one allocation per log section, split at every restart line, each resuming from the checkpoint named in that line
def allocations(log):
    parts = re.split(r"Restarting training from '([^']+)'", log)
    sections = [{"text": parts[0], "resumed_from": None}]
    for i in range(1, len(parts), 2):
        sections.append({"text": parts[i + 1], "resumed_from": parts[i]})
    return sections


def audit(run_dir):
    checks, lines = [], []
    log = (run_dir / "train.log").read_text()
    steps = sorted({int(m) for m in re.findall(r"grad-clip diagnostics: steps (\d+)", log)})
    assert len(steps) == 1, steps
    steps = steps[0]
    sections = allocations(log)
    parsed = [parse(s["text"]) for s in sections]
    lines.append(f"- {len(sections)} allocation(s); epochs logged per allocation {[ (min(p), max(p)) if p else None for p in parsed]}; steps per epoch {steps}")

    # lineage, the epochs that survive into the final model
    lineage = {}
    for k, (section, rows) in enumerate(zip(sections, parsed)):
        if section["resumed_from"] is not None:
            resumed_epoch = int(re.search(r"model_(\d+)\.ckpt", section["resumed_from"]).group(1))
            lines.append(f"- allocation {k + 1} resumed from epoch {resumed_epoch} checkpoint, first logged epoch {min(rows)}")
            checks.append((f"allocation {k + 1} starts one epoch after its checkpoint ({resumed_epoch} then {min(rows)})", min(rows) == resumed_epoch + 1))
            previous = parsed[k - 1]
            overlap = [e for e in rows if e in previous]
            if overlap:
                lr_gap = max(abs(previous[e]["lr"] - rows[e]["lr"]) / previous[e]["lr"] for e in overlap)
                ratio = rows[min(overlap)]["train_loss"] / previous[min(overlap)]["train_loss"]
                lines.append(f"- epochs {min(overlap)} to {max(overlap)} redone, learning rate gap {lr_gap:.1e}, training loss ratio at epoch {min(overlap)} {ratio:.3f}")
                checks.append((f"allocation {k + 1}: scheduler restored, learning rates of the redone epochs identical", lr_gap < 1e-3))
                checks.append((f"allocation {k + 1}: training loss continues (ratio within 0.8 to 1.25)", 0.8 < ratio < 1.25))
            checks.append((f"allocation {k + 1}: no warmup after the restart, learning rate non increasing", all(rows[e + 1]["lr"] <= rows[e]["lr"] for e in sorted(rows)[:-1])))
            lineage = {e: v for e, v in lineage.items() if e <= resumed_epoch}
        lineage.update(rows)
    final_epochs = sorted(lineage)
    checks.append((f"lineage covers epochs 0 to {max(final_epochs)} without gaps", final_epochs == list(range(0, max(final_epochs) + 1))))

    found = {p.name: p for p in run_dir.glob("outputs/*/*/model*.ckpt")}
    found["model.ckpt (final)"] = run_dir / "model.ckpt"
    states = {k: torch.load(p, map_location="cpu", weights_only=False) for k, p in found.items()}
    final = states["model.ckpt (final)"]
    comp_ref, scale_ref = buffers(final["model_state_dict"])
    comp_gap = scale_gap = 0.0
    for k, c in sorted(states.items()):
        opt_steps = sorted({int(v["step"]) for v in c["optimizer_state_dict"]["state"].values()}) if c["optimizer_state_dict"] and c["optimizer_state_dict"].get("state") else []
        sched = (c["scheduler_state_dict"] or {}).get("last_epoch")
        comp, scale = buffers(c["model_state_dict"])
        comp_gap = max(comp_gap, float(np.abs(comp - comp_ref).max()))
        scale_gap = max(scale_gap, float(np.abs(scale - scale_ref).max()))
        if c["best_model_state_dict"] is not None:
            comp_gap = max(comp_gap, float(np.abs(buffers(c["best_model_state_dict"])[0] - comp_ref).max()))
        expected = (c["epoch"] + 1) * steps
        lines.append(f"- {k}: epoch {c['epoch']}, best_epoch {c['best_epoch']}, best_metric {c['best_metric']:.6e}, adam steps {opt_steps}, scheduler last_epoch {sched}, expected {expected}")
        checks.append((f"{k}: optimizer and scheduler step counts equal (epoch + 1) x {steps} = {expected}", opt_steps == [expected] and sched == expected))
    lines.append(f"- composition weights and scale identical across all checkpoints and best states, largest gaps {comp_gap:.1e} and {scale_gap:.1e} eV")
    checks.append(("normalization unchanged across all checkpoints", comp_gap == 0.0 and scale_gap == 0.0))

    # the log prints MAEs with four to five significant digits, so products from the log separate epochs only beyond 5e-4 relative
    prod = {e: v["mae_prod"] for e, v in lineage.items()}
    best_all = min(prod, key=prod.get)
    ratio = final["best_metric"] / prod[best_all]
    lines.append(f"- lineage mae_prod minimum from the log at epoch {best_all}, final best_epoch {final['best_epoch']}, checkpoint best_metric over the log product {ratio:.3e} (a unit factor)")

    def tracked(c, limit):
        candidates = {e: v for e, v in prod.items() if e <= limit}
        best = min(candidates, key=candidates.get)
        recorded = candidates.get(c["best_epoch"])
        if recorded is None:
            return False, best, None
        rel = (recorded - candidates[best]) / candidates[best]
        return c["best_epoch"] == best or rel < 5e-4, best, rel

    ok, best, rel = tracked(final, final["epoch"])
    note = "" if rel is None or rel == 0 else f", recorded epoch within {rel:.1e} relative of the log minimum, below the log rounding"
    checks.append((f"final best epoch ({final['best_epoch']}) is the lineage minimum of validation mae_prod ({best}{note})", ok))
    for k, c in sorted(states.items()):
        if c["epoch"] < final["epoch"]:
            ok, best, rel = tracked(c, c["epoch"])
            note = "" if rel is None or rel == 0 else f", within {rel:.1e} relative, below the log rounding"
            checks.append((f"{k}: recorded best epoch ({c['best_epoch']}) is the lineage minimum up to its epoch ({best}{note})", ok))
    # the checkpoint metric itself must never increase along the lineage of checkpoints
    ordered = sorted((c for c in states.values()), key=lambda c: c["epoch"])
    checks.append(("best_metric never increases from one checkpoint to the next", all(b["best_metric"] <= a["best_metric"] + 1e-12 for a, b in zip(ordered, ordered[1:]))))
    best_state, last_state = final["best_model_state_dict"], final["model_state_dict"]
    keys = [k for k in best_state if k in last_state and torch.is_tensor(best_state[k]) and best_state[k].dtype.is_floating_point and best_state[k].shape == last_state[k].shape]
    lines.append(f"- best weights differ from the last weights by at most {max(float((best_state[k] - last_state[k]).abs().max()) for k in keys):.3e}")
    lines.append("- not restored by metatrain on a restart: random number generator and sampler state, so redone epochs saw another batch order")
    return lines, checks


def main():
    text = "# direct_r2scan_18k, restart audit\n\nChecks from the checkpoints and training log of each run: allocations, optimizer and scheduler step counts, normalization, best checkpoint tracking.\n"
    passed = True
    record = {}
    for run_dir in sorted(p.parent for p in (D / "runs" / "main").glob("*/options.yaml")):
        lines, checks = audit(run_dir)
        passed = passed and all(ok for _, ok in checks)
        record[run_dir.name] = [{"check": c, "passed": ok} for c, ok in checks]
        text += f"\n## {run_dir.name}\n\n" + "\n".join(lines) + "\n\n" + "\n".join(f"- {'PASS' if ok else 'FAIL'} {c}" for c, ok in checks) + "\n"
    (OUT / "RESTART_AUDIT.md").write_text(text)
    (OUT / "restart_audit.json").write_text(json.dumps(record, indent=1) + "\n")
    print(text)
    print("AUDIT", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
