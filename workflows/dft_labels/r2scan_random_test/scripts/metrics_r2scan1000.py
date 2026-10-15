import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TEST = Path(__file__).resolve().parent.parent
VALID = TEST.parent
sys.path.insert(0, str(VALID / "delta_pbe_10k_vs_mad16" / "scripts"))
sys.path.insert(0, str(VALID / "paper_demo" / "delta_capacity" / "scripts"))

from delta_common import read_tsv, write_tsv
from error_cache import errors, read_pred, read_truth

SET_TAG = "test_r2scan_1000"
SET_XYZ = TEST / "manifests" / "xyz" / f"{SET_TAG}.xyz"
MANIFEST = TEST / "manifests" / f"{SET_TAG}.tsv"
SET_JSON = TEST / "manifests" / f"{SET_TAG}.json"
MISSING = TEST / "results" / "missing_frames.tsv"

PRED_DIRS = {"matched": VALID / "delta_pbe_10k_vs_mad16" / "reports" / "preds",
             "capacity": VALID / "paper_demo" / "delta_capacity" / "reports" / "preds"}
REPORTS = TEST / "reports"
CACHE = REPORTS / "cache"

MIN_GROUP = 15
IDENTITY_TOL = 1e-6


def stats(err):
    signed = np.asarray(err, dtype=float)
    a = np.abs(signed)
    keep = a <= np.percentile(a, 99)
    return {"mae": float(a.mean()), "rmse": float(np.sqrt((a ** 2).mean())), "p8": float((a ** 8).mean() ** (1 / 8)),
            "sup": float(a.max()), "median": float(np.median(a)), "mae_trim1": float(a[keep].mean()),
            "rmse_trim1": float(np.sqrt((a[keep] ** 2).mean())), "mean_signed": float(signed.mean())}


def describe(stem):
    body = stem.split("-", 1)[1].rsplit("__", 1)[0]
    parts = body.split("--")
    target = parts[0]
    model = parts[1] if len(parts) > 2 else "omat-s"
    arm = parts[2] if len(parts) > 3 else parts[-2]
    return target, model, arm


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    truth = read_truth(SET_XYZ)
    natoms = truth["natoms"]
    n = len(natoms)
    delta_forces_true = [r - p for r, p in zip(truth["f_r2scan"], truth["f_pbe"])]
    groups = {r["configuration_id"]: r["subset"] for r in read_tsv(MANIFEST)}
    group_of_frame = np.array([groups[c] for c in truth["configuration_id"]])

    baseline = errors(truth["e_pbe"], truth["f_pbe"], truth["e_r2scan"], truth["f_r2scan"], natoms)
    np.savez(CACHE / f"pbe_alone@{SET_TAG}.npz", **baseline)

    rows = [{"stage": "", "target": "pbe_alone", "model": "", "arm": "", "n": n,
             **{f"{q}_{s}": round(v, 2) for q, e in (("E", stats(baseline["de"])), ("F", stats(baseline["df_mean"]))) for s, v in e.items()}}]
    per_structure = {"pbe_alone": baseline}
    identity = {}

    for stage, pred_dir in PRED_DIRS.items():
        for pred in sorted(pred_dir.glob(f"*@{SET_TAG}.xyz")):
            stem = pred.stem.rsplit("@", 1)[0]
            target, model, arm = describe(stem)

            n_pred, e_pred, f_pred = read_pred(pred, "mtt::delta_energy" if target == "delta" else "energy")
            assert len(n_pred) == n and (n_pred == natoms).all(), f"{pred.name}: frame mismatch"
            assert np.isfinite(e_pred).all() and all(np.isfinite(f).all() for f in f_pred), f"{pred.name}: non-finite prediction"

            if target == "delta":
                delta_err = errors(e_pred, f_pred, truth["d_e"], delta_forces_true, natoms)
                reconstructed = errors(truth["e_pbe"] + e_pred, [p + q for p, q in zip(truth["f_pbe"], f_pred)],
                                       truth["e_r2scan"], truth["f_r2scan"], natoms)
                gap_e = float(np.abs(reconstructed["de"] - delta_err["de"]).max())
                gap_f = float(np.abs(reconstructed["df_atoms"] - delta_err["df_atoms"]).max())
                identity[stem] = {"max_abs_gap_E_meV_per_atom": gap_e, "max_abs_gap_F_meV_per_A": gap_f}
                assert gap_e < IDENTITY_TOL and gap_f < IDENTITY_TOL, f"{pred.name}: reconstruction identity violated"
                err = reconstructed
            else:
                err = errors(e_pred, f_pred, truth["e_r2scan"], truth["f_r2scan"], natoms)

            np.savez(CACHE / f"{stem}@{SET_TAG}.npz", **err)
            per_structure[f"{target}--{model}--{arm}"] = err
            rows.append({"stage": stage, "target": target, "model": model, "arm": arm, "n": n,
                         **{f"{q}_{s}": round(v, 2) for q, e in (("E", stats(err["de"])), ("F", stats(err["df_mean"]))) for s, v in e.items()}})

    write_tsv(REPORTS / "metrics.tsv", rows, list(rows[0]))

    group_counts = Counter(group_of_frame.tolist())
    big_groups = sorted([g for g, c in group_counts.items() if c >= MIN_GROUP], key=lambda g: -group_counts[g])
    per_group = defaultdict(dict)
    for name, err in per_structure.items():
        for g in big_groups:
            mask = group_of_frame == g
            per_group[name][g] = {"n": int(mask.sum()), "E_mae": round(float(np.abs(err["de"][mask]).mean()), 2),
                                  "F_mae": round(float(err["df_mean"][mask].mean()), 1)}

    set_summary = json.load(open(SET_JSON))
    missing_rows = read_tsv(MISSING)
    mechanisms = dict(Counter(r["mechanism"] for r in missing_rows))

    with open(REPORTS / "metrics.json", "w") as fh:
        json.dump({"set": set_summary, "missing_mechanisms": mechanisms, "rows": rows,
                   "per_dataset_group": per_group, "reconstruction_identity": identity}, fh, indent=1)

    write_report(rows, per_group, big_groups, group_counts, set_summary, mechanisms, identity, n)
    for r in rows:
        print(f"{r['target']:9s} {r['model']:8s} {r['arm']:22s} E {r['E_mae']:7.1f}/{r['E_rmse']:<7.1f} F {r['F_mae']:6.0f}/{r['F_rmse']:<6.0f}")


def write_report(rows, per_group, big_groups, group_counts, set_summary, mechanisms, identity, n):
    lines = [f"# Independent r2SCAN test: {n} random MAD-COSMO structures computed on Lyra", ""]
    lines.append(f"E in meV/atom (per-structure |dE|/N), F in meV/A (per-structure mean atomic |dF|); MAE, RMSE, "
                 f"p8 = (mean e^8)^(1/8), SUP over structures. Best validation checkpoint, PET-OMat, seed 42.")
    lines.append("Delta rows are reconstructed r2SCAN predictions E_PBE + dE_model and F_PBE + dF_model; they require a PBE calculation at evaluation time.")
    lines.append("")
    lines.append("## What this set is")
    lines.append("")
    lines.append(f"A uniform random draw of 1,000 structures from the PBE random test, computed at r2SCAN on SCITAS Lyra. "
                 f"**{n} converged**; {sum(mechanisms.values())} did not and are excluded "
                 f"({', '.join(f'{k} {v}' for k, v in sorted(mechanisms.items()))}). Membership is frozen; the failures are "
                 f"listed in `results/missing_frames.tsv` and are not resampled.")
    lines.append("")
    lines.append(f"Coverage: {set_summary['total_atoms']:,} atoms, {set_summary['elements']} elements, "
                 f"{len(set_summary['dataset_groups'])} source datasets, "
                 f"{set_summary['pbc']['FFF']} isolated / {set_summary['pbc'].get('TTT', 0)} periodic. "
                 f"Max |F| {max(set_summary['fmax_r2scan_max'], set_summary['fmax_pbe_max'])} eV/A, so no structure is near the 100 eV/A eligibility cut.")
    lines.append("")
    lines.append("**This set is disjoint from every set used to build or tune the models.** Checked by the same rounded-geometry "
                 "hash the delta experiment uses (positions and cell to 1e-4, pbc deliberately excluded):")
    lines.append("")
    lines.append("| compared against | n | shared structures |")
    lines.append("|---|---|---|")
    for name, overlap in set_summary["overlap_with_delta_sets"].items():
        size = len(read_tsv(VALID / "delta_pbe_10k_vs_mad16" / "manifests" / f"{name}.tsv"))
        lines.append(f"| {name} | {size:,} | {overlap} |")
    lines.append("")
    lines.append("Both sides build atomization energies from the same r2SCAN monomer reference "
                 "(`e0_r2scan_mad16_monomers.json`, sha256 911f8d01...), so the energies are directly comparable to the "
                 "delta-experiment and capacity tables.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| target | model | arm | E MAE / RMSE | F MAE / RMSE | E p8 | F p8 | E SUP | F SUP |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        label = "PBE alone" if r["target"] == "pbe_alone" else r["target"]
        lines.append(f"| {label} | {r['model']} | {r['arm']} | {r['E_mae']:.1f} / {r['E_rmse']:.1f} | "
                     f"{r['F_mae']:.0f} / {r['F_rmse']:.0f} | {r['E_p8']:.1f} | {r['F_p8']:.0f} | {r['E_sup']:.0f} | {r['F_sup']:.0f} |")
    lines.append("")
    lines.append("## Energy outlier sensitivity")
    lines.append("")
    lines.append("`trim1` drops the 1% largest structure errors. `mean signed` is the mean of the signed per-structure error, "
                 "so |mean signed| close to the MAE means the error is a systematic offset rather than scatter.")
    lines.append("")
    lines.append("| target | model | arm | E MAE | E MAE trim1 | E median | E RMSE | E RMSE trim1 | E mean signed |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        label = "PBE alone" if r["target"] == "pbe_alone" else r["target"]
        lines.append(f"| {label} | {r['model']} | {r['arm']} | {r['E_mae']:.2f} | {r['E_mae_trim1']:.2f} | "
                     f"{r['E_median']:.2f} | {r['E_rmse']:.2f} | {r['E_rmse_trim1']:.2f} | {r['E_mean_signed']:+.2f} |")
    lines.append("")
    lines.append(f"## Per source dataset (E MAE / F MAE, groups with n >= {MIN_GROUP})")
    lines.append("")
    names = [k for k in per_group if k != "pbe_alone"]
    lines.append("| dataset | n | PBE alone | " + " | ".join(names) + " |")
    lines.append("|---" * (3 + len(names)) + "|")
    for g in big_groups:
        cells = [f"{per_group[k][g]['E_mae']:.1f} / {per_group[k][g]['F_mae']:.0f}" for k in ["pbe_alone"] + names]
        lines.append(f"| {g} | {group_counts[g]} | " + " | ".join(cells) + " |")
    lines.append("")
    if identity:
        worst_e = max(v["max_abs_gap_E_meV_per_atom"] for v in identity.values())
        worst_f = max(v["max_abs_gap_F_meV_per_A"] for v in identity.values())
        lines.append(f"Reconstruction identity for the {len(identity)} delta models: max |reconstructed-r2SCAN error - delta error| "
                     f"= {worst_e:.1e} meV/atom, {worst_f:.1e} meV/A.")
        lines.append("")
    (REPORTS / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
