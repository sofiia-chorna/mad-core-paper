import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
LABELLED = HERE / "existing_labels"

SPECIES = ("H", "O")


def errors(pred, ref, symbols):
    out = {}
    for s in SPECIES + ("all",):
        mask = np.ones(len(symbols), bool) if s == "all" else symbols == s
        d = (pred[mask] - ref[mask]).ravel()
        out[s] = {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d**2).mean())), "max": float(np.abs(d).max()), "ref_rms": float(np.sqrt((ref[mask] ** 2).mean()))}
    return out


def fmt(e, s):
    return f"{1000 * e[s]['mae']:.0f} / {1000 * e[s]['rmse']:.0f} / {1000 * e[s]['max']:.0f}"


def main():
    predictions = json.load(open(TABLES / "model_predictions.json"))["labelled"]
    arrays = np.load(TABLES / "model_predictions.npz")
    info = json.load(open(LABELLED / "frames.json"))

    water = {cid: i for cid, i in info.items() if i["n_h"] == 2 * i["n_o"] and cid in predictions}
    groups = {"periodic liquid water frame, 80 molecules, 1.249 g/cm3 (AENET)": [c for c, i in water.items() if i["periodic"] and i["natoms"] == 240],
              "periodic ice frame, 64 molecules in a 23.5 A box (water_ice)": [c for c, i in water.items() if i["periodic"] and i["natoms"] == 192],
              "isolated clusters, 70 molecules (OMol25)": [c for c, i in water.items() if not i["periodic"] and i["natoms"] == 210],
              "isolated clusters, 40 molecules (OMol25)": [c for c, i in water.items() if not i["periodic"] and i["natoms"] == 120],
              "isolated clusters, 8 to 25 molecules": [c for c, i in water.items() if not i["periodic"] and 24 <= i["natoms"] < 120]}
    comparisons = {
        "C1 approx. LDA against LDA": ("lda", lambda c: arrays[f"labelled/{c}/baseline"] - arrays[f"labelled/{c}/delta_lda"]),
        "C2 approx. PBE against PBE": ("pbe", lambda c: arrays[f"labelled/{c}/baseline"] - arrays[f"labelled/{c}/delta_pbe"]),
        "D1 context, baseline alone against LDA": ("lda", lambda c: arrays[f"labelled/{c}/baseline"]),
        "D2 context, baseline alone against PBE": ("pbe", lambda c: arrays[f"labelled/{c}/baseline"]),
        "D5 context, delta_LDA against F_PBE minus F_LDA (not its target)": ("lda", lambda c: arrays[f"labelled/{c}/pbe_forces"] - arrays[f"labelled/{c}/delta_lda"] + arrays[f"labelled/{c}/delta_pbe"]),
    }
    lines = ["# Existing LDA and PBE labels on water-like structures", "",
             "Frames of the LDA/PBE campaign whose atoms are only H and O with H to O ratio 2, outside the correction training set (random arms). They carry LDA and PBE labels but no r2SCAN label, so only the combined potentials can be tested on them. Forces as MAE / RMSE / max in meV/A per Cartesian component.", ""]
    record = {}
    for group, cids in groups.items():
        if not cids:
            continue
        lines += [f"## {group}, {len(cids)} frame(s), {sum(info[c]['natoms'] for c in cids)} atoms", "", "| comparison | H | O | all | RMS of the reference, H / O |", "|---|---|---|---|---|"]
        record[group] = {"frames": cids, "forces": {}}
        for label, (functional, pick) in comparisons.items():
            pred = np.concatenate([pick(c) for c in cids])
            ref = np.concatenate([arrays[f"labelled/{c}/{functional}_forces"] for c in cids])
            symbols = np.concatenate([arrays[f"labelled/{c}/symbols"] for c in cids])
            e = errors(pred, ref, symbols)
            record[group]["forces"][label] = e
            lines.append(f"| {label} | {fmt(e, 'H')} | {fmt(e, 'O')} | {fmt(e, 'all')} | {1000 * e['H']['ref_rms']:.0f} / {1000 * e['O']['ref_rms']:.0f} |")
        lines.append("")
        # relative energies within groups of identical composition
        if len(cids) > 1 and len({info[c]["natoms"] for c in cids}) == 1:
            n = info[cids[0]]["natoms"]
            lines += [f"Relative energies among these {len(cids)} frames of identical composition, each source minus its own mean, meV/atom:", "", "| frame | DFT LDA | approx. LDA | DFT PBE | approx. PBE | PET-MAD-1.6-M |", "|---|---|---|---|---|---|"]
            series = {
                "DFT LDA": np.array([predictions[c]["lda_free_energy_eV"] for c in cids]),
                "approx. LDA": np.array([predictions[c]["model_energy_eV"]["baseline"] - predictions[c]["model_energy_eV"]["delta_lda"] for c in cids]),
                "DFT PBE": np.array([predictions[c]["pbe_free_energy_eV"] for c in cids]),
                "approx. PBE": np.array([predictions[c]["model_energy_eV"]["baseline"] - predictions[c]["model_energy_eV"]["delta_pbe"] for c in cids]),
                "PET-MAD-1.6-M": np.array([predictions[c]["model_energy_eV"]["baseline"] for c in cids]),
            }
            rel = {k: 1000 * (v - v.mean()) / n for k, v in series.items()}
            for i, c in enumerate(cids):
                lines.append(f"| {c} | " + " | ".join(f"{rel[k][i]:+.1f}" for k in series) + " |")
            record[group]["relative_energy_mae_meV_per_atom"] = {"approx. LDA vs DFT LDA": float(np.abs(rel["approx. LDA"] - rel["DFT LDA"]).mean()), "approx. PBE vs DFT PBE": float(np.abs(rel["approx. PBE"] - rel["DFT PBE"]).mean())}
            lines.append("")
            lines.append(f"MAE of the relative energies: approx. LDA vs DFT LDA {record[group]['relative_energy_mae_meV_per_atom']['approx. LDA vs DFT LDA']:.1f}, approx. PBE vs DFT PBE {record[group]['relative_energy_mae_meV_per_atom']['approx. PBE vs DFT PBE']:.1f} meV/atom.")
            lines.append("")

    (TABLES / "labelled_comparison.json").write_text(json.dumps(record, indent=1) + "\n")
    (HERE / "REPORT_LABELLED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
