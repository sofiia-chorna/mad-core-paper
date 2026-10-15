import json
import re
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
SNAPSHOTS = HERE / "snapshots"
AIMS = HERE / "aims"
TABLES = HERE / "tables"

FUNCTIONALS = ("pw-lda", "pbe", "r2scan")
DELTA_OF = {"pw-lda": "delta_lda", "pbe": "delta_pbe"}
SPECIES = ("H", "O")
EVENT_OFFSETS = {"before": 0.0, "during": 0.5, "after": 1.0}


def parse_aims(path):
    text = path.read_text()
    result = {"converged": "Self-consistency cycle converged." in text, "finished": "Have a nice day." in text,
              "scf_cycles": text.count("Begin self-consistency iteration")}
    energies = re.findall(r"\| Electronic free energy\s+:\s+(-?\d+\.\d+E[+-]\d+) eV", text)
    result["free_energy_eV"] = float(energies[-1]) if energies else None
    blocks = text.split("Total atomic forces (unitary forces cleaned) [eV/Ang]:")
    if len(blocks) > 1:
        rows = []
        for line in blocks[-1].splitlines()[1:]:
            m = re.match(r"\s*\|\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
            if not m:
                break
            rows.append([float(m.group(i)) for i in (2, 3, 4)])
        result["forces"] = np.array(rows)
    else:
        result["forces"] = None
    return result


def errors(pred, ref, symbols):
    out = {}
    for s in SPECIES + ("all",):
        mask = np.ones(len(symbols), bool) if s == "all" else symbols == s
        d = (pred[mask] - ref[mask]).ravel()
        out[s] = {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d**2).mean())), "max": float(np.abs(d).max()),
                  "ref_rms": float(np.sqrt((ref[mask] ** 2).mean()))}
    return out


def pooled(pairs):
    pred = np.concatenate([p for p, _, _ in pairs])
    ref = np.concatenate([r for _, r, _ in pairs])
    symbols = np.concatenate([s for _, _, s in pairs])
    return errors(pred, ref, symbols)


def fmt(e, s):
    return f"{1000 * e[s]['mae']:.0f} / {1000 * e[s]['rmse']:.0f} / {1000 * e[s]['max']:.0f}"


def mic(v, length):
    return v - length * np.round(v / length)


def main():
    predictions = json.load(open(TABLES / "model_predictions.json"))
    arrays = np.load(TABLES / "model_predictions.npz")
    event = json.load(open(TABLES / "event.json"))["event"]

    frames = {}
    for path in sorted(SNAPSHOTS.glob("*.extxyz")):
        atoms = read(path, format="extxyz")
        name = path.stem
        f = {"atoms": atoms, "symbols": np.array(atoms.get_chemical_symbols()), "natoms": len(atoms), "dft": {}, "model_energy": predictions["snapshots"][name],
             "model_forces": {m: arrays[f"snapshots/{name}/{m}"] for m in ("baseline", "delta_lda", "delta_pbe")}}
        for functional in FUNCTIONALS:
            out = AIMS / name / functional / "aims.out"
            if out.exists():
                f["dft"][functional] = parse_aims(out)
        frames[name] = f

    lines = ["# Snapshot check against LDA, PBE and r2SCAN single points", ""]
    status = []
    for name, f in frames.items():
        for functional in FUNCTIONALS:
            r = f["dft"].get(functional)
            status.append(f"| {name} | {functional} | " + ("missing" if r is None else ("converged" if r["converged"] and r["finished"] else "NOT converged")) +
                          f" | {r['scf_cycles'] if r else ''} | {r['free_energy_eV'] if r else ''} |")
    lines += ["## FHI-aims single points", "", "| snapshot | functional | state | scf cycles | free energy, eV |", "|---|---|---|---|---|"] + status + [""]

    complete = {n: f for n, f in frames.items() if all(functional in f["dft"] and f["dft"][functional]["converged"] and f["dft"][functional]["forces"] is not None for functional in FUNCTIONALS)}
    lines.append(f"{len(complete)} of {len(frames)} snapshots have all three functionals converged; the comparisons below use those.")
    lines.append("")
    if not complete:
        (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
        return

    # force comparisons, pooled over the complete snapshots, meV/A
    comparisons = {
        "A baseline PET-MAD-1.6-M against r2SCAN": lambda f: (f["model_forces"]["baseline"], f["dft"]["r2scan"]["forces"]),
        "B1 delta_LDA against F_r2SCAN minus F_LDA": lambda f: (f["model_forces"]["delta_lda"], f["dft"]["r2scan"]["forces"] - f["dft"]["pw-lda"]["forces"]),
        "B2 delta_PBE against F_r2SCAN minus F_PBE": lambda f: (f["model_forces"]["delta_pbe"], f["dft"]["r2scan"]["forces"] - f["dft"]["pbe"]["forces"]),
        "C1 approx. LDA (baseline minus delta_LDA) against LDA": lambda f: (f["model_forces"]["baseline"] - f["model_forces"]["delta_lda"], f["dft"]["pw-lda"]["forces"]),
        "C2 approx. PBE (baseline minus delta_PBE) against PBE": lambda f: (f["model_forces"]["baseline"] - f["model_forces"]["delta_pbe"], f["dft"]["pbe"]["forces"]),
        "D1 context, baseline alone against LDA": lambda f: (f["model_forces"]["baseline"], f["dft"]["pw-lda"]["forces"]),
        "D2 context, baseline alone against PBE": lambda f: (f["model_forces"]["baseline"], f["dft"]["pbe"]["forces"]),
        "D3 context, DFT r2SCAN against DFT LDA": lambda f: (f["dft"]["r2scan"]["forces"], f["dft"]["pw-lda"]["forces"]),
        "D4 context, DFT r2SCAN against DFT PBE": lambda f: (f["dft"]["r2scan"]["forces"], f["dft"]["pbe"]["forces"]),
    }
    record = {"complete_snapshots": sorted(complete), "pooled_forces_meV_A": {}, "per_snapshot_forces_meV_A": {}}
    lines += ["## Forces pooled over the complete snapshots, MAE / RMSE / max in meV/A per Cartesian component", "",
              "| comparison | H | O | all | RMS of the reference, H / O |", "|---|---|---|---|---|"]
    for label, pick in comparisons.items():
        e = pooled([(*pick(f), f["symbols"]) for f in complete.values()])
        record["pooled_forces_meV_A"][label] = e
        lines.append(f"| {label} | {fmt(e, 'H')} | {fmt(e, 'O')} | {fmt(e, 'all')} | {1000 * e['H']['ref_rms']:.0f} / {1000 * e['O']['ref_rms']:.0f} |")
    lines.append("")
    lines += ["## Per snapshot force RMSE in meV/A, H / O", "", "| snapshot | A base vs r2SCAN | B1 dLDA | B2 dPBE | C1 approx LDA vs LDA | C2 approx PBE vs PBE | D3 r2SCAN vs LDA | D4 r2SCAN vs PBE |", "|---|---|---|---|---|---|---|---|"]
    keys = list(comparisons)
    for name, f in complete.items():
        es = {k: errors(*comparisons[k](f), f["symbols"]) for k in keys}
        record["per_snapshot_forces_meV_A"][name] = es
        cells = [f"{1000 * es[k]['H']['rmse']:.0f} / {1000 * es[k]['O']['rmse']:.0f}" for k in (keys[0], keys[1], keys[2], keys[3], keys[4], keys[7], keys[8])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # decomposition, the approx LDA error is the baseline error plus the correction error
    lines += ["## Decomposition of the approximate potential error, RMS over all components in meV/A", "", "approx minus target = (baseline minus r2SCAN) + (r2SCAN minus target minus delta). The two terms and their sum per snapshot.", "",
              "| snapshot | LDA: baseline term | LDA: correction term | LDA: sum | PBE: baseline term | PBE: correction term | PBE: sum |", "|---|---|---|---|---|---|---|"]
    for name, f in complete.items():
        row = [name]
        for functional in ("pw-lda", "pbe"):
            base_term = f["model_forces"]["baseline"] - f["dft"]["r2scan"]["forces"]
            corr_term = f["dft"]["r2scan"]["forces"] - f["dft"][functional]["forces"] - f["model_forces"][DELTA_OF[functional]]
            total = base_term + corr_term
            row += [f"{1000 * np.sqrt((base_term**2).mean()):.0f}", f"{1000 * np.sqrt((corr_term**2).mean()):.0f}", f"{1000 * np.sqrt((total**2).mean()):.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # relative energies, every snapshot is H64O32, so the mean over the complete snapshots is the common reference
    sources = {
        "DFT LDA": lambda f: f["dft"]["pw-lda"]["free_energy_eV"],
        "DFT PBE": lambda f: f["dft"]["pbe"]["free_energy_eV"],
        "DFT r2SCAN": lambda f: f["dft"]["r2scan"]["free_energy_eV"],
        "PET-MAD-1.6-M": lambda f: f["model_energy"]["baseline"],
        "approx. LDA": lambda f: f["model_energy"]["baseline"] - f["model_energy"]["delta_lda"],
        "approx. PBE": lambda f: f["model_energy"]["baseline"] - f["model_energy"]["delta_pbe"],
        "DFT r2SCAN minus LDA": lambda f: f["dft"]["r2scan"]["free_energy_eV"] - f["dft"]["pw-lda"]["free_energy_eV"],
        "delta_LDA model": lambda f: f["model_energy"]["delta_lda"],
        "DFT r2SCAN minus PBE": lambda f: f["dft"]["r2scan"]["free_energy_eV"] - f["dft"]["pbe"]["free_energy_eV"],
        "delta_PBE model": lambda f: f["model_energy"]["delta_pbe"],
    }
    names = sorted(complete)
    natoms = complete[names[0]]["natoms"]
    rel = {}
    for label, pick in sources.items():
        values = np.array([pick(complete[n]) for n in names])
        rel[label] = 1000.0 * (values - values.mean()) / natoms
    record["relative_energies_meV_per_atom"] = {label: dict(zip(names, map(float, v))) for label, v in rel.items()}
    lines += ["## Relative energies in meV/atom, each source minus its own mean over the complete snapshots", "", "| snapshot | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for i, n in enumerate(names):
        lines.append(f"| {n} | " + " | ".join(f"{rel[label][i]:+.1f}" for label in sources) + " |")
    pairs = [("PET-MAD-1.6-M", "DFT r2SCAN"), ("approx. LDA", "DFT LDA"), ("approx. PBE", "DFT PBE"), ("delta_LDA model", "DFT r2SCAN minus LDA"), ("delta_PBE model", "DFT r2SCAN minus PBE")]
    lines += ["", "| model against reference | MAE of relative energies, meV/atom | max | spread of the reference (max minus min) |", "|---|---|---|---|"]
    record["relative_energy_errors_meV_per_atom"] = {}
    for a, b in pairs:
        d = rel[a] - rel[b]
        record["relative_energy_errors_meV_per_atom"][f"{a} vs {b}"] = {"mae": float(np.abs(d).mean()), "max": float(np.abs(d).max()), "reference_spread": float(rel[b].max() - rel[b].min())}
        lines.append(f"| {a} vs {b} | {np.abs(d).mean():.1f} | {np.abs(d).max():.1f} | {rel[b].max() - rel[b].min():.1f} |")
    lines.append("")

    # the event, energies relative to before and forces on the three atoms along the donor to acceptor axis
    roles = [r for r in ("before", "during", "after") if f"event_{r}" in complete]
    if "before" in roles and len(roles) > 1:
        lines += ["## The proton transfer event, energies relative to the before snapshot in meV/atom (eV per cell in brackets)", "", "| source | " + " | ".join(roles[1:]) + " |", "|---|" + "---|" * (len(roles) - 1)]
        for label in ("DFT LDA", "DFT PBE", "DFT r2SCAN", "PET-MAD-1.6-M", "approx. LDA", "approx. PBE"):
            pick = sources[label]
            e0 = pick(complete["event_before"])
            lines.append(f"| {label} | " + " | ".join(f"{1000 * (pick(complete[f'event_{r}']) - e0) / natoms:+.1f} ({pick(complete[f'event_{r}']) - e0:+.3f})" for r in roles[1:]) + " |")
        lines.append("")
    h, donor, acceptor = event["hydrogen"], event["donor"], event["acceptor"]
    lines += ["## Forces on the transferring proton and its two oxygens, component along the donor to acceptor axis in eV/A (positive pushes toward the acceptor)", "",
              f"hydrogen {h}, donor oxygen {donor}, acceptor oxygen {acceptor}, run {event['run']}. For the oxygens the same axis is used, so a positive value moves the oxygen toward the acceptor side.", "",
              "| snapshot | source | H along axis | H magnitude | donor O along axis | acceptor O along axis | donor H, acceptor H distance, A |", "|---|---|---|---|---|---|---|"]
    record["event_forces"] = {}
    for r in roles:
        f = complete[f"event_{r}"]
        pos = f["atoms"].positions
        length = f["atoms"].cell.array[0, 0]
        axis = mic(pos[acceptor] - pos[donor], length)
        axis /= np.linalg.norm(axis)
        d_donor = np.linalg.norm(mic(pos[h] - pos[donor], length))
        d_acceptor = np.linalg.norm(mic(pos[h] - pos[acceptor], length))
        forces = {"DFT LDA": f["dft"]["pw-lda"]["forces"], "DFT PBE": f["dft"]["pbe"]["forces"], "DFT r2SCAN": f["dft"]["r2scan"]["forces"],
                  "PET-MAD-1.6-M": f["model_forces"]["baseline"], "approx. LDA": f["model_forces"]["baseline"] - f["model_forces"]["delta_lda"],
                  "approx. PBE": f["model_forces"]["baseline"] - f["model_forces"]["delta_pbe"]}
        record["event_forces"][r] = {}
        for label, F in forces.items():
            record["event_forces"][r][label] = {"h_along": float(F[h] @ axis), "h_norm": float(np.linalg.norm(F[h])), "donor_along": float(F[donor] @ axis), "acceptor_along": float(F[acceptor] @ axis)}
            lines.append(f"| event_{r} | {label} | {F[h] @ axis:+.2f} | {np.linalg.norm(F[h]):.2f} | {F[donor] @ axis:+.2f} | {F[acceptor] @ axis:+.2f} | {d_donor:.3f}, {d_acceptor:.3f} |")
    lines.append("")

    # the md forces against the float64 models
    gaps = {n: f["model_energy"]["md_force_gap_eV_A"] for n, f in frames.items()}
    lines += ["## Consistency of the stored MD forces with the float64 models", "", f"largest difference between the forces i-PI stored and baseline minus correction recomputed in float64, over the nine snapshots: {max(gaps.values()):.1e} eV/A", ""]
    record["md_force_gap_eV_A"] = gaps

    (TABLES / "comparison.json").write_text(json.dumps(record, indent=1) + "\n")
    (HERE / "REPORT_SNAPSHOTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
