import os
import csv
import datetime
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts")
from common import read_tsv  # noqa: E402

S = Path(__file__).resolve().parent.parent
R = S / "runs"
M = S / "manifests"
REPORTS = S / "reports"
V2 = S.parent / "scratch_scaling_v2"
FIX = S.parent / "voronoi_weighted" / "pbe_20k" / "fix"
OUT = REPORTS / "STATUS.md"
SIZES = ("2k", "5k", "10k", "20k")
ARMS = ("madcore", "random")
RUNS = [f"scratchv3-s--{arm}_{size}--s42" for arm in ARMS for size in SIZES]
LABEL = {r: r.split("--")[1].replace("madcore", "MAD-CORE").replace("_", " ") + " + refs" for r in RUNS}
METRIC_KEYS = ["E_mae", "E_rmse", "E_p8", "E_sup", "F_mae", "F_rmse", "F_p8", "F_sup"]
TESTS = ("test_random", "test_madcore")
EPOCH = re.compile(r"\[([0-9-]+ [0-9:]+)\].*Epoch: +(\d+) \| learning rate: ([0-9.e+-]+) \| training loss: ([0-9.e+-]+) \|.*?validation loss: ([0-9.e+-]+) "
                   r"\| validation energy RMSE \(per atom\): +([0-9.]+) meV \| validation energy MAE \(per atom\): +([0-9.]+) meV \| validation forces RMSE: +([0-9.]+) meV/A \| validation forces MAE: +([0-9.]+) meV/A")
BEST = re.compile(r"Using best model from epoch (\d+)")
PACKAGE = re.compile(r"Package directory: (\S+)")


def parse_log(path):
    rows, restarts = {}, 0
    if not path.exists():
        return [], 0
    for line in path.read_text(errors="replace").splitlines():
        if "Starting training" in line:
            restarts += 1
        m = EPOCH.search(line)
        if m:
            e = int(m.group(2))
            rows[e] = {"epoch": e, "time": datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), "val_e_mae": float(m.group(7)), "val_f_mae": float(m.group(9))}
    return [rows[e] for e in sorted(rows)], restarts


def eta(rows, planned):
    if len(rows) < 5:
        return "-"
    k = min(40, len(rows) - 1)
    rate = (rows[-1]["time"] - rows[-1 - k]["time"]).total_seconds() / k
    remaining = rate * (planned - len(rows))
    return f"{rate:.1f} s/epoch, about {remaining / 3600:.1f} h of run time left"


def datasets_section(sets):
    prov = json.load(open(R / "PROVENANCE.json"))["runs"]
    lines = ["| set | ordinary structures | added references | total structures | atoms (ordinary + 102) | elements before -> after | steps per epoch | epochs | updates | snapshot interval |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        train = run.split("--")[1]
        v, p = sets["sets"][train], prov[run]
        lines.append(f"| {train} | {v['ordinary_structures']} | {v['added_references']} | {v['total_structures']} | {v['atoms_total']} | {v['elements_before']} -> {v['elements_after']} | {v['steps_per_epoch_batch_64_drop_last']} | {p['epochs']} | {p['optimizer_updates']} | {p.get('checkpoint_interval', 5)} |")
    dup = {k: v["ordinary_nonperiodic_single_atoms_also_present_as_reference"] for k, v in sets["duplicates"].items() if v["ordinary_nonperiodic_single_atoms_also_present_as_reference"]}
    lines += ["", f"Ordinary structures are the round 2 orderings and exclusions unchanged (nested 2k in 5k in 10k in 20k, verified; ordinary frames byte identical to round 2). The 102 references are one non periodic single atom per element 1 to 102, cut from `dft/lda_pbe/references/monomers_pbe.xyz` "
              f"(sha256 {sets['monomers_pbe_sha256'][:16]}), the very monomer free energies that are subtracted in every training atomization energy, so their targets are exactly 0 eV atomization energy and zero forces, applied to the complete prediction (composition baseline and scale included). "
              f"They are appended after the ordinary structures, sampled like any structure, unweighted loss, no oversampling. Species vocabulary 102 for every run, as in rounds 1 and 2.", "",
              f"Content duplicates reported, not resolved (ordinary non periodic single atoms already in a training set): {dup}. No frozen validation or test set contains a non periodic single atom and no reference id collides with any frozen id (`SETS.json`, `checks`). "
              f"Label consistency of those duplicates (`reports/duplicate_isolated_atoms_check.txt`): ordinary frame and reference carry the same monomer free energy to within 3e-9 eV, atomization energies of order 1e-9 eV versus exactly 0.0, zero forces in both, both non periodic; nothing was changed. "
              f"Eligibility of the ordinary structures (pair rule, both labels, max force norm <= 100 eV/A in PBE and LDA, zero violations) was verified for every round 2 set in `{sets['checks']['eligibility_verification']}`.", ""]
    return lines


def provenance_section():
    base = (FIX / "provenance" / "base_commit.txt").read_text().strip()
    hashes = (FIX / "frozen_src" / "SHA256SUMS").read_text().strip().replace("\n", "; ")
    patch = subprocess.run(["sha256sum", str(FIX / "metatrain_attention_mask.patch")], capture_output=True, text=True).stdout.split()[0]
    packages = set()
    for run in RUNS + ["smoke/scratchv3-s--smoke_mixed--s42"]:
        p = R / run / "train.log"
        if p.exists():
            packages |= set(PACKAGE.findall(p.read_text(errors="replace")))
    return [f"Environment: `${HOME}/venv-metatrain-fix`, metatrain v2026.4 upstream commit {base}, corrected attention mask patch `fix/metatrain_attention_mask.patch` (sha256 {patch[:16]}), frozen installed sources {hashes}; "
            f"every job runs `fix/verify_fix.sh` (checksums, entry point, imported package path) before training. Package directories seen in the train logs of this round: {sorted(packages) or 'none yet'}. "
            f"The local loss.py and trainer.py patches of the original runs (gradient clip diagnostics, weighted loss hook) are unchanged (`fix/provenance/`).", ""]


def smoke_section():
    d = R / "smoke" / "scratchv3-s--smoke_mixed--s42"
    lines = []
    for name in ("init_check.json", "selection_check.json", "references_check.json"):
        try:
            json.load(open(d / name))
        except Exception:
            lines.append(f"{name}: missing or unreadable")
    if (d / "init_check.json").exists() and f"init_check.json: missing or unreadable" not in lines:
        j = json.load(open(d / "init_check.json"))
        lines.append(f"Smoke (602 structures: 500 ordinary MAD-CORE + 102 references, 2 epochs, 9 mixed batches per epoch): random initialization confirmed = {j['random_init_confirmed']} ({j['network_tensors_equal_to_pretrained']} of {j['network_tensors_compared']} tensors equal to the pretrained ones), "
                     f"composition baseline fitted on the augmented smoke set ({j['composition_elements']} elements), energy scale {j['energy_scale_eV_per_atom']:.4f} eV/atom; baseline H {j['composition_baseline_eV_per_atom'].get('1', float('nan')):.3f}, Ne {j['composition_baseline_eV_per_atom'].get('10', float('nan')):.3f} eV/atom.")
    if (d / "selection_check.json").exists() and "selection_check.json: missing or unreadable" not in lines:
        j = json.load(open(d / "selection_check.json"))
        lines.append(f"Native selection recomputed on the combined 1250 validation file: logged E MAE {j['logged']['energy_mae_per_atom_meV']:.1f} vs recomputed {j['recomputed_combined']['energy_mae_per_atom_meV']:.2f} meV, F MAE {j['logged']['forces_mae_meV_per_A']:.1f} vs {j['recomputed_combined']['forces_mae_meV_per_A']:.2f} meV/A, best epoch {j['best_epoch_in_checkpoint']} = argmin {j['argmin_of_logged_mae_prod']}, passed = {j['passed']}.")
    if (d / "references_check.json").exists() and "references_check.json: missing or unreadable" not in lines:
        j = json.load(open(d / "references_check.json"))
        s = j["isolated_atom_summary_at_snapshot"]
        lines.append(f"References: max neighbours in the model's own neighbour list {j['max_neighbours_in_pet_neighbor_list']} (all zero = {j['all_zero_neighbours']}), training losses finite = {j['losses_finite']}, predictions finite = {j['predictions_finite']}, "
                     f"manual empty list equals vesin list = {j['manual_empty_list_equals_vesin_list']}; after 2 smoke epochs median |E| {s['median_abs_eV']:.2f} eV, max {s['max_abs_eV']:.2f} ({s['max_element']}), H {s['H_eV']:.2f}, Ne {s['Ne_eV']:.2f}; passed = {j['passed']}.")
    jid = (S / "markers" / "smoke.jobid").read_text().strip() if (S / "markers" / "smoke.jobid").exists() else "?"
    retry = sorted(d.parent.glob("ssv3-refcheck-*.out"))
    retry_id = retry[-1].stem.split("-")[-1] if retry else "?"
    lines += [f"Smoke job history: job {jid} trained the 2 epochs and passed the initialization and selection checks, but its first references check crashed when writing its JSON (a numpy bool), so Slurm records that job as FAILED. "
              f"The references check was rerun unchanged in content on the same snapshot in job {retry_id} and passed; its output is appended to the smoke job output and the rerun is logged in `logs/driver.log`. Production was launched on that basis."]
    return (lines or ["Smoke check pending."]) + [""]


def inventory():
    prov = json.load(open(R / "PROVENANCE.json"))["runs"]
    squeue = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j %T %M %L"], capture_output=True, text=True).stdout
    lines = ["| run | epochs planned | updates planned | epochs done | restarts | selected epoch | selected optimizer step | measured rate and ETA | node job | state |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        p = prov[run]
        rows, restarts = parse_log(R / run / "train.log")
        text = (R / run / "train.log").read_text(errors="replace") if (R / run / "train.log").exists() else ""
        done = "Training finished!" in text
        best = BEST.findall(text)
        if done and best:
            sel = int(best[-1]); selected, step = str(sel), str((sel + 1) * p["steps_per_epoch"])
        elif rows:
            i = int(np.argmin([r["val_e_mae"] * r["val_f_mae"] for r in rows])); selected, step = f"best so far {rows[i]['epoch']}", f"{(rows[i]['epoch'] + 1) * p['steps_per_epoch']} (so far)"
        else:
            selected, step = "-", "-"
        job = "; ".join(j.split()[0] + " " + j.split()[2] for j in squeue.splitlines() if f"ssv3-node_{p['node']}" in j)
        lines.append(f"| {LABEL[run]} | {p['epochs']} | {p['optimizer_updates']} | {len(rows)} | {max(restarts - 1, 0)} | {selected} | {step} | {'finished' if done else eta(rows, p['epochs'])} | {job or '-'} | {'finished' if done else ('running' if rows else 'not started')} |")
    chain = []
    for node in ("a", "b"):
        f = R / f"chain_node_{node}.log"
        if f.exists():
            chain.append(f"node {node}: " + ", ".join(l.split()[-1] for l in f.read_text().splitlines() if "Submitted" in l))
    lines += ["", "Restart history: self chained successor jobs (afterany) " + ("; ".join(chain) if chain else "none yet") + ". Each restart resumes from the newest snapshot and restores model, optimizer, scheduler and best validation state, not the RNG or sampler state.", ""]
    return lines


def table(path, title, compare=None):
    if not Path(path).exists():
        return [f"{title}: not available yet", ""]
    rows = {(r["test"], r["model"]): r for r in read_tsv(path)}
    comp = {(r["test"], r["model"]): r for r in read_tsv(compare)} if compare and Path(compare).exists() else {}
    out = [f"**{title}**" + (" (in brackets: round 2, same orderings without references, where available)" if comp else "") + "; rows marked pending are unfinished runs, evaluated incrementally as each run finishes", ""]
    for test in TESTS:
        out += [f"{test}", "", "| run | " + " | ".join(METRIC_KEYS) + " |", "|---|" + "---|" * len(METRIC_KEYS)]
        for run in RUNS:
            r = rows.get((test, run))
            c = comp.get((test, run.replace("scratchv3", "scratchv2")))
            text = (R / run / "train.log").read_text(errors="replace") if (R / run / "train.log").exists() else ""
            state = "pending, training finished, evaluation queued" if "Training finished!" in text else ("pending, training" if "Starting training" in text else "pending, not started")
            cells = []
            for k in METRIC_KEYS:
                cell = f"{float(r[k]):.1f}" if r else state
                if c:
                    cell += f" ({float(c[k]):.1f})"
                cells.append(cell)
            out.append(f"| {LABEL[run]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def subsets_section():
    p = REPORTS / "validation_subsets" / "metrics_val_subsets.tsv"
    if not p.exists():
        return ["Not available yet (inference only, computed after training by `scripts/val_subsets.py`).", ""]
    rows = read_tsv(p)
    lines = ["Recomputed on the combined validation file; energy MAE per atom in meV, component wise force MAE in meV/A; did not influence selection.", "", "| run | checkpoint | epoch | combined E / F | random 1000 E / F | MAD-CORE 250 E / F |", "|---|---|---|---|---|---|"]
    for run in RUNS:
        for kind in ("selected", "final"):
            for r in rows:
                if r["run"] == run and r["kind"] == kind:
                    lines.append(f"| {LABEL[run]} | {kind} | {r['epoch']} | {float(r['combined_E_mae_per_atom']):.1f} / {float(r['combined_F_mae']):.1f} | {float(r['random_1000_E_mae_per_atom']):.1f} / {float(r['random_1000_F_mae']):.1f} | {float(r['madcore_250_E_mae_per_atom']):.1f} / {float(r['madcore_250_F_mae']):.1f} |")
    return lines + [""]


def isolated_section():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = ["Training diagnostic, not a selection criterion: complete predicted energies of the 102 isolated atom references (target 0 eV) from the stored snapshots (`reports/isolated_atoms/<run>.tsv`, `scripts/iso_track.py`). Latest snapshot per run, plus the selected epoch where finished:", "",
             "| run | snapshot epoch | median abs error eV | max abs error eV (element) | H eV | Ne eV | elements above 1 eV |", "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    any_rows = False
    for run in RUNS:
        p = REPORTS / "isolated_atoms" / f"{run}.tsv"
        if not p.exists():
            continue
        rows = [r for r in csv.DictReader(open(p), delimiter="\t")]
        snaps = [r for r in rows if not str(r["epoch"]).startswith("final")]
        if not snaps:
            continue
        any_rows = True
        last = snaps[-1]
        lines.append(f"| {LABEL[run]} | {last['epoch']} | {float(last['median_abs_eV']):.3f} | {float(last['max_abs_eV']):.2f} ({last['max_element']}) | {float(last['H_eV']):.3f} | {float(last['Ne_eV']):.3f} | {last['n_above_1eV']} |")
        text = (R / run / "train.log").read_text(errors="replace") if (R / run / "train.log").exists() else ""
        best = BEST.findall(text)
        if best:
            sel = [r for r in rows if str(r["epoch"]) == best[-1]]
            if sel:
                r = sel[0]
                lines.append(f"| {LABEL[run]} | selected {r['epoch']} | {float(r['median_abs_eV']):.3f} | {float(r['max_abs_eV']):.2f} ({r['max_element']}) | {float(r['H_eV']):.3f} | {float(r['Ne_eV']):.3f} | {r['n_above_1eV']} |")
        x = [int(r["optimizer_updates"]) for r in snaps]
        for ax, key in zip(axes.ravel(), ("median_abs_eV", "max_abs_eV", "H_eV", "Ne_eV")):
            ax.plot(x, [abs(float(r[key])) for r in snaps], "-", lw=1.5, label=LABEL[run])
    for ax, key in zip(axes.ravel(), ("median |E| over 102 elements (eV)", "max |E| over 102 elements (eV)", "|E(H)| (eV)", "|E(Ne)| (eV)")):
        ax.set_yscale("log"); ax.set_ylabel(key); ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("optimizer updates")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("isolated atom references, predicted energy (target 0), training diagnostic", fontsize=9)
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "isolated_atoms.png", dpi=120)
    plt.close(fig)
    if not any_rows:
        return ["No snapshot tracked yet.", ""]
    lines += ["", "`reports/figures/isolated_atoms.png`: the four quantities against optimizer updates.", ""]
    return lines


def baselines_section():
    lines = ["Composition baseline (eV/atom) fitted on each augmented training set, from the first snapshot of each run (`init_check.json`); round 2 smoke value (madcore_5k without references) for comparison:", "", "| run | elements in baseline | H | Ne | energy scale eV/atom | random init confirmed |", "|---|---|---|---|---|---|"]
    v2 = V2 / "runs" / "smoke" / "scratchv2-s--madcore_5k--s42" / "init_check.json"
    if v2.exists():
        j = json.load(open(v2))
        lines.append(f"| round 2 smoke madcore_5k, no references | {j['composition_elements']} | {j['composition_baseline_eV_per_atom'].get('1', float('nan')):.3f} | {j['composition_baseline_eV_per_atom'].get('10', float('nan')):.3f} | {j['energy_scale_eV_per_atom']:.4f} | {j['random_init_confirmed']} |")
    for run in RUNS:
        p = R / run / "init_check.json"
        if p.exists():
            j = json.load(open(p))
            lines.append(f"| {LABEL[run]} | {j['composition_elements']} | {j['composition_baseline_eV_per_atom'].get('1', float('nan')):.3f} | {j['composition_baseline_eV_per_atom'].get('10', float('nan')):.3f} | {j['energy_scale_eV_per_atom']:.4f} | {j['random_init_confirmed']} |")
    return lines + [""]


def jobs_section():
    # job states come from the accounting of the cluster the report runs on; a job number from the other cluster can
    # belong to an unrelated job there, so a state is only reported when the job name belongs to this round
    host = socket.gethostname().split("-")[0]
    lines = [f"Job states read on {host}; jobs submitted on the other cluster are marked as such and must be checked there.", "", "| step | job id | state |", "|---|---|---|"]
    for name in ("smoke", "node_a", "node_b", "eval", "node_a_daint", "node_b_daint", "partial_eval"):
        f = S / "markers" / f"{name}.jobid"
        if not f.exists():
            continue
        jid = f.read_text().strip()
        out = subprocess.run(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "JobName,State"], capture_output=True, text=True).stdout.strip().split("\n")[0]
        job_name, _, state = out.partition("|")
        if job_name.startswith("ssv3"):
            lines.append(f"| {name} | {jid} | {state or 'unknown'} |")
        else:
            lines.append(f"| {name} | {jid} | not in {host} accounting (submitted on the other cluster) |")
    chain = []
    for node in ("a", "b"):
        for suffix in ("", "_daint"):
            f = R / f"chain_node_{node}{suffix}.log"
            if f.exists():
                chain += [f"node {node}{suffix.replace('_', ' ')} successor: {l.split()[-1]}" for l in f.read_text().splitlines() if "Submitted" in l]
    return lines + [""] + (["Self chained successors (afterany): " + "; ".join(chain), ""] if chain else [])

def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prov = json.load(open(R / "PROVENANCE.json"))["runs"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    any_rows = False
    for run in RUNS:
        rows, _ = parse_log(R / run / "train.log")
        if not rows:
            continue
        any_rows = True
        updates = [(r["epoch"] + 1) * prov[run]["steps_per_epoch"] for r in rows]
        axes[0].semilogy(updates, [r["val_e_mae"] for r in rows], lw=1.5, label=LABEL[run])
        axes[1].semilogy(updates, [r["val_f_mae"] for r in rows], lw=1.5)
        axes[2].semilogy(updates, [r["val_e_mae"] * r["val_f_mae"] for r in rows], lw=1.5)
    for ax, label in zip(axes, ["val E MAE per atom (meV), combined 1250", "val F MAE, component wise (meV/A), combined 1250", "mae_prod (selection metric)"]):
        ax.set_ylabel(label); ax.grid(alpha=0.25)
    if any_rows:
        axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("optimizer updates")
    fig.tight_layout()
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "validation_curves.png", dpi=120)
    plt.close(fig)
    return any_rows


def main():
    REPORTS.mkdir(exist_ok=True)
    sets = json.load(open(M / "SETS.json"))
    now = time.strftime("%F %T")
    lines = [f"# Scratch scaling with isolated atom references (round 3), status ({now})", "",
             "Follow up motivated by the isolated atom failures found in round 1 (`scratch_scaling/reports/audit/final_mae_10k/DECOMPOSITION.md`: atoms without any neighbour within 8 A received species constants of up to 290,000 eV from a MAD-CORE trained model, "
             "and no isolated H or Ne existed in that training set). Eight PET-S models from random initialization (heads included) with the corrected attention mask: MAD-CORE and random training sets of 2k, 5k, 10k and 20k ordinary structures (round 2 orderings and exclusions), "
             "each with the same 102 isolated atom references appended. Validation: the 1000 frozen random structures plus the frozen 250 MAD-CORE structures in one file, metatrain native `mae_prod` (energy MAE per atom, structure weighted, times component wise force MAE, component weighted; "
             "MAD-CORE subset 20.0 and 4.0 percent of the two terms), native cadence, selected optimizer step derived as (epoch + 1) x steps per epoch. Tests: the unchanged random (5,847) and MAD-CORE (2,000) tests. Seed 42, unweighted MSE, batch 64, lr 3e-4, 1 percent warm up, cosine, float32, default clipping, about 187,200 updates (table below). "
             "Limitations: single seed, MAD-CORE validation of 250 structures, MAD-CORE and random sets of equal structure count differ in atom count by a factor 5 to 6, and the references add 102 structures to every set. Rounds 1 and 2 continue unchanged in their own directories. Everything is uncommitted and unstaged.", "",
             f"No test result of this round can exist before training finishes; sections 4 to 7 show what is available at {now} and label interim diagnostics as such.", "",
             "## 1. Data, references and provenance", ""] + datasets_section(sets) + provenance_section() + ["## 2. Smoke check", ""] + smoke_section()
    lines += ["## 3. Run inventory", ""] + inventory() + baselines_section()
    lines += ["## 4. Results on the frozen tests (selected checkpoints, corrected PET, batch 16)", "", "Energy error per structure = |total energy error| / atoms (meV/atom); force errors pooled over all force components of the test set (meV/A), as metatrain does; MAE, RMSE, p8 = (mean e^8)^(1/8) and SUP = max over structures for energy and over components for forces. Single seed, no uncertainty bands.", ""]
    lines += table(REPORTS / "selected_checkpoints" / "metrics_mtt.tsv", "selected checkpoints", V2 / "reports" / "selected_checkpoints" / "metrics_mtt.tsv")
    lines += ["## 5. Final checkpoints (diagnostic)", ""] + table(REPORTS / "final_checkpoints" / "metrics_final_mtt.tsv", "final checkpoints (last epoch)", V2 / "reports" / "final_checkpoints" / "metrics_final_mtt.tsv")
    lines += ["## 6. Validation populations (diagnostic)", ""] + subsets_section()
    lines += ["## 7. Isolated atom references during training (diagnostic)", ""] + isolated_section()
    lines += ["## 8. Jobs", ""] + jobs_section()
    lines += ["## 9. Curves and figures", ""]
    if curves():
        lines.append("`reports/figures/validation_curves.png`: combined validation energy MAE, force MAE and mae_prod against optimizer updates.")
    for name in ("sketchmap_energy_scaling_selected", "sketchmap_force_scaling_selected", "learning_curves", "learning_curves_atoms"):
        if (REPORTS / "figures" / f"{name}.png").exists():
            lines.append(f"`reports/figures/{name}.png` (+ pdf)")
    OUT.write_text("\n".join(lines) + "\n")
    print("report written", OUT)


if __name__ == "__main__":
    main()
