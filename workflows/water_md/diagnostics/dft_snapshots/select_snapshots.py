import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
sys.path.insert(0, str(D / "scripts"))
sys.path.insert(0, str(HERE.parent / "proton_transfer"))
from water import read_trajectory  # noqa: E402
from inspect_events import h_to_o, mic, episodes_of, defect_stretches  # noqa: E402

RUNS = D / "runs" / "production"
TABLES = HERE / "tables"
SNAPSHOTS = HERE / "snapshots"

DT_PS = 0.00025
FRAME_PS = 0.0125
CONDITIONS = ("baseline", "lda_corrected", "pbe_corrected")
LDA_RUNS = [f"lda_corrected_seed{k}" for k in range(5)]

# rule fixed before any numbers were looked at
LATE = ((1, 140000), (3, 200000))
EVENT_AFTER_PS = 20.0
SEARCH_FRAMES = 8
OFFSET_FRAMES = 40
SETTLED_FRAMES = 40
BRIEF_PS = 0.1
LASTING_FRAMES = 40
SHORT_OO_A = 2.45

RULES = {
    "late": "seed 1 at step 140,000 (35.0 ps) and seed 3 at step 200,000 (50.0 ps) of every potential, chosen before any comparison",
    "event": "lasting proton transfer in an approximate LDA run after 20 ps with one crossing, largest settled stay on both sides, earliest on ties; "
             "during is the frame of smallest donor H minus acceptor H distance difference within 0.1 ps of the crossing, before and after are 0.5 ps away",
}


def load(name):
    traj, check = read_trajectory(RUNS / name)
    assert check["missing_step_count"] == 0 and check["unpaired_step_count"] == 0, (name, check)
    symbols = np.array(traj["symbols"])
    length = float(traj["cells"][0][0, 0])
    assert np.allclose(traj["cells"], traj["cells"][0]) and np.allclose(traj["cells"][0], np.diag([length] * 3))
    return traj, symbols, length


def owners(traj, symbols, length):
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    d = h_to_o(traj["positions"], h, o, length)
    return o, h, d, o[d.argmin(-1)]


def snapshot(traj, symbols, k, name, role, run, extra=None):
    atoms = Atoms(symbols=symbols, positions=traj["positions"][k], cell=traj["cells"][k], pbc=True)
    atoms.arrays["md_forces"] = traj["forces"][k].copy()
    atoms.info.update({"snapshot": name, "role": role, "run": run, "step": int(traj["steps"][k]), "time_ps": float(traj["steps"][k] * DT_PS)})
    if extra:
        atoms.info.update(extra)
    write(SNAPSHOTS / f"{name}.extxyz", atoms, format="extxyz")
    return atoms


def classify(name, traj, symbols, length):
    o, h, d, owner = owners(traj, symbols, length)
    steps = traj["steps"]
    rows = []
    for k in range(len(h)):
        for e in episodes_of(owner[:, k], d, k, traj["positions"], length, steps):
            e["hydrogen"] = int(h[k])
            rows.append(e)
    counts = np.array([np.bincount(np.searchsorted(o, row), minlength=len(o)) for row in owner])
    stretches = defect_stretches(counts, steps)
    durations = np.array([s["frames"] * FRAME_PS for s in stretches]) if stretches else np.zeros(0)

    returned = [e for e in rows if e["outcome"] == "returned"]
    transferred = [e for e in rows if e["outcome"] == "transferred"]
    lasting = [e for e in transferred if e["settled_before_frames"] >= LASTING_FRAMES and e["settled_after_frames"] >= LASTING_FRAMES]
    summary = {
        "run": name,
        "frames": int(len(steps)),
        "episodes": len(rows),
        "transient_returns_within_0p1ps": sum(1 for e in returned if e["duration_ps"] <= BRIEF_PS),
        "returns_longer_than_0p1ps": sum(1 for e in returned if e["duration_ps"] > BRIEF_PS),
        "lasting_transfers_settled_0p5ps_both_sides": len(lasting),
        "transfers_settled_less": len(transferred) - len(lasting),
        "open_at_trajectory_ends": sum(1 for e in rows if e["outcome"] == "open"),
        "hydrogens_not_on_initial_oxygen_at_end": int((owner[-1] != 3 * (h // 3)).sum()),
        "defect_stretches": len(stretches),
        "defect_frames": int((counts != 2).any(1).sum()),
        "defect_frame_fraction": float((counts != 2).any(1).mean()),
        "defect_frame_fraction_after_20ps": float((counts != 2).any(1)[steps * DT_PS >= 20.0].mean()),
        "stretches_single_frame": int((durations <= FRAME_PS + 1e-9).sum()),
        "stretches_0p1_to_0p5ps": int(((durations > BRIEF_PS) & (durations <= 0.5)).sum()),
        "stretches_0p5_to_1ps": int(((durations > 0.5) & (durations <= 1.0)).sum()),
        "stretches_over_1ps": int((durations > 1.0).sum()),
        "longest_defect_ps": float(durations.max()) if len(durations) else 0.0,
        "max_oxygens_with_other_than_two_h_in_one_frame": int((counts != 2).sum(1).max()),
        "min_transfer_oo_A": float(min(e["min_oo_A"] for e in transferred)) if transferred else None,
    }
    return summary, rows, owner, d, o, h


def choose_event(candidates):
    pool = [e for e in candidates if e["outcome"] == "transferred" and e["crossings"] == 1 and e["first_crossing_ps"] > EVENT_AFTER_PS]
    pool.sort(key=lambda e: (-min(e["settled_before_frames"], e["settled_after_frames"]), e["first_crossing_ps"]))
    return pool[0]


def main():
    summaries = []
    all_rows = []
    loaded = {}
    for name in LDA_RUNS:
        traj, symbols, length = load(name)
        summary, rows, owner, d, o, h = classify(name, traj, symbols, length)
        summaries.append(summary)
        for e in rows:
            all_rows.append({"run": name, **{k: v for k, v in e.items() if k != "oxygens_visited"}, "oxygens_visited": len(e["oxygens_visited"])})
        loaded[name] = (traj, symbols, length, owner, d, o, h)
        print(json.dumps(summary))

    columns = list(summaries[0])
    with open(TABLES / "event_classes.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for s in summaries:
            fh.write("\t".join(str(s[c]) for c in columns) + "\n")
    columns = list(all_rows[0])
    with open(TABLES / "episodes_with_settled_stays.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in all_rows:
            fh.write("\t".join(str(r[c]) for c in columns) + "\n")

    # the event
    event = choose_event(all_rows)
    traj, symbols, length, owner, d, o, h = loaded[event["run"]]
    steps = traj["steps"]
    k = int(np.where(h == event["hydrogen"])[0][0])
    donor, acceptor = event["donor"], event["acceptor"]
    crossing = int(np.searchsorted(steps, round(event["first_crossing_ps"] / DT_PS)))
    pos = traj["positions"]
    d_donor = np.linalg.norm(mic(pos[:, event["hydrogen"]] - pos[:, donor], length), axis=-1)
    d_acceptor = np.linalg.norm(mic(pos[:, event["hydrogen"]] - pos[:, acceptor], length), axis=-1)
    d_oo = np.linalg.norm(mic(pos[:, acceptor] - pos[:, donor], length), axis=-1)
    window = np.arange(crossing - SEARCH_FRAMES, crossing + SEARCH_FRAMES + 1)
    during = int(window[np.abs(d_donor[window] - d_acceptor[window]).argmin()])
    frames = {"before": during - OFFSET_FRAMES, "during": during, "after": during + OFFSET_FRAMES}

    record = {
        **event,
        "frame_of_crossing": crossing,
        "frames": frames,
        "times_ps": {r: float(steps[f] * DT_PS) for r, f in frames.items()},
        "donor_h_A": {r: float(d_donor[f]) for r, f in frames.items()},
        "acceptor_h_A": {r: float(d_acceptor[f]) for r, f in frames.items()},
        "donor_acceptor_oo_A": {r: float(d_oo[f]) for r, f in frames.items()},
        "owner_at_frames": {r: int(owner[f, k]) for r, f in frames.items()},
        "owner_constant_before_window": bool((owner[frames["before"]: during, k] == donor).all()),
        "owner_constant_after_window": bool((owner[during + 1: frames["after"] + 1, k] == acceptor).all()),
        "during_oo_short": bool(d_oo[during] < SHORT_OO_A),
        "oxygens_with_other_than_two_h": {r: int((np.bincount(np.searchsorted(o, owner[f]), minlength=len(o)) != 2).sum()) for r, f in frames.items()},
    }
    span = np.arange(max(crossing - 160, 0), min(crossing + 161, len(steps)))
    with open(TABLES / "event_series.tsv", "w") as fh:
        fh.write("frame\tstep\ttime_ps\tdonor_h_A\tacceptor_h_A\tdonor_acceptor_oo_A\towner\n")
        for f in span:
            fh.write(f"{f}\t{steps[f]}\t{steps[f] * DT_PS:.4f}\t{d_donor[f]:.4f}\t{d_acceptor[f]:.4f}\t{d_oo[f]:.4f}\t{owner[f, k]}\n")

    # the nine snapshots
    rows = []
    for condition in CONDITIONS:
        for seed, step in LATE:
            name = f"{condition}_seed{seed}"
            traj_c, symbols_c, length_c = load(name)
            j = int(np.where(traj_c["steps"] == step)[0][0])
            o_c, h_c, d_c, owner_c = owners(traj_c, symbols_c, length_c)
            counts = np.bincount(np.searchsorted(o_c, owner_c[j]), minlength=len(o_c))
            label = f"late_{condition}_seed{seed}_step{step}"
            snapshot(traj_c, symbols_c, j, label, "late", name)
            rows.append({"snapshot": label, "role": "late", "run": name, "step": step, "time_ps": step * DT_PS, "min_oo_A": float(np.min([np.linalg.norm(mic(traj_c["positions"][j, a] - traj_c["positions"][j, b], length_c)) for i, a in enumerate(o_c) for b in o_c[i + 1:]])),
                         "oxygens_with_other_than_two_h": int((counts != 2).sum()), "max_md_force_eV_A": float(np.linalg.norm(traj_c["forces"][j], axis=1).max())})
    for role, f in frames.items():
        label = f"event_{role}"
        extra = {"hydrogen": event["hydrogen"], "donor": donor, "acceptor": acceptor}
        snapshot(traj, symbols, f, label, role, event["run"], extra)
        rows.append({"snapshot": label, "role": role, "run": event["run"], "step": int(steps[f]), "time_ps": float(steps[f] * DT_PS), "min_oo_A": float(d_oo[f]),
                     "oxygens_with_other_than_two_h": record["oxygens_with_other_than_two_h"][role], "max_md_force_eV_A": float(np.linalg.norm(traj["forces"][f], axis=1).max())})
    columns = list(rows[0])
    with open(TABLES / "snapshots.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in columns) + "\n")
    (TABLES / "event.json").write_text(json.dumps({"rules": RULES, "event": record}, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
