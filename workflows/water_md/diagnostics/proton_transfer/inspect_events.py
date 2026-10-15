import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
sys.path.insert(0, str(D / "scripts"))
from observables import frame_analysis  # noqa: E402
from water import CONDITIONS, SEEDS, read_trajectory  # noqa: E402


RUNS = D / "runs" / "production"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
SNAPSHOTS = HERE / "snapshots"

DT_PS = 0.00025
FRAME_PS = 0.0125
SETTLED_FRAMES = 40
BRIEF_PS = 0.1
CLUSTER_GAP_PS = 0.5
SHUTTLING_CROSSINGS = 100
NO_OXYGEN_A = 1.3
SNAPSHOT_SHELL_A = 3.5
HBOND_A = 2.0
SAMPLE_FRAMES = 200
SAMPLE_SEED = 20261011

SELECTION_RULES = {
    "a_longest_defect": "the longest stretch of frames with a coordination defect, over all runs",
    "b_persistent": "a sustained transfer settled at least 2 ps before and after, fewest crossings, earliest",
    "c_brief": "the first isolated two-crossing return within 0.05 ps after 5 ps, settled at least 1 ps before and after",
}


def step_of(atoms):
    return int(re.search(r"Step:\s*(\d+)", atoms.info["ipi_comment"]).group(1))


# positions read with ase, every occurrence of a step kept so that restart duplicates can be compared
def read_positions(run):
    occurrences = {}
    symbols_ok = True
    reference = None

    for atoms in read(run / "sim.pos_0.extxyz", index=":"):
        symbols = atoms.get_chemical_symbols()
        reference = reference or symbols
        symbols_ok = symbols_ok and symbols == reference
        occurrences.setdefault(step_of(atoms), []).append(atoms.positions.copy())

    steps = np.array(sorted(occurrences))
    positions = np.array([occurrences[s][-1] for s in steps])
    duplicate_gaps = [float(np.abs(o[0] - o[-1]).max()) for o in occurrences.values() if len(o) > 1]
    cell = atoms.cell.array.copy()

    return steps, positions, np.array(reference), cell, {
        "symbols_identical_in_every_frame": symbols_ok,
        "symbols_order_o_h_h": all(reference[3 * k: 3 * k + 3] == ["O", "H", "H"] for k in range(len(reference) // 3)),
        "unique_steps": len(steps),
        "restart_duplicates": int(sum(len(o) - 1 for o in occurrences.values())),
        "duplicate_max_position_difference_A": max(duplicate_gaps) if duplicate_gaps else 0.0,
        "steps_evenly_spaced_by_50": bool((np.diff(steps) == 50).all()),
    }


# minimum-image H to O distances for an orthogonal cell, written independently of the production code
def h_to_o(positions, h, o, length):
    d = positions[..., h, None, :] - positions[..., None, o, :]
    d -= length * np.round(d / length)
    return np.linalg.norm(d, axis=-1)


# the minimum over all 27 neighbouring images of the wrapped positions, a brute-force check of the minimum image
def h_to_o_bruteforce(frame, h, o, cell):
    length = cell[0, 0]
    wrapped = frame - length * np.floor(frame / length)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ cell
    d = wrapped[h, None, None, :] - wrapped[None, o, None, :] + shifts[None, None, :, :]
    return np.linalg.norm(d, axis=-1).min(-1)


def mic(vector, length):
    return vector - length * np.round(vector / length)


def restart_steps(run):
    text = (run / "segments.log").read_text()
    return sorted(int(s) for s in re.findall(r"start from resume\.xml at step (\d+)", text))


def runs_of(sequence):
    result, start = [], 0

    for i in range(1, len(sequence) + 1):
        if i == len(sequence) or sequence[i] != sequence[start]:
            result.append((int(sequence[start]), start, i - start))
            start = i

    return result


# episodes between settled stays of one H, repeated crossings inside an episode are not separate events
def episodes_of(owner, distances, k, positions, length, steps):
    stays = runs_of(owner)
    settled = [i for i, (_, _, n) in enumerate(stays) if n >= SETTLED_FRAMES]
    anchors = [None] + settled + [None]
    found = []

    for left, right in zip(anchors[:-1], anchors[1:]):
        start = 0 if left is None else stays[left][1] + stays[left][2] - 1
        end = len(owner) - 1 if right is None else stays[right][1]
        changes = np.where(owner[start + 1:end + 1] != owner[start:end])[0] + start + 1
        if len(changes) == 0:
            continue

        before = None if left is None else stays[left][0]
        after = None if right is None else stays[right][0]
        donor = before if before is not None else int(owner[start])

        # the acceptor is the settled O after the episode, or the other O the H spent most frames on
        others = owner[start:end + 1][owner[start:end + 1] != donor]
        if after is not None and after != donor:
            acceptor = after
        else:
            values, frames_on = np.unique(others, return_counts=True)
            acceptor = int(values[frames_on.argmax()])

        if before is None or after is None:
            outcome = "open"
        elif before == after:
            outcome = "returned"
        else:
            outcome = "transferred"

        span = np.arange(max(changes[0] - 1, 0), changes[-1] + 1)
        ranked = np.sort(distances[span, k, :], axis=1)
        oo = np.linalg.norm(mic(positions[span, acceptor] - positions[span, donor], length), axis=-1)

        found.append({
            "donor": int(donor),
            "acceptor": int(acceptor),
            "oxygens_visited": sorted({int(x) for x in owner[start:end + 1]}),
            "outcome": outcome,
            "first_crossing_ps": float(steps[changes[0]] * DT_PS),
            "last_crossing_ps": float(steps[changes[-1]] * DT_PS),
            "duration_ps": float((changes[-1] - changes[0]) * FRAME_PS),
            "crossings": int(len(changes)),
            "settled_before_frames": int(stays[left][2]) if left is not None else 0,
            "settled_after_frames": int(stays[right][2]) if right is not None else 0,
            "min_oo_A": float(oo.min()),
            "min_nearest_minus_second_A": float((ranked[:, 1] - ranked[:, 0]).min()),
            "frames_without_oxygen_within_1p3A": int((ranked[:, 0] > NO_OXYGEN_A).sum()),
        })

    return found


def defect_stretches(owner_counts, steps):
    defect = (owner_counts != 2).any(1)
    stretches, start = [], None

    for i, flag in enumerate(list(defect) + [False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            stretches.append((start, i - 1))
            start = None

    return [{"first_frame": a, "last_frame": b, "start_ps": float(steps[a] * DT_PS), "end_ps": float(steps[b] * DT_PS),
             "frames": b - a + 1} for a, b in stretches]


def inspect_run(name):
    run = RUNS / name
    steps, positions, symbols, cell, checks = read_positions(run)
    length = float(cell[0, 0])
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]

    checks["orthogonal_cubic_cell"] = bool(np.allclose(cell, np.diag(np.diag(cell))) and np.allclose(np.diag(cell), length))

    # nearest O per H, independent route
    d = h_to_o(positions, h, o, length)
    owner = o[d.argmin(-1)]

    # the production route on its own reader and its own minimum image
    traj, _ = read_trajectory(run)
    production = np.array([frame_analysis(p, c, f, np.array(traj["symbols"]))["owner"] for p, c, f in zip(traj["positions"], traj["cells"], traj["forces"])])
    checks["same_steps_as_production_reader"] = bool(np.array_equal(traj["steps"], steps))
    checks["positions_equal_production_reader_max_A"] = float(np.abs(traj["positions"] - positions).max())
    checks["owners_equal_production_in_every_frame"] = bool(np.array_equal(production, owner))

    # brute force over 27 images on sampled frames
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = rng.choice(len(steps), SAMPLE_FRAMES, replace=False)
    brute_gap = max(float(np.abs(h_to_o_bruteforce(positions[k], h, o, cell) - d[k]).max()) for k in sample)
    checks["bruteforce_27_images_max_difference_A"] = brute_gap
    checks["largest_nearest_oh_A"] = float(d.min(-1).max())
    checks["half_cell_A"] = length / 2.0

    # the same assignment with every atom wrapped into the cell
    wrapped = positions - length * np.floor(positions / length)
    checks["fraction_of_coordinates_outside_cell"] = float(((positions < 0) | (positions >= length)).mean())
    checks["owners_equal_after_wrapping"] = bool(np.array_equal(o[h_to_o(wrapped, h, o, length).argmin(-1)], owner))

    # continuity between consecutive frames, overall and across restart boundaries
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=-1).max(-1)
    restarts = restart_steps(run)
    restart_frames = [int(np.searchsorted(steps, s)) for s in restarts if s in set(steps.tolist())]
    near_restart = sorted({k for r in restart_frames for k in (r - 1, r) if 0 <= k < len(jumps)})
    checks["max_atomic_jump_between_frames_A"] = float(jumps.max())
    checks["max_atomic_jump_at_restarts_A"] = float(jumps[near_restart].max()) if near_restart else None
    checks["restart_steps"] = restarts

    # crossings next to a restart against the rate elsewhere
    crossing_intervals = (owner[1:] != owner[:-1]).sum(1)
    checks["crossings_total"] = int(crossing_intervals.sum())
    checks["crossings_in_intervals_touching_restarts"] = int(crossing_intervals[near_restart].sum()) if near_restart else 0
    checks["crossings_expected_there_at_average_rate"] = float(crossing_intervals.mean() * len(near_restart))

    counts = np.array([np.bincount(np.searchsorted(o, row), minlength=len(o)) for row in owner])
    events = []
    for k in range(len(h)):
        for e in episodes_of(owner[:, k], d, k, positions, length, steps):
            events.append({"run": name, "hydrogen": int(h[k]), **e})

    # sustained transfers close in time, linked donor to acceptor, a closed loop is tested for wrapping the cell
    transfers = sorted((e for e in events if e["outcome"] == "transferred"), key=lambda e: e["first_crossing_ps"])
    clusters, current = [], []
    for e in transfers:
        if current and e["first_crossing_ps"] - current[-1]["first_crossing_ps"] > CLUSTER_GAP_PS:
            clusters.append(current)
            current = []
        current.append(e)
    if current:
        clusters.append(current)

    loops = []
    for group in clusters:
        edges = {e["donor"]: e["acceptor"] for e in group}
        for start in edges:
            path = [start]
            while path[-1] in edges and edges[path[-1]] not in path:
                path.append(edges[path[-1]])
            if path[-1] in edges and edges[path[-1]] == start and start == min(path):
                mid = int(np.searchsorted(steps, np.mean([e["first_crossing_ps"] for e in group]) / DT_PS))
                vector = sum(mic(positions[mid, b] - positions[mid, a], length) for a, b in zip(path, path[1:] + [start]))
                loops.append({"oxygens": path, "time_ps": float(steps[mid] * DT_PS), "wrap_vector_A": np.round(vector, 3).tolist(),
                              "wraps_cell": bool(np.abs(vector).max() > 1.0)})

    shuttling = []
    for k in range(len(h)):
        changes = np.where(owner[1:, k] != owner[:-1, k])[0] + 1
        if len(changes) < SHUTTLING_CROSSINGS:
            continue
        span = np.arange(changes[0], changes[-1] + 1)
        pair = np.unique(owner[span, k], return_counts=True)
        a, b = (int(x) for x in pair[0][np.argsort(-pair[1])][:2])
        da = np.linalg.norm(mic(positions[span, h[k]] - positions[span, a], length), axis=-1)
        db = np.linalg.norm(mic(positions[span, h[k]] - positions[span, b], length), axis=-1)
        oo = np.linalg.norm(mic(positions[span, b] - positions[span, a], length), axis=-1)
        shuttling.append({"run": name, "hydrogen": int(h[k]), "oxygen_a": a, "oxygen_b": b, "crossings": int(len(changes)),
                          "span_start_ps": float(steps[changes[0]] * DT_PS), "span_end_ps": float(steps[changes[-1]] * DT_PS),
                          "fraction_nearer_a": float((da < db).mean()), "median_abs_difference_A": float(np.median(np.abs(da - db))),
                          "mean_oo_A": float(oo.mean())})

    reassigned_at_end = int((owner[-1] != owner[0]).sum())
    reversed_transfers = 0
    for k in range(len(h)):
        mine = [e for e in transfers if e["hydrogen"] == int(h[k])]
        reversed_transfers += sum(1 for a, b in zip(mine, mine[1:]) if b["donor"] == a["acceptor"] and b["acceptor"] == a["donor"])

    return {
        "run": name,
        "checks": checks,
        "events": events,
        "shuttling": shuttling,
        "transfer_clusters": len(clusters),
        "transfer_cluster_sizes": [len(c) for c in clusters],
        "closed_loops": loops,
        "hydrogens_reassigned_at_end": reassigned_at_end,
        "transfers_reversed_later": reversed_transfers,
        "defects": defect_stretches(counts, steps),
        "frames_with_hydrogen_beyond_1p3A_of_every_oxygen": int((d.min(-1) > NO_OXYGEN_A).any(-1).sum()),
    }


# atoms near an O pair, each placed at its minimum image next to the donor so that molecules are whole
def local_cluster(frame, symbols, length, donor, acceptor, extra=()):
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    centre = frame[donor]
    o_rel = mic(frame[o] - centre, length)
    acc_rel = mic(frame[acceptor] - centre, length)
    keep_o = [i for i, r in zip(o, o_rel) if np.linalg.norm(r) < SNAPSHOT_SHELL_A or np.linalg.norm(mic(r - acc_rel, length)) < SNAPSHOT_SHELL_A or i in extra]

    placed = {i: centre + mic(frame[i] - centre, length) for i in keep_o}
    for k in h:
        d = np.linalg.norm(mic(frame[k] - frame[o], length), axis=-1)
        owner = o[d.argmin()]
        if owner in placed:
            placed[k] = placed[owner] + mic(frame[k] - frame[owner], length)

    return placed


def draw_snapshot(ax, placed, symbols, donor, acceptor, hydrogen, length, title):
    origin = placed[donor]
    x_axis = placed[acceptor] - origin
    x_axis /= np.linalg.norm(x_axis)
    y_axis = placed[hydrogen] - origin - ((placed[hydrogen] - origin) @ x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis) if np.linalg.norm(y_axis) > 1e-6 else np.cross(x_axis, [0.0, 0.0, 1.0])
    xy = {i: np.array([(p - origin) @ x_axis, (p - origin) @ y_axis]) for i, p in placed.items()}

    oxygens = [i for i in placed if symbols[i] == "O"]
    hydrogens = [i for i in placed if symbols[i] == "H"]

    for k in hydrogens:
        d = {i: np.linalg.norm(placed[k] - placed[i]) for i in oxygens}
        ranked = sorted(d, key=d.get)
        ax.plot(*zip(xy[k], xy[ranked[0]]), color="0.4", lw=1.2, zorder=1)
        if len(ranked) > 1 and d[ranked[1]] < HBOND_A:
            ax.plot(*zip(xy[k], xy[ranked[1]]), color="0.6", lw=0.8, ls="--", zorder=1)

    for i in oxygens:
        edge = "black" if i in (donor, acceptor) else "none"
        ax.scatter(*xy[i], s=260, color="C3", edgecolors=edge, linewidths=1.5, zorder=2)
    for k in hydrogens:
        ax.scatter(*xy[k], s=90, color="C1" if k == hydrogen else "0.85", edgecolors="black", linewidths=0.6, zorder=3)

    ax.annotate(f"O{donor}", xy[donor], textcoords="offset points", xytext=(-14, -18), fontsize=8)
    ax.annotate(f"O{acceptor}", xy[acceptor], textcoords="offset points", xytext=(4, -18), fontsize=8)
    ax.annotate(f"H{hydrogen}", xy[hydrogen], textcoords="offset points", xytext=(4, 6), fontsize=8, color="C1")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# every O wrapped into the cell and its nearest H placed next to it, a full frame with whole molecules
def whole_frame(frame, symbols, length):
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    out = frame.copy()
    out[o] = frame[o] - length * np.floor(frame[o] / length)

    for k in h:
        d = np.linalg.norm(mic(frame[k] - frame[o], length), axis=-1)
        owner = o[d.argmin()]
        out[k] = out[owner] + mic(frame[k] - frame[owner], length)

    return Atoms(symbols=list(symbols), positions=out, cell=[length] * 3, pbc=True)


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def example(name, spec, results):
    run = RUNS / spec["run"]
    steps, positions, symbols, cell, _ = read_positions(run)
    length = float(cell[0, 0])
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    time_ps = steps * DT_PS
    owner = o[h_to_o(positions, h, o, length).argmin(-1)]
    counts = np.array([np.bincount(np.searchsorted(o, row), minlength=len(o)) for row in owner])
    restarts = [s * DT_PS for s in restart_steps(run)]
    record = {"run": spec["run"]}

    # the H to follow, its donor and acceptor, and the time window
    if spec["kind"] == "longest_defect":
        stretch = max(results[spec["run"]]["defects"], key=lambda s: s["frames"])
        first, last = stretch["first_frame"], stretch["last_frame"]
        movers = np.where(owner[first] != owner[first - 1])[0]
        k = int(movers[0])
        hydrogen = int(h[k])
        donor, acceptor = int(owner[first - 1, k]), int(owner[first, k])
        lo, hi = max(first - 80, 0), min(last + 80, len(steps) - 1)
        event_frames = np.arange(first, last + 1)
        record["stretch"] = stretch

        hops = []
        for i in range(first, last + 2):
            for j in np.where(owner[i] != owner[i - 1])[0]:
                hops.append({"time_ps": float(time_ps[i]), "hydrogen": int(h[j]), "from": int(owner[i - 1, j]), "to": int(owner[i, j])})
        record["hops_in_stretch"] = hops
        defect_frames = (counts[first:last + 1] != 2).sum(0)
        ranked = [int(o[q]) for q in np.argsort(-defect_frames) if defect_frames[q] > 0]
        defect_sites = [donor, acceptor] + [s for s in ranked if s not in (donor, acceptor)]
        record["defect_frames_per_oxygen"] = {int(o[q]): int(defect_frames[q]) for q in np.argsort(-defect_frames) if defect_frames[q] > 0}
    else:
        hydrogen = spec["h"]
        k = int(np.searchsorted(h, hydrogen))
        changes = np.where(owner[1:, k] != owner[:-1, k])[0] + 1
        if "around_ps" in spec:
            changes = changes[np.abs(time_ps[changes] - spec["around_ps"]) <= 1.0]
        first, last = int(changes[0]), int(changes[-1])
        donor, acceptor = int(owner[first - 1, k]), int(owner[first, k])
        window = 160 if len(changes) == 1 else 80
        lo, hi = max(first - window, 0), min(last + window, len(steps) - 1)
        event_frames = np.arange(max(first - 8, 0), min(last + 9, len(steps)))
        defect_sites = [donor, acceptor]
        record["crossing_times_ps"] = [float(time_ps[c]) for c in changes]

    frames = np.arange(lo, hi + 1)
    d_hd = np.linalg.norm(mic(positions[frames, hydrogen] - positions[frames, donor], length), axis=-1)
    d_ha = np.linalg.norm(mic(positions[frames, hydrogen] - positions[frames, acceptor], length), axis=-1)
    d_oo = np.linalg.norm(mic(positions[frames, acceptor] - positions[frames, donor], length), axis=-1)
    on_acceptor = owner[frames, k] == acceptor
    t = time_ps[frames]

    record.update(hydrogen=hydrogen, donor=donor, acceptor=acceptor, window_ps=[float(t[0]), float(t[-1])],
                  returns_to_donor=bool(owner[-1, k] == donor), final_owner=int(owner[-1, k]),
                  frames_on_acceptor_in_window=int(on_acceptor.sum()), min_oo_in_window_A=float(d_oo.min()),
                  min_abs_asymmetry_in_window_A=float(np.abs(d_hd - d_ha).min()))

    # distances and coordination against time
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, facecolor="white", gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t, d_hd, color="C0", lw=1.0, label=f"H{hydrogen}–O{donor} (donor)")
    axes[0].plot(t, d_ha, color="C1", lw=1.0, label=f"H{hydrogen}–O{acceptor} (acceptor)")
    axes[0].plot(t, d_oo, color="C2", lw=1.0, label=f"O{donor}–O{acceptor}")
    axes[0].fill_between(t, 0, 3.2, where=on_acceptor, color="C1", alpha=0.12, lw=0, label="nearest O is the acceptor")
    axes[0].set_ylim(0.8, 3.2)
    axes[0].set_ylabel("distance (Å)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    for site, color in zip(defect_sites[:6], ("C0", "C1", "C2", "C4", "C5", "C6")):
        axes[1].step(t, counts[frames, int(np.searchsorted(o, site))], where="mid", color=color, lw=1.0, label=f"O{site}")
    axes[1].set_ylabel("nearest H")
    axes[1].set_yticks([0, 1, 2, 3, 4])
    axes[1].set_xlabel("time (ps)")
    axes[1].legend(frameon=False, fontsize=7, ncol=6, loc="upper right")

    for ax in axes:
        for r in restarts:
            if t[0] <= r <= t[-1]:
                ax.axvline(r, color="0.5", ls=":", lw=0.8)

    fig.suptitle(f"{spec['run']}, H{hydrogen}", fontsize=10)
    fig.tight_layout()
    save(fig, f"{name}_distances")

    # snapshots before, at the most symmetric frame, and after
    inside = event_frames[(event_frames >= lo) & (event_frames <= hi)]
    sym_d = np.abs(np.linalg.norm(mic(positions[inside, hydrogen] - positions[inside, donor], length), axis=-1)
                   - np.linalg.norm(mic(positions[inside, hydrogen] - positions[inside, acceptor], length), axis=-1))
    middle = int(inside[sym_d.argmin()])
    offset = 40 if spec["kind"] != "hydrogen" or len(record.get("crossing_times_ps", [])) == 1 else 20
    chosen = {"before": max(first - offset, 0), "during": middle, "after": min(max(last, first) + offset, len(steps) - 1)}
    record["snapshot_times_ps"] = {key: float(time_ps[i]) for key, i in chosen.items()}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor="white")
    for ax, (key, i) in zip(axes, chosen.items()):
        placed = local_cluster(positions[i], symbols, length, donor, acceptor)
        draw_snapshot(ax, placed, symbols, donor, acceptor, hydrogen, length, f"{key}, t = {time_ps[i]:.3f} ps")
        indices = sorted(placed)
        write(SNAPSHOTS / f"{name}_{key}_cluster.extxyz", Atoms(symbols=[symbols[j] for j in indices], positions=[placed[j] for j in indices]))
        write(SNAPSHOTS / f"{name}_{key}_whole_cell.extxyz", whole_frame(positions[i], symbols, length))
    fig.suptitle(f"{spec['run']}: H{hydrogen} (orange) between O{donor} and O{acceptor}, projected on their plane", fontsize=10)
    fig.tight_layout()
    save(fig, f"{name}_snapshots")

    return record


def write_tsv(path, rows, columns):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in columns) + "\n")


def main():
    for folder in (TABLES, FIGURES, SNAPSHOTS):
        folder.mkdir(parents=True, exist_ok=True)

    names = [f"{c}_seed{s}" for c in CONDITIONS for s in SEEDS]
    with Pool(len(names)) as pool:
        results = dict(zip(names, pool.map(inspect_run, names)))

    (TABLES / "verification.json").write_text(json.dumps({n: r["checks"] for n, r in results.items()}, indent=1) + "\n")

    events = [e for r in results.values() for e in r["events"]]
    columns = ["run", "hydrogen", "donor", "acceptor", "outcome", "first_crossing_ps", "last_crossing_ps", "duration_ps", "crossings",
               "settled_after_frames", "min_oo_A", "min_nearest_minus_second_A", "frames_without_oxygen_within_1p3A"]
    write_tsv(TABLES / "episodes.tsv", events, columns)

    summary = []
    for name, r in results.items():
        ev = r["events"]
        summary.append({
            "run": name,
            "crossings": r["checks"]["crossings_total"],
            "episodes": len(ev),
            "returned": sum(e["outcome"] == "returned" for e in ev),
            "returned_within_0p1ps": sum(e["outcome"] == "returned" and e["duration_ps"] <= BRIEF_PS for e in ev),
            "transferred": sum(e["outcome"] == "transferred" for e in ev),
            "open_at_trajectory_end": sum(e["outcome"] == "open" for e in ev),
            "hydrogens_with_episodes": len({e["hydrogen"] for e in ev}),
            "defect_stretches": len(r["defects"]),
            "longest_defect_ps": max((s["frames"] for s in r["defects"]), default=0) * FRAME_PS,
            "frames_with_h_beyond_1p3A": r["frames_with_hydrogen_beyond_1p3A_of_every_oxygen"],
            "transfers_reversed_later": r["transfers_reversed_later"],
            "hydrogens_reassigned_at_end": r["hydrogens_reassigned_at_end"],
            "transfer_clusters": r["transfer_clusters"],
            "closed_loops": len(r["closed_loops"]),
            "closed_loops_wrapping_cell": sum(lp["wraps_cell"] for lp in r["closed_loops"]),
        })
    write_tsv(TABLES / "episode_summary.tsv", summary, list(summary[0]))

    (TABLES / "transfer_loops.json").write_text(json.dumps({n: {"cluster_sizes": r["transfer_cluster_sizes"], "closed_loops": r["closed_loops"]}
                                                           for n, r in results.items() if r["events"]}, indent=1) + "\n")

    shuttling = [s for r in results.values() for s in r["shuttling"]]
    if shuttling:
        write_tsv(TABLES / "shuttling_hydrogens.tsv", shuttling, list(shuttling[0]))

    defects = [{"run": n, **s} for n, r in results.items() for s in r["defects"]]
    write_tsv(TABLES / "defect_stretches.tsv", defects, ["run", "start_ps", "end_ps", "frames"])

    # examples chosen by rule, a category without a candidate is skipped
    examples = {}
    stretches = [(s["frames"], n) for n, r in results.items() for s in r["defects"]]
    if stretches:
        examples["a_longest_defect"] = {"run": max(stretches)[1], "kind": "longest_defect"}

    persistent = [e for e in events if e["outcome"] == "transferred" and e["settled_before_frames"] >= 160 and e["settled_after_frames"] >= 160]
    if persistent:
        chosen = sorted(persistent, key=lambda e: (e["crossings"], e["first_crossing_ps"]))[0]
        examples["b_persistent"] = {"run": chosen["run"], "kind": "hydrogen", "h": chosen["hydrogen"], "around_ps": chosen["first_crossing_ps"]}

    brief = [e for e in events if e["outcome"] == "returned" and e["crossings"] == 2 and e["duration_ps"] <= 0.05
             and e["settled_before_frames"] >= 80 and e["settled_after_frames"] >= 80 and e["first_crossing_ps"] >= 5.0]
    if brief:
        chosen = sorted(brief, key=lambda e: e["first_crossing_ps"])[0]
        examples["c_brief"] = {"run": chosen["run"], "kind": "hydrogen", "h": chosen["hydrogen"], "around_ps": chosen["first_crossing_ps"]}

    records = {name: example(name, spec, results) for name, spec in examples.items()}
    for name in records:
        records[name]["selection"] = SELECTION_RULES[name]
    records["skipped_categories"] = [name for name in SELECTION_RULES if name not in examples]
    (TABLES / "examples.json").write_text(json.dumps(records, indent=1) + "\n")

    for row in summary:
        print(json.dumps(row))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "hops_in_stretch"} if isinstance(v, dict) else v) for k, v in records.items()}, indent=1))


if __name__ == "__main__":
    main()
