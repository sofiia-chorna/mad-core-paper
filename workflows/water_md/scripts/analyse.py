import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase.io import read
from ase.units import _amu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_runs  # noqa: E402
from observables import (  # noqa: E402
    EVENTS,
    INTEGRATION_WARNING_MEV_PER_ATOM_PER_PS,
    NEAREST_OH_EDGES,
    RDF_BIN,
    RDF_CENTRES,
    RDF_RMAX,
    SECOND_OH_EDGES,
    THRESHOLDS,
    coordination,
    density,
    frame_analysis,
    kinetic_temperatures,
    oo_shells,
    rdf,
)
from water import CONDITIONS, D, LABELS, NATOMS, NMOLECULES, SEEDS, STRIDE, read_properties, read_trajectory  # noqa: E402


OUT = D / "analysis"
PHASE = "production"
TOTAL_STEPS = 200000
DT_PS = 0.00025
FRAME_PS = DT_PS * STRIDE
WINDOW_START_PS = 5.0

LATE_WINDOW_START_PS = 20.0
DISCARDED = (0.0, 5.0)
BLOCKS = ((5.0, 20.0), (20.0, 35.0), (35.0, 50.0))
TEMPERATURE_BIN_PS = 1.0
CONVERGENCE_RMIN_A = 1.3
COMMON_OO_RADIUS_A = 3.3
PERSISTENT_FRAMES = 40
PAIRS = ("oo", "oh", "hh")
PAIR_LABELS = {"oo": "O-O", "oh": "O-H", "hh": "H-H"}

SHELL_COLUMNS = [
    "first_peak_r_A",
    "first_peak_g",
    "first_min_r_A",
    "first_min_g",
    "second_peak_r_A",
    "second_peak_g",
    "peak_to_min_ratio",
    "n_oo_common_radius",
    "n_oo_own_first_min",
]
BLOCK_EXTRAS = [
    "potential_mean_meV_per_molecule",
    "temperature_mean_K",
    "temperature_H_mean_K",
    "temperature_O_mean_K",
    "temperature_H_minus_O_K",
    "nearest_oh_mean_A",
]
RUN_MEASURES = [
    "temperature_mean_K",
    "temperature_std_K",
    "temperature_H_mean_K",
    "temperature_O_mean_K",
    "ipi_temperature_mean_K",
    "conserved_drift_meV_per_atom_per_ps",
    "max_atomic_force_eV_A",
    "min_pair_distance_A",
    "potential_mean_meV_per_molecule",
    "correction_energy_mean_meV_per_molecule",
    "pressure_mean_bar",
    "nearest_oh_mean_A",
    "hoh_angle_mean_deg",
    "frames_with_coordination_change",
    "frames_with_coordination_change_after_5ps",
    "longest_coordination_change_ps",
    "hydrogen_owner_changes_between_frames",
    "hydrogens_on_other_oxygen_at_end",
    "hydrogens_persistently_on_other_oxygen",
    "frames_with_hh_below_1A",
    "frames_with_oo_below_2A",
    "min_hh_A",
    "min_oo_A",
]


def block_key(lo, hi):
    return f"block_{lo:g}_{hi:g}"


def longest_true_run(flags):
    best, current = 0, 0

    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)

    return best


def block_mask(time_ps, analysed, lo, hi):
    upper = time_ps <= hi if hi == BLOCKS[-1][1] else time_ps < hi
    return analysed & (time_ps >= lo) & upper


def structure(frames, mask, volume):
    selected = [f for f, keep in zip(frames, mask) if keep]
    if not selected:
        return None

    n = len(selected)
    pairs = selected[0]["pairs"]

    curves = {p: rdf(np.sum([f[f"hist_{p}"] for f in selected], axis=0), n, pairs[p], volume) for p in PAIRS}
    curves["nearest_oh"] = density(np.sum([f["hist_nearest_oh"] for f in selected], axis=0), NEAREST_OH_EDGES)
    curves["second_oh"] = density(np.sum([f["hist_second_oh"] for f in selected], axis=0), SECOND_OH_EDGES)

    shells = oo_shells(curves["oo"])
    shells["n_oo_common_radius"] = coordination(curves["oo"], NMOLECULES, volume, COMMON_OO_RADIUS_A)
    shells["n_oo_own_first_min"] = coordination(curves["oo"], NMOLECULES, volume, shells["first_min_r_A"])
    shells["frames"] = n

    return curves, shells


# per-run frame bookkeeping from the completion check
def bookkeeping(completion):
    fc = completion["frame_check"]
    files = ("positions", "velocities", "forces")

    return {
        "paired_frames": fc["paired_frames"],
        "restart_duplicates": sum(fc[k]["occurrences"] - fc[k]["unique_steps"] for k in files),
        "damaged_frames": sum(fc[k]["damaged_frames"] for k in files),
        "damaged_property_rows": completion["property_check"]["damaged_property_rows"],
        "missing_steps": fc["missing_step_count"],
        "unpaired_steps": fc["unpaired_step_count"],
        "segments": completion["segments"],
    }


def threshold_crossings(frames, temperatures, time_ps, end):
    measures = {
        "min_pair_distance": (np.array([f["min_pair_distance_A"] for f in frames[:end]]), "below", THRESHOLDS["min_pair_distance_A"]),
        "max_atomic_force": (np.array([f["max_atomic_force_eV_A"] for f in frames[:end]]), "above", THRESHOLDS["max_atomic_force_eV_A"]),
        "kinetic_temperature": (temperatures["all"][:end], "above", THRESHOLDS["kinetic_temperature_K"]),
    }

    crossings = {}
    for name, (values, side, limit) in measures.items():
        crossed = values < limit if side == "below" else values > limit
        if crossed.any():
            crossings[name] = {
                "threshold": limit,
                "first_ps": float(time_ps[int(np.argmax(crossed))]),
                "frames": int(crossed.sum()),
                "extreme": float(values.min() if side == "below" else values.max()),
            }

    return crossings


def coordination_changes(frames, time_ps):
    changed = np.array([f["non_water_oxygens"] > 0 for f in frames])
    owners = np.array([f["owner"] for f in frames])

    if len(owners) == 0:
        return {}

    away = owners != owners[0]
    persistent = [h for h in range(away.shape[1]) if longest_true_run(away[:, h]) >= PERSISTENT_FRAMES]

    return {
        "frames_with_coordination_change": int(changed.sum()),
        "frames_with_coordination_change_after_5ps": int((changed & (time_ps[:len(changed)] >= WINDOW_START_PS)).sum()),
        "longest_coordination_change_ps": float(longest_true_run(changed) * FRAME_PS),
        "hydrogen_owner_changes_between_frames": int((owners[1:] != owners[:-1]).sum()),
        "hydrogens_on_other_oxygen_at_end": int(away[-1].sum()),
        "hydrogens_persistently_on_other_oxygen": len(persistent),
        "frames_with_hh_below_1A": int(sum(f["min_hh_A"] < EVENTS["hh_close_A"] for f in frames)),
        "frames_with_oo_below_2A": int(sum(f["min_oo_A"] < EVENTS["oo_close_A"] for f in frames)),
        "min_hh_A": float(min(f["min_hh_A"] for f in frames)),
        "min_oo_A": float(min(f["min_oo_A"] for f in frames)),
    }


# mean kinetic temperatures in bins of one picosecond, for the equilibration check
def temperature_bins(temperatures, time_ps, end):
    edges = np.arange(0.0, TOTAL_STEPS * DT_PS + 0.5 * TEMPERATURE_BIN_PS, TEMPERATURE_BIN_PS)
    which = np.clip(np.digitize(time_ps[:end], edges) - 1, 0, len(edges) - 2)
    bins = {"centres": 0.5 * (edges[1:] + edges[:-1])}

    for name, values in temperatures.items():
        bins[name] = np.array([values[:end][which == k].mean() if (which == k).any() else np.nan for k in range(len(edges) - 1)])

    return bins


def analyse_run(run):
    meta = json.loads((run / "run.json").read_text())
    rec = {"run": run.name, "condition": meta["condition"], "seed": meta["seed"]}

    completion, complete = check_runs.check(run, TOTAL_STEPS)
    rec["completion"] = {k: v for k, v in completion.items() if k not in ("property_check", "frame_check")}
    rec["complete"] = complete

    if completion["state"] == "not started":
        return rec, None

    rec["bookkeeping"] = bookkeeping(completion)

    props, _ = read_properties(run)
    traj, _ = read_trajectory(run)
    symbols = np.array(traj["symbols"])
    masses = read(run / "init.extxyz").get_masses()
    time_ps = traj["steps"] * DT_PS
    volume = abs(np.linalg.det(traj["cells"][0]))

    frames = [frame_analysis(p, c, f, symbols) for p, c, f in zip(traj["positions"], traj["cells"], traj["forces"])]
    temperatures = kinetic_temperatures(masses, traj["velocities"], symbols)

    finite = np.array([f["finite"] for f in frames]) & np.isfinite(temperatures["all"])
    end = len(frames) if finite.all() else int(np.argmin(finite))
    exits = rec["completion"]["ipi_exits"]
    crashed = not complete and bool(exits) and exits[-1] != 0

    failures = []
    if end < len(frames):
        failures.append(f"non_finite_at_{time_ps[end]:.4f}_ps")
    if crashed:
        failures.append(f"crash_exit_{exits[-1]}")

    rec.update(
        density_g_cm3=float(masses.sum() * _amu * 1000.0 / (volume * 1e-24)),
        cell_A=float(volume ** (1.0 / 3.0)),
        frames=len(frames),
        analysed_frames=end,
        analysed_end_ps=float(time_ps[end - 1]) if end else 0.0,
        numerical_failure=",".join(failures) if failures else "none",
        threshold_crossings=threshold_crossings(frames, temperatures, time_ps, end),
    )

    analysed = np.arange(len(frames)) < end
    in_window = analysed & (time_ps >= WINDOW_START_PS)
    props_analysed = (props["step"] <= (traj["steps"][end - 1] if end else -1)) & np.isfinite(props["potential"])
    props_window = props_analysed & (props["time"] >= WINDOW_START_PS)

    if in_window.any():
        rec.update(
            analysis_frames=int(in_window.sum()),
            temperature_mean_K=float(temperatures["all"][in_window].mean()),
            temperature_std_K=float(temperatures["all"][in_window].std()),
            temperature_H_mean_K=float(temperatures["H"][in_window].mean()),
            temperature_O_mean_K=float(temperatures["O"][in_window].mean()),
            ipi_temperature_mean_K=float(props["temperature"][props_window].mean()),
            potential_mean_meV_per_molecule=float(props["potential"][props_window].mean() / NMOLECULES * 1000.0),
            pressure_mean_bar=float(props["pressure_md"][props_window].mean()),
            pressure_std_bar=float(props["pressure_md"][props_window].std()),
            nearest_oh_mean_A=float(np.mean([f["nearest_oh_mean_A"] for f, keep in zip(frames, in_window) if keep])),
            hoh_angle_mean_deg=float(np.nanmean([f["hoh_angle_mean_deg"] for f, keep in zip(frames, in_window) if keep])),
        )

    if "pot_component_raw(1)" in props and props_window.any():
        rec["correction_energy_mean_meV_per_molecule"] = float(props["pot_component_raw(1)"][props_window].mean() / NMOLECULES * 1000.0)

    if props_analysed.sum() > 2:
        drift = float(np.polyfit(props["time"][props_analysed], props["conserved"][props_analysed], 1)[0] / NATOMS * 1000.0)
        rec["conserved_drift_meV_per_atom_per_ps"] = drift
        rec["conserved_max_deviation_meV_per_atom"] = float(np.abs(props["conserved"][props_analysed] - props["conserved"][0]).max() / NATOMS * 1000.0)
        rec["integration_warning"] = abs(drift) > INTEGRATION_WARNING_MEV_PER_ATOM_PER_PS

    rec["max_atomic_force_eV_A"] = float(np.nanmax([f["max_atomic_force_eV_A"] for f in frames[:end]])) if end else None
    rec["min_pair_distance_A"] = float(np.nanmin([f["min_pair_distance_A"] for f in frames[:end]])) if end else None
    rec.update(coordination_changes(frames[:end], time_ps))

    potential = np.interp(time_ps, props["time"][props_analysed], props["potential"][props_analysed]) if props_analysed.any() else np.full(len(time_ps), np.nan)
    curves = {
        "time_ps": time_ps,
        "temperature": temperatures["all"],
        "temperature_H": temperatures["H"],
        "temperature_O": temperatures["O"],
        "temperature_bins": temperature_bins(temperatures, time_ps, end),
        "potential": potential,
    }

    window = structure(frames, in_window, volume)
    if window is not None:
        curves["window"], rec["window"] = window

    late = analysed & (time_ps >= LATE_WINDOW_START_PS)
    late_structure = structure(frames, late, volume)
    if late_structure is not None:
        curves["late"], rec["late"] = late_structure
        rec["late"]["potential_mean_meV_per_molecule"] = float(potential[late].mean() / NMOLECULES * 1000.0)
        rec["late"]["nearest_oh_mean_A"] = float(np.mean([f["nearest_oh_mean_A"] for f, keep in zip(frames, late) if keep]))

    # the discarded first 5 ps and the three production blocks, for equilibration and drift
    for lo, hi in (DISCARDED,) + BLOCKS:
        mask = block_mask(time_ps, analysed, lo, hi)
        block = structure(frames, mask, volume)
        if block is None:
            continue

        key = block_key(lo, hi)
        curves[key], rec[key] = block
        rec[key]["potential_mean_meV_per_molecule"] = float(potential[mask].mean() / NMOLECULES * 1000.0)
        rec[key]["temperature_mean_K"] = float(temperatures["all"][mask].mean())
        rec[key]["temperature_H_mean_K"] = float(temperatures["H"][mask].mean())
        rec[key]["temperature_O_mean_K"] = float(temperatures["O"][mask].mean())
        rec[key]["temperature_H_minus_O_K"] = rec[key]["temperature_H_mean_K"] - rec[key]["temperature_O_mean_K"]
        rec[key]["nearest_oh_mean_A"] = float(np.mean([f["nearest_oh_mean_A"] for f, keep in zip(frames, mask) if keep]))

    return rec, curves


def mean_se(values):
    values = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)

    if len(values) == 0:
        return None, None, 0
    if len(values) == 1:
        return float(values[0]), None, 1

    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))), len(values)


def num(record, key, spec):
    value = record.get(key)
    return format(value, spec) if value is not None else "n/a"


def fmt(mean, se, digits=3):
    if mean is None:
        return "n/a"
    if se is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def write_tsv(path, columns, rows):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + "\n")


def integrated_difference(a, b):
    sel = RDF_CENTRES >= CONVERGENCE_RMIN_A
    return float(np.abs(a[sel] - b[sel]).sum() * RDF_BIN)


def rdf_convergence(rows, curves):
    keys = [block_key(*b) for b in (DISCARDED,) + BLOCKS]
    result = {}

    for pair in PAIRS:
        within = {f"{keys[i]}_vs_{keys[i + 1]}": [] for i in range(len(keys) - 1)}
        for r in rows:
            c = curves[r["run"]]
            for i in range(len(keys) - 1):
                if c and keys[i] in c and keys[i + 1] in c:
                    within[f"{keys[i]}_vs_{keys[i + 1]}"].append(integrated_difference(c[keys[i]][pair], c[keys[i + 1]][pair]))

        across = []
        for key in keys[1:]:
            have = [curves[r["run"]][key][pair] for r in rows if curves[r["run"]] and key in curves[r["run"]]]
            across += [integrated_difference(a, b) for a, b in itertools.combinations(have, 2)]

        entry = {name: mean_se(values) for name, values in within.items()}
        entry["seed_to_seed_within_blocks"] = mean_se(across)

        production_changes = [entry[f"{keys[i]}_vs_{keys[i + 1]}"][0] for i in range(1, len(keys) - 1)]
        seed = entry["seed_to_seed_within_blocks"][0]
        if seed and all(v is not None for v in production_changes):
            entry["largest_production_block_change_over_seed_spread"] = max(production_changes) / seed

        result[pair] = entry

    return result


def summarise(records, curves):
    summary = {}

    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        with_window = [r for r in rows if "window" in r]

        entry = {
            "runs": len(rows),
            "complete": sum(r["complete"] for r in rows),
            "numerical_failures": {r["run"]: r["numerical_failure"] for r in rows if r.get("numerical_failure", "none") != "none"},
            "threshold_crossings": {r["run"]: r["threshold_crossings"] for r in rows if r.get("threshold_crossings")},
        }

        for c in RUN_MEASURES:
            entry[c] = mean_se([r.get(c) for r in rows])

        for c in SHELL_COLUMNS:
            entry[f"oo_{c}"] = mean_se([r["window"][c] for r in with_window])

        for c in SHELL_COLUMNS + ["potential_mean_meV_per_molecule", "nearest_oh_mean_A"]:
            entry[f"late_{c}"] = mean_se([r["late"][c] for r in rows if "late" in r])

        for lo, hi in (DISCARDED,) + BLOCKS:
            key = block_key(lo, hi)
            entry[key] = {c: mean_se([r[key][c] for r in rows if key in r]) for c in SHELL_COLUMNS + BLOCK_EXTRAS}

        # change from the first to the last production block within each run, a drift measure
        first, last = block_key(*BLOCKS[0]), block_key(*BLOCKS[-1])
        drift_runs = [r for r in rows if first in r and last in r]
        entry["last_minus_first_block"] = {c: mean_se([r[last][c] - r[first][c] for r in drift_runs]) for c in SHELL_COLUMNS + BLOCK_EXTRAS}

        entry["rdf_convergence"] = rdf_convergence(rows, curves)

        if with_window:
            mean_oo = np.mean([curves[r["run"]]["window"]["oo"] for r in with_window], axis=0)
            entry["mean_curve_oo_shells"] = oo_shells(mean_oo)

        summary[condition] = entry

    return summary


def write_curves(records, curves):
    np.save(OUT / "production_curves.npy", curves, allow_pickle=True)
    (OUT / "curves").mkdir(exist_ok=True)

    tables = [
        ("rdf_oo", RDF_CENTRES, "oo"),
        ("rdf_oh", RDF_CENTRES, "oh"),
        ("rdf_hh", RDF_CENTRES, "hh"),
        ("oh_nearest", 0.5 * (NEAREST_OH_EDGES[1:] + NEAREST_OH_EDGES[:-1]), "nearest_oh"),
        ("oh_second_nearest", 0.5 * (SECOND_OH_EDGES[1:] + SECOND_OH_EDGES[:-1]), "second_oh"),
    ]

    for window, suffix in (("window", "5_50ps"), ("late", "20_50ps")):
        for name, x, key in tables:
            columns, data = ["r_A"], [x]

            for condition in CONDITIONS:
                runs_here = [r["run"] for r in records if r["condition"] == condition and window in r]
                if runs_here:
                    columns.append(f"{condition}_mean")
                    data.append(np.mean([curves[n][window][key] for n in runs_here], axis=0))
                for n in runs_here:
                    columns.append(n)
                    data.append(curves[n][window][key])

            np.savetxt(OUT / "curves" / f"{name}_{suffix}.tsv", np.column_stack(data), delimiter="\t", header="\t".join(columns), comments="", fmt="%.6g")

    # mean kinetic temperatures per picosecond, per run
    columns, data = ["time_ps"], [None]
    for r in records:
        bins = curves[r["run"]]["temperature_bins"] if curves[r["run"]] else None
        if bins is None:
            continue
        data[0] = bins["centres"]
        for name in ("all", "H", "O"):
            columns.append(f"{r['run']}_{name}")
            data.append(bins[name])

    if data[0] is not None:
        np.savetxt(OUT / "curves" / "temperatures_1ps.tsv", np.column_stack(data), delimiter="\t", header="\t".join(columns), comments="", fmt="%.6g")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [D / "runs" / PHASE / f"{c}_seed{s}" for c in CONDITIONS for s in SEEDS]

    with Pool(len(runs)) as pool:
        results = pool.map(analyse_run, runs)

    records = [r for r, _ in results]
    curves = {r["run"]: c for r, c in results}
    (OUT / "production_runs.json").write_text(json.dumps(records, indent=1, default=float) + "\n")

    # one row per run
    head = ["run", "condition", "seed", "complete", "frames", "analysed_frames", "analysed_end_ps", "numerical_failure",
            "threshold_crossings", "integration_warning", "analysis_frames"]
    rows = [[r.get(c, "") if c != "threshold_crossings" else ",".join(r.get(c, {})) or "none" for c in head]
            + [r.get(c, "") for c in RUN_MEASURES] + [r.get("window", {}).get(c, "") for c in SHELL_COLUMNS] for r in records]
    write_tsv(OUT / "production_runs.tsv", head + RUN_MEASURES + [f"oo_{c}" for c in SHELL_COLUMNS], rows)

    # one row per run and time block, the discarded first 5 ps included
    block_rows = []
    for r in records:
        for lo, hi in (DISCARDED,) + BLOCKS:
            b = r.get(block_key(lo, hi))
            if b:
                block_rows.append([r["run"], r["condition"], r["seed"], f"{lo:g}-{hi:g}", b["frames"]] + [b[c] for c in SHELL_COLUMNS + BLOCK_EXTRAS])
    write_tsv(OUT / "production_blocks.tsv", ["run", "condition", "seed", "block_ps", "frames"] + SHELL_COLUMNS + BLOCK_EXTRAS, block_rows)

    summary = summarise(records, curves)
    (OUT / "production_conditions.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    write_curves(records, curves)
    write_report(records, summary)

    print(json.dumps({c: {k: s[k] for k in ("runs", "complete", "numerical_failures", "threshold_crossings", "oo_first_peak_g")} for c, s in summary.items()}, default=float))


def write_report(records, summary):
    densities = sorted({round(r["density_g_cm3"], 4) for r in records if "density_g_cm3" in r})
    cells = sorted({round(r["cell_A"], 6) for r in records if "cell_A" in r})
    blocks_text = ", ".join(f"{lo:g}-{hi:g} ps" for lo, hi in BLOCKS)

    lines = [
        "# Water production, automatic summary",
        "",
        f"Written by `scripts/analyse.py`. 32 H2O in a fixed periodic cube of {' and '.join(f'{c:.6f}' for c in cells)} Å, "
        "the Cookbook start with its molecular centres scaled to 1.0 g/cm³, "
        f"density {' and '.join(f'{d:.4f}' for d in densities)} g/cm³. NVT at 300 K.",
        "",
        "Means are over the velocity-seeded runs of a condition, ± one standard error over runs. "
        f"Analysis window {WINDOW_START_PS:g} ps to the end of each analysed trajectory, which ends only at a confirmed numerical failure. "
        f"Production blocks {blocks_text}, plus the discarded 0-5 ps for reference.",
        "",
        "Read the equilibration and convergence section before the structure tables. Removing the first 5 ps does not by itself establish equilibrium.",
    ]

    # runs, confirmed numerical failures apart from threshold crossings
    lines += [
        "",
        "## Runs",
        "",
        "A confirmed numerical failure is a non-finite position, velocity or force, or a crashed run. "
        f"Threshold crossings (any distance below {THRESHOLDS['min_pair_distance_A']} Å, any force above {THRESHOLDS['max_atomic_force_eV_A']} eV/Å, "
        f"kinetic temperature above {THRESHOLDS['kinetic_temperature_K']} K) are listed separately, flagged for inspection, and do not cut the trajectory.",
        "",
        "| run | complete | analysed to (ps) | confirmed numerical failure | threshold crossings | integration warning | T (K) | T_H (K) | T_O (K) | conserved drift (meV/atom/ps) | max F (eV/Å) | min d (Å) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "frames" not in r:
            lines.append(f"| {r['run']} | no | 0 | not started | | | | | | | | |")
            continue

        crossing_text = ", ".join(r["threshold_crossings"]) or "none"
        lines.append(
            f"| {r['run']} | {'yes' if r['complete'] else 'no'} | {r['analysed_end_ps']:.2f} | {r['numerical_failure']} | {crossing_text} | "
            f"{'yes' if r.get('integration_warning') else 'no'} | {num(r, 'temperature_mean_K', '.1f')} | {num(r, 'temperature_H_mean_K', '.1f')} | "
            f"{num(r, 'temperature_O_mean_K', '.1f')} | {num(r, 'conserved_drift_meV_per_atom_per_ps', '+.4f')} | {num(r, 'max_atomic_force_eV_A', '.2f')} | "
            f"{num(r, 'min_pair_distance_A', '.3f')} |"
        )

    crossing_rows = [(r["run"], name, c) for r in records for name, c in r.get("threshold_crossings", {}).items()]
    lines += ["", "## Threshold crossings", ""]
    if crossing_rows:
        lines += ["| run | criterion | threshold | first crossing (ps) | frames crossing | extreme value |", "|---|---|---|---|---|---|"]
        lines += [f"| {run} | {name} | {c['threshold']} | {c['first_ps']:.4f} | {c['frames']} | {c['extreme']:.4g} |" for run, name, c in crossing_rows]
    else:
        lines.append("No frame of any run crosses a threshold.")

    lines += [
        "",
        "## Frame bookkeeping",
        "",
        "Positions, velocities and forces are paired by step, the latest occurrence of a step wins. Restart duplicates count the superseded occurrences over the three files. "
        "All trajectory files are kept in full.",
        "",
        "| run | segments | paired frames (of 4001) | restart duplicates | damaged frames | damaged property rows | missing steps | unpaired steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        b = r.get("bookkeeping")
        if b:
            lines.append(f"| {r['run']} | {b['segments']} | {b['paired_frames']} | {b['restart_duplicates']} | {b['damaged_frames']} | "
                         f"{b['damaged_property_rows']} | {b['missing_steps']} | {b['unpaired_steps']} |")

    # equilibration and convergence before any structural reading
    lines += [
        "",
        "## Equilibration and convergence",
        "",
        "### Kinetic temperatures by block",
        "",
        "From the saved velocities, each subset over its share of 3N-3 degrees of freedom. Per-picosecond means are in `curves/temperatures_1ps.tsv` and `figures/temperature_water.png`.",
        "",
        "| condition | block (ps) | T (K) | T_H (K) | T_O (K) | T_H - T_O (K) |",
        "|---|---|---|---|---|---|",
    ]
    for condition, s in summary.items():
        for lo, hi in (DISCARDED,) + BLOCKS:
            b = s[block_key(lo, hi)]
            tag = " (discarded)" if (lo, hi) == DISCARDED else ""
            lines.append(f"| {LABELS[condition]} | {lo:g}-{hi:g}{tag} | {fmt(*b['temperature_mean_K'][:2], 1)} | {fmt(*b['temperature_H_mean_K'][:2], 1)} | "
                         f"{fmt(*b['temperature_O_mean_K'][:2], 1)} | {fmt(*b['temperature_H_minus_O_K'][:2], 1)} |")
        d = s["last_minus_first_block"]
        lines.append(f"| {LABELS[condition]} | last minus first production block | {fmt(*d['temperature_mean_K'][:2], 1)} | {fmt(*d['temperature_H_mean_K'][:2], 1)} | "
                     f"{fmt(*d['temperature_O_mean_K'][:2], 1)} | {fmt(*d['temperature_H_minus_O_K'][:2], 1)} |")

    lines += [
        "",
        "### O-O structure, potential energy and O-H bond by block",
        "",
        "| condition | block (ps) | first peak r (Å) | first peak g | first min g | second peak g | n_OO(3.3 Å) | potential (meV/molecule) | nearest O-H (Å) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, s in summary.items():
        for lo, hi in (DISCARDED,) + BLOCKS:
            b = s[block_key(lo, hi)]
            tag = " (discarded)" if (lo, hi) == DISCARDED else ""
            lines.append(f"| {LABELS[condition]} | {lo:g}-{hi:g}{tag} | {fmt(*b['first_peak_r_A'][:2], 2)} | {fmt(*b['first_peak_g'][:2], 2)} | "
                         f"{fmt(*b['first_min_g'][:2], 2)} | {fmt(*b['second_peak_g'][:2], 2)} | {fmt(*b['n_oo_common_radius'][:2], 2)} | "
                         f"{fmt(*b['potential_mean_meV_per_molecule'][:2], 1)} | {fmt(*b['nearest_oh_mean_A'][:2], 4)} |")
        d = s["last_minus_first_block"]
        lines.append(f"| {LABELS[condition]} | last minus first production block | {fmt(*d['first_peak_r_A'][:2], 2)} | {fmt(*d['first_peak_g'][:2], 2)} | "
                     f"{fmt(*d['first_min_g'][:2], 2)} | {fmt(*d['second_peak_g'][:2], 2)} | {fmt(*d['n_oo_common_radius'][:2], 2)} | "
                     f"{fmt(*d['potential_mean_meV_per_molecule'][:2], 1)} | {fmt(*d['nearest_oh_mean_A'][:2], 4)} |")

    lines += ["", "Drift flags, a last-minus-first change larger than twice its standard error over runs:", ""]
    for condition, s in summary.items():
        flagged = [c for c, (m, se, n) in s["last_minus_first_block"].items() if m is not None and se is not None and abs(m) > 2 * se]
        lines.append(f"- {LABELS[condition]}: " + (", ".join(flagged) if flagged else "none"))

    keys = [block_key(*b) for b in (DISCARDED,) + BLOCKS]
    labels = [f"{lo:g}-{hi:g}" for lo, hi in (DISCARDED,) + BLOCKS]
    lines += [
        "",
        "### RDF convergence across blocks",
        "",
        f"Integrated absolute difference of two g(r), summed over r ≥ {CONVERGENCE_RMIN_A} Å in Å, which leaves out the covalent O-H peak. "
        "Within-run columns compare successive blocks of the same run. The seed-to-seed column compares different runs in the same production block, "
        "the size of the difference expected from finite sampling alone. A ratio above one means a block-to-block change larger than that spread, "
        "a ratio well below one means the runs differ from each other more than each run changes over time, a sign that they sample different states. "
        "Neither alone proves convergence, read it with the drift flags above.",
        "",
        "| pair | condition | " + " | ".join(f"{labels[i]} vs {labels[i + 1]} ps" for i in range(len(keys) - 1)) + " | seed to seed | largest production change / seed spread |",
        "|---|---|" + "---|" * (len(keys) - 1) + "---|---|",
    ]
    for pair in PAIRS:
        for condition, s in summary.items():
            c = s["rdf_convergence"][pair]
            cells = [fmt(*c[f"{keys[i]}_vs_{keys[i + 1]}"][:2], 3) for i in range(len(keys) - 1)]
            cells.append(fmt(*c["seed_to_seed_within_blocks"][:2], 3))
            cells.append(num(c, "largest_production_block_change_over_seed_spread", ".2f"))
            lines.append(f"| {PAIR_LABELS[pair]} | {LABELS[condition]} | " + " | ".join(cells) + " |")

    # structure over the full analysis window
    lines += [
        "",
        "## O-O structure, 5-50 ps",
        "",
        f"RDF grid {RDF_BIN} Å bins to {RDF_RMAX} Å, below half the cell width, minimum image, ideal-gas normalisation at the density of this cell. "
        "O-H and H-H RDFs include intramolecular pairs.",
        "",
        "| condition | first peak r (Å) | first peak g | first min r (Å) | first min g | second peak r (Å) | second peak g | peak/min | n_OO(3.3 Å) | n_OO(own first min) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, s in summary.items():
        lines.append(f"| {LABELS[condition]} | {fmt(*s['oo_first_peak_r_A'][:2], 2)} | {fmt(*s['oo_first_peak_g'][:2], 2)} | {fmt(*s['oo_first_min_r_A'][:2], 2)} | "
                     f"{fmt(*s['oo_first_min_g'][:2], 2)} | {fmt(*s['oo_second_peak_r_A'][:2], 2)} | {fmt(*s['oo_second_peak_g'][:2], 2)} | "
                     f"{fmt(*s['oo_peak_to_min_ratio'][:2], 2)} | {fmt(*s['oo_n_oo_common_radius'][:2], 2)} | {fmt(*s['oo_n_oo_own_first_min'][:2], 2)} |")

    lines += [
        "",
        "## O-O structure, 20-50 ps, supplementary",
        "",
        "Added after the block analysis showed the LDA-corrected structure still relaxing through 5-20 ps. The 5-50 ps table above is the frozen protocol, "
        "this later window leaves out that relaxation for all three conditions alike.",
        "",
        "| condition | first peak r (Å) | first peak g | first min r (Å) | first min g | second peak r (Å) | second peak g | peak/min | n_OO(3.3 Å) | potential (meV/molecule) | nearest O-H (Å) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, s in summary.items():
        lines.append(f"| {LABELS[condition]} | {fmt(*s['late_first_peak_r_A'][:2], 2)} | {fmt(*s['late_first_peak_g'][:2], 2)} | {fmt(*s['late_first_min_r_A'][:2], 2)} | "
                     f"{fmt(*s['late_first_min_g'][:2], 2)} | {fmt(*s['late_second_peak_r_A'][:2], 2)} | {fmt(*s['late_second_peak_g'][:2], 2)} | "
                     f"{fmt(*s['late_peak_to_min_ratio'][:2], 2)} | {fmt(*s['late_n_oo_common_radius'][:2], 2)} | "
                     f"{fmt(*s['late_potential_mean_meV_per_molecule'][:2], 1)} | {fmt(*s['late_nearest_oh_mean_A'][:2], 4)} |")

    lines += [
        "",
        "## Thermodynamics and molecules, 5-50 ps",
        "",
        "Nearest O-H distance: for every H, the distance to its nearest O. Pressure is the virial pressure of a 96-atom cell, a diagnostic only.",
        "",
        "| condition | T (K) | T_H (K) | T_O (K) | potential (meV/molecule) | pressure (bar) | nearest O-H (Å) | H-O-H (deg) | correction energy (meV/molecule) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for condition, s in summary.items():
        lines.append(f"| {LABELS[condition]} | {fmt(*s['temperature_mean_K'][:2], 1)} | {fmt(*s['temperature_H_mean_K'][:2], 1)} | {fmt(*s['temperature_O_mean_K'][:2], 1)} | "
                     f"{fmt(*s['potential_mean_meV_per_molecule'][:2], 1)} | {fmt(*s['pressure_mean_bar'][:2], 0)} | {fmt(*s['nearest_oh_mean_A'][:2], 4)} | "
                     f"{fmt(*s['hoh_angle_mean_deg'][:2], 2)} | {fmt(*s['correction_energy_mean_meV_per_molecule'][:2], 1)} |")

    # coordination changes over the whole analysed trajectory
    lines += [
        "",
        "## Coordination changes, possible proton-transfer events",
        "",
        "Each H is assigned to its nearest O in every frame. A frame with any O that owns other than two H, or an H whose nearest O changes, "
        "is a coordination change and a possible proton-transfer event. It is reported and never counted as a numerical failure. "
        f"Persistently on another O means away from the initial O for at least {PERSISTENT_FRAMES * FRAME_PS:g} ps in a row.",
        "",
        "| run | frames with a coordination change (all / after 5 ps) | longest stretch (ps) | nearest-O changes between frames | H on another O at end | H persistently on another O | frames with H-H < 1 Å | frames with O-O < 2 Å |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if "frames_with_coordination_change" not in r:
            continue
        lines.append(f"| {r['run']} | {r['frames_with_coordination_change']} / {r['frames_with_coordination_change_after_5ps']} | {r['longest_coordination_change_ps']:.4f} | "
                     f"{r['hydrogen_owner_changes_between_frames']} | {r['hydrogens_on_other_oxygen_at_end']} | {r['hydrogens_persistently_on_other_oxygen']} | "
                     f"{r['frames_with_hh_below_1A']} | {r['frames_with_oo_below_2A']} |")

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
