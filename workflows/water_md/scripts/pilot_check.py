import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observables import THRESHOLDS, frame_analysis, kinetic_temperatures  # noqa: E402
from water import (  # noqa: E402
    BASELINE_OUTPUT,
    BASELINE_PT,
    CONDITIONS,
    CORRECTION_OF,
    D,
    DELTA_OUTPUT,
    LDA_PT,
    NATOMS,
    PBE_PT,
    PHASE_DIRS,
    STRIDE,
    TEMPERATURE_K,
    Potential,
    atoms_call,
    read_properties,
    read_trajectory,
)


RUNS = D / "runs"
OUT = D / "verification"
FRAMES = 10
NVT_STEPS = 8000
NVT_EQUILIBRATION_PS = 0.5


# drift of the conserved quantity from a straight-line fit, and its fluctuation around the fit
def drift(props):
    t, e = props["time"], props["conserved"]
    slope, intercept = np.polyfit(t, e, 1)
    residual = e - (slope * t + intercept)

    return {
        "steps": int(props["step"][-1]),
        "time_ps": float(t[-1]),
        "drift_meV_per_atom_per_ps": float(slope / NATOMS * 1000.0),
        "fluctuation_rms_meV_per_atom": float(residual.std() / NATOMS * 1000.0),
        "max_deviation_meV_per_atom": float(np.abs(e - e[0]).max() / NATOMS * 1000.0),
    }


# steps per second of the first segment, without the 15 s start-up pause
def wall_rate(run, steps):
    lines = [line for line in (run / "segments.log").read_text().splitlines() if "start from" in line or "i-PI exit" in line]
    start = datetime.strptime(lines[0].split()[0], "%Y-%m-%d_%H:%M:%S")
    end = datetime.strptime(lines[1].split()[0], "%Y-%m-%d_%H:%M:%S")
    return float(steps / ((end - start).total_seconds() - 15.0))


# geometry, temperatures and failure measures over the saved frames of one run
def trajectory_summary(run, after_ps):
    traj, check = read_trajectory(run)
    props, prop_check = read_properties(run)
    symbols = np.array(traj["symbols"])
    masses = read(run / "init.extxyz").get_masses()
    time_ps = traj["steps"] * float(json.loads((run / "run.json").read_text())["timestep_fs"]) / 1000.0
    late = time_ps >= after_ps

    frames = [frame_analysis(p, c, f, symbols) for p, c, f in zip(traj["positions"], traj["cells"], traj["forces"])]
    temperatures = kinetic_temperatures(masses, traj["velocities"], symbols)
    momentum = (masses[None, :, None] * traj["velocities"]).sum(1)

    # the saved velocities against the kinetic energy that i-PI reports at the same steps
    kinetic_saved = 0.5 * (masses[None, :, None] * traj["velocities"] ** 2).sum((1, 2))
    kinetic_ipi = dict(zip(props["step"].astype(int), props["kinetic_md"]))
    kinetic_gap = max(abs(kinetic_saved[i] - kinetic_ipi[s]) / kinetic_ipi[s] for i, s in enumerate(traj["steps"]))

    return {
        "frames": len(frames),
        "frame_check": check,
        "property_check": prop_check,
        "window_ps": [after_ps, float(time_ps[-1])],
        "temperature_mean_K": float(temperatures["all"][late].mean()),
        "temperature_std_K": float(temperatures["all"][late].std()),
        "temperature_H_mean_K": float(temperatures["H"][late].mean()),
        "temperature_O_mean_K": float(temperatures["O"][late].mean()),
        "temperature_max_K": float(temperatures["all"].max()),
        "max_atomic_force_eV_A": max(f["max_atomic_force_eV_A"] for f in frames),
        "mean_frame_max_force_eV_A": float(np.mean([f["max_atomic_force_eV_A"] for f in frames])),
        "min_pair_distance_A": min(f["min_pair_distance_A"] for f in frames),
        "nearest_oh_mean_A": float(np.mean([f["nearest_oh_mean_A"] for f, keep in zip(frames, late) if keep])),
        "nearest_oh_max_A": max(f["nearest_oh_max_A"] for f in frames),
        "hoh_angle_mean_deg": float(np.nanmean([f["hoh_angle_mean_deg"] for f, keep in zip(frames, late) if keep])),
        "min_hh_A": min(f["min_hh_A"] for f in frames),
        "min_oo_A": min(f["min_oo_A"] for f in frames),
        "frames_with_non_water_oxygen": int(sum(f["non_water_oxygens"] > 0 for f in frames)),
        "max_hydrogens_off_initial_oxygen": max(f["hydrogens_off_initial_oxygen"] for f in frames),
        "all_finite": all(f["finite"] for f in frames),
        "com_momentum_max_ase": float(np.abs(momentum).max()),
        "kinetic_saved_vs_ipi_max_relative_gap": float(kinetic_gap),
    }


# usage pilot_check.py, every condition
def main():
    checks, report = [], {}
    baseline = Potential(BASELINE_PT, BASELINE_OUTPUT, "cpu")
    deltas = {"lda": Potential(LDA_PT, DELTA_OUTPUT, "cpu"), "pbe": Potential(PBE_PT, DELTA_OUTPUT, "cpu")}

    for condition in CONDITIONS:
        correction = CORRECTION_OF[condition]

        # NVE energy conservation at the production time step and at half of it
        for phase in ("pilot_nve", "pilot_nve_half_dt"):
            run = RUNS / PHASE_DIRS[phase] / f"{condition}_seed0"
            if not (run / "sim.out").exists():
                continue

            props, _ = read_properties(run)
            rec = drift(props)
            finished = "i-PI exit" in (run / "segments.log").read_text()
            rec["steps_per_second"] = wall_rate(run, props["step"][-1]) if finished else None
            rec.update(trajectory_summary(run, 0.0))
            report[f"{condition}:{phase}"] = rec

        run = RUNS / PHASE_DIRS["pilot_nve"] / f"{condition}_seed0"

        # the initial kinetic temperature that i-PI starts from equals the one of the written velocities
        meta = json.loads((run / "run.json").read_text())
        props, _ = read_properties(run)
        masses = read(run / "init.extxyz").get_masses()
        kinetic_ev = props["kinetic_md"][0]
        start_temperature = 2.0 * kinetic_ev / ((3 * NATOMS - 3) * 8.617333262e-5)
        gap = abs(start_temperature - meta["initial_temperature_K_3N_minus_3"])
        checks.append((f"{condition}: i-PI starts from the written velocities ({start_temperature:.3f} K against {meta['initial_temperature_K_3N_minus_3']:.3f} K over 3N-3)",
                       gap < 1e-3))

        # both clients of a corrected run receive the same geometry and cell at every logged call, in the units of the start file
        if correction is not None:
            base = [json.loads(line) for line in (run / "geometry_baseline.jsonl").read_text().splitlines()]
            corr = [json.loads(line) for line in (run / "geometry_correction.jsonl").read_text().splitlines()]
            same = len(base) == len(corr) > 0 and all(
                a["positions_sha256"] == b["positions_sha256"] and a["cell_sha256"] == b["cell_sha256"] for a, b in zip(base, corr)
            )
            checks.append((f"{condition}: both force-field clients received identical positions and cells in all {len(base)} logged calls", same))

            init = read(run / "init.extxyz")
            unit_gap = max(
                float(np.abs(np.array(base[0]["first_atom_angstrom"]) - init.positions[0]).max() / np.abs(init.positions[0]).max()),
                float(np.abs(np.array(base[0]["cell_angstrom"]).T - init.cell.array).max() / np.abs(init.cell.array).max()),
            )
            checks.append((f"{condition}: positions and cell received by the clients equal the start file in angstrom (max relative gap {unit_gap:.1e})",
                           unit_gap < 1e-7))

        # i-PI energies and forces against independent evaluation and subtraction on saved frames
        traj, _ = read_trajectory(run)
        potential = dict(zip(props["step"].astype(int), props["potential"]))
        comparison = []

        for k in np.linspace(0, len(traj["steps"]) - 1, FRAMES).astype(int):
            step = int(traj["steps"][k])
            atoms = Atoms(symbols=traj["symbols"], positions=traj["positions"][k], cell=traj["cells"][k], pbc=True)
            e_ref, f_ref = atoms_call(baseline, atoms)

            if correction is not None:
                e_d, f_d = atoms_call(deltas[correction], atoms)
                e_ref, f_ref = e_ref - e_d, f_ref - f_d

            comparison.append({
                "step": step,
                "E_ipi_eV": float(potential[step]),
                "E_independent_eV": e_ref,
                "dE_eV": abs(float(potential[step]) - e_ref),
                "dF_max_eV_A": float(np.abs(traj["forces"][k] - f_ref).max()),
                "F_max_eV_A": float(np.abs(f_ref).max()),
            })

        report[f"{condition}:ipi_vs_independent"] = comparison
        worst_e = max(c["dE_eV"] for c in comparison)
        worst_f = max(c["dF_max_eV_A"] for c in comparison)
        checks.append((f"{condition}: i-PI energies and forces equal independent evaluation and subtraction on {len(comparison)} saved frames "
                       f"(max {worst_e:.1e} eV, {worst_f:.1e} eV/A)", worst_e < 1e-3 and worst_f < 1e-3))

        nve = report[f"{condition}:pilot_nve"]
        checks.append((f"{condition}: saved velocities reproduce the i-PI kinetic energy (max relative gap {nve['kinetic_saved_vs_ipi_max_relative_gap']:.1e})",
                       nve["kinetic_saved_vs_ipi_max_relative_gap"] < 1e-5))

        # NVT pilot run as segments killed without warning, stitched outputs must be continuous and complete
        run = RUNS / PHASE_DIRS["pilot_nvt"] / f"{condition}_seed0"
        if (run / "sim.out").exists():
            props, _ = read_properties(run)
            rec = trajectory_summary(run, NVT_EQUILIBRATION_PS)
            log = (run / "segments.log").read_text()
            rec["segments"] = log.count("start from")
            rec["ipi_exits"] = [int(x) for x in re.findall(r"i-PI exit (\d+)", log)]
            rec["last_step"] = int(props["step"][-1])
            rec["conserved_drift_meV_per_atom_per_ps"] = float(np.polyfit(props["time"], props["conserved"], 1)[0] / NATOMS * 1000.0)
            rec["ipi_temperature_mean_K"] = float(props["temperature"][props["time"] >= NVT_EQUILIBRATION_PS].mean())
            rec["pressure_mean_bar"] = float(props["pressure_md"][props["time"] >= NVT_EQUILIBRATION_PS].mean())
            report[f"{condition}:pilot_nvt"] = rec

            continuous = (rec["frames"] == NVT_STEPS // STRIDE + 1 and not rec["frame_check"]["missing_steps"]
                          and not rec["frame_check"]["unexpected_steps"] and not rec["frame_check"]["steps_not_in_all_three"])
            checks.append((f"{condition}: NVT pilot reached step {rec['last_step']} over {rec['segments']} killed and resumed segments with "
                           f"{rec['frames']} continuous paired frames", rec["last_step"] >= NVT_STEPS and continuous))

    # the largest force seen in any pilot frame sets the force failure threshold
    pilot_max_force = max(r["max_atomic_force_eV_A"] for k, r in report.items() if isinstance(r, dict) and "max_atomic_force_eV_A" in r)
    report["pilot_max_atomic_force_eV_A"] = pilot_max_force
    report["failure_criteria_before_force_threshold"] = THRESHOLDS

    passed = all(ok for _, ok in checks)
    report["checks"] = [{"check": c, "passed": ok} for c, ok in checks]
    (OUT / "pilot_checks.json").write_text(json.dumps(report, indent=1, default=float) + "\n")

    for key, r in report.items():
        if key.endswith(("pilot_nve", "pilot_nve_half_dt")):
            rate = f"{r['steps_per_second']:.1f} steps/s" if r["steps_per_second"] else "unfinished"
            print(f"{key:34s} {r['time_ps']:.2f} ps, drift {r['drift_meV_per_atom_per_ps']:+.4f} meV/atom/ps, fluctuation {r['fluctuation_rms_meV_per_atom']:.4f}, "
                  f"max deviation {r['max_deviation_meV_per_atom']:.4f} meV/atom, T {r['temperature_mean_K']:.0f} K (H {r['temperature_H_mean_K']:.0f}, O {r['temperature_O_mean_K']:.0f}), "
                  f"max F {r['max_atomic_force_eV_A']:.2f} eV/A, {rate}")

        if key.endswith("pilot_nvt"):
            print(f"{key:34s} T after {NVT_EQUILIBRATION_PS} ps {r['temperature_mean_K']:.1f} +- {r['temperature_std_K']:.1f} K (H {r['temperature_H_mean_K']:.1f}, O {r['temperature_O_mean_K']:.1f}), "
                  f"conserved drift {r['conserved_drift_meV_per_atom_per_ps']:+.4f} meV/atom/ps, max F {r['max_atomic_force_eV_A']:.2f} eV/A, "
                  f"min d {r['min_pair_distance_A']:.3f} A, O-H {r['nearest_oh_mean_A']:.4f} A, HOH {r['hoh_angle_mean_deg']:.1f} deg, "
                  f"p {r['pressure_mean_bar']:.0f} bar, non-water frames {r['frames_with_non_water_oxygen']}")

    print(f"largest atomic force in any pilot frame {pilot_max_force:.2f} eV/A, bath {TEMPERATURE_K} K")

    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)

    print("PILOT CHECKS", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
