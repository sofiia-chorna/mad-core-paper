import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.units import _amu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water import (  # noqa: E402
    BASELINE_OUTPUT,
    BASELINE_PT,
    D,
    DELTA_OUTPUT,
    EXPECTED_SHA256,
    FLOAT64,
    HASHED_FILES,
    LDA_PT,
    NATOMS,
    NMOLECULES,
    PBE_PT,
    SEEDS,
    TARGET_DENSITY_G_CM3,
    TEMPERATURE_K,
    Potential,
    atoms_call,
    kinetic_temperature,
    sha256,
    source_atoms,
    water_start,
)


OUT = D / "verification"
SEED = 20261010
FD_STEPS = (0.001, 0.002, 0.005, 0.01)
FD_COMPONENTS = 12


def magnitudes(energy, forces):
    return {
        "energy_eV": energy,
        "energy_meV_per_molecule": energy / NMOLECULES * 1000.0,
        "max_atomic_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "rms_force_component_eV_A": float(np.sqrt((forces**2).mean())),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks, report = [], {"device": device}

    # identities of the source file, the models and the driver
    hashes = {name: sha256(path) for name, path in HASHED_FILES.items()}
    report["sha256"] = hashes
    report["sha256_float64_copies"] = {p.name: sha256(p) for p in sorted(FLOAT64.glob("*.pt"))}

    for name, value in EXPECTED_SHA256.items():
        checks.append((f"{name} sha256 is {value[:16]}", hashes[name] == value))

    # the source geometry, its cell and its density
    atoms = source_atoms()
    symbols = np.array(atoms.get_chemical_symbols())
    o = np.where(symbols == "O")[0]
    h = np.where(symbols == "H")[0]
    distances = atoms.get_all_distances(mic=True)
    nearest_o = o[distances[np.ix_(h, o)].argmin(1)]
    density = atoms.get_masses().sum() * _amu * 1000.0 / (atoms.get_volume() * 1e-24)

    report["source"] = {
        "natoms": len(atoms),
        "oxygen": len(o),
        "hydrogen": len(h),
        "cell_parameters": atoms.cell.cellpar().tolist(),
        "volume_A3": atoms.get_volume(),
        "density_g_cm3": density,
        "masses_amu": {"H": float(atoms.get_masses()[h[0]]), "O": float(atoms.get_masses()[o[0]])},
        "nearest_oh_A_min_max": [float(distances[np.ix_(h, o)].min(1).min()), float(distances[np.ix_(h, o)].min(1).max())],
        "min_distance_A": float(distances[~np.eye(len(atoms), dtype=bool)].min()),
        "half_smallest_perpendicular_width_A": float(min(atoms.cell.volume / np.linalg.norm(np.cross(atoms.cell[i - 1], atoms.cell[i - 2])) for i in range(3)) / 2.0),
    }
    checks.append(("source has 96 atoms, 32 O and 64 H", len(atoms) == NATOMS and len(o) == NMOLECULES and len(h) == 2 * NMOLECULES))

    molecules = all(symbols[3 * k] == "O" and symbols[3 * k + 1] == "H" and symbols[3 * k + 2] == "H" for k in range(NMOLECULES))
    own = all(nearest_o[i] == 3 * (h[i] // 3) for i in range(len(h)))
    checks.append(("atoms come as O H H per molecule and every H is nearest to its own O", molecules and own))

    # the cubic cell that holds the actual masses at 1.0 g/cm3, the derivation from the pdb is checked by build_start.py
    expected_length = (atoms.get_masses().sum() * _amu * 1000.0 / TARGET_DENSITY_G_CM3 * 1e24) ** (1.0 / 3.0)
    cubic = np.allclose(atoms.cell.cellpar(), [expected_length] * 3 + [90.0] * 3) and atoms.pbc.all()
    checks.append((f"cell is the periodic {expected_length:.6f} A cube of 1.0 g/cm3", bool(cubic)))
    checks.append((f"density is {density:.6f} g/cm3", bool(abs(density - TARGET_DENSITY_G_CM3) < 1e-6)))

    # the five velocity-seeded starts share the positions, carry no centre-of-mass momentum, and differ in velocities
    starts = []
    for seed in SEEDS:
        start = water_start(seed)
        momentum = (start.get_masses()[:, None] * start.get_velocities()).sum(0)
        starts.append({
            "seed": seed,
            "positions_equal_source": bool(np.array_equal(start.positions, atoms.positions)),
            "temperature_K_3N_minus_3": float(kinetic_temperature(start.get_masses(), start.get_velocities())),
            "com_momentum_max_ase": float(np.abs(momentum).max()),
            "velocities_sha256": hashlib.sha256(start.get_velocities().tobytes()).hexdigest(),
        })

    report["starts"] = starts
    checks.append(("all starts have the source positions and zero centre-of-mass momentum",
                   all(s["positions_equal_source"] and s["com_momentum_max_ase"] < 1e-10 for s in starts)))
    checks.append(("the five velocity sets differ", len({s["velocities_sha256"] for s in starts}) == len(SEEDS)))

    # each model on the source geometry, float32 against float64 copies of the same weights
    float32 = {
        "baseline": Potential(BASELINE_PT, BASELINE_OUTPUT, device),
        "lda": Potential(LDA_PT, DELTA_OUTPUT, device),
        "pbe": Potential(PBE_PT, DELTA_OUTPUT, device),
    }
    float64 = {
        "baseline": Potential(FLOAT64 / "baseline_inner.float64.pt", BASELINE_OUTPUT, device),
        "lda": Potential(FLOAT64 / "delta_lda.float64.pt", DELTA_OUTPUT, device),
        "pbe": Potential(FLOAT64 / "delta_pbe.float64.pt", DELTA_OUTPUT, device),
    }

    values32 = {name: atoms_call(p, atoms) for name, p in float32.items()}
    values64 = {name: atoms_call(p, atoms) for name, p in float64.items()}

    static = {name: magnitudes(*values32[name]) for name in float32}
    for name in ("lda", "pbe"):
        energy = values32["baseline"][0] - values32[name][0]
        forces = values32["baseline"][1] - values32[name][1]
        static[f"baseline minus {name} correction"] = magnitudes(energy, forces)
        static[name]["rms_force_ratio_to_baseline"] = static[name]["rms_force_component_eV_A"] / static["baseline"]["rms_force_component_eV_A"]

    report["static_on_source_geometry"] = static

    for name in ("lda", "pbe"):
        f32 = values32["baseline"][1] - values32[name][1]
        f64 = values64["baseline"][1] - values64[name][1]
        e32 = values32["baseline"][0] - values32[name][0]
        e64 = values64["baseline"][0] - values64[name][0]
        gap_f = float(np.abs(f32 - f64).max())
        gap_e = abs(e32 - e64) / NATOMS
        report[f"{name}_float32_vs_float64"] = {"max_force_gap_eV_A": gap_f, "energy_gap_eV_per_atom": gap_e}
        checks.append((f"{name}: float32 combined forces equal float64 combined forces on water (max {gap_f:.1e} eV/A, {gap_e:.1e} eV/atom)",
                       gap_f < 5e-3 and gap_e < 1e-4))

    # each combined force against central differences of its combined energy, float64
    rng = np.random.default_rng(SEED)
    picks = [(int(i), int(c)) for i, c in zip(rng.choice(NATOMS, FD_COMPONENTS, replace=False), rng.integers(0, 3, FD_COMPONENTS))]
    report["finite_differences"] = {}

    for name in ("lda", "pbe"):
        analytic = values64["baseline"][1] - values64[name][1]
        rows = []

        for step in FD_STEPS:
            for i, c in picks:
                energies = []
                for sign in (1, -1):
                    moved = atoms.copy()
                    moved.positions[i, c] += sign * step
                    energies.append(atoms_call(float64["baseline"], moved)[0] - atoms_call(float64[name], moved)[0])

                numeric = -(energies[0] - energies[1]) / (2 * step)
                rows.append({"h_A": step, "atom": i, "component": c, "analytic_eV_A": float(analytic[i, c]),
                             "numeric_eV_A": float(numeric), "difference_eV_A": float(abs(numeric - analytic[i, c]))})

        best = {step: max(r["difference_eV_A"] for r in rows if r["h_A"] == step) for step in FD_STEPS}
        report["finite_differences"][name] = {"rows": rows, "max_by_step": best}
        checks.append((f"{name}: combined forces equal central differences of the combined energy in float64 (max {min(best.values()):.1e} eV/A at the best step)",
                       min(best.values()) < 1e-4))

    passed = all(ok for _, ok in checks)
    report["checks"] = [{"check": c, "passed": ok} for c, ok in checks]
    report["temperature_K"] = TEMPERATURE_K
    (OUT / "water_checks.json").write_text(json.dumps(report, indent=1) + "\n")

    for name, values in static.items():
        print(f"{name:32s} " + ", ".join(f"{k} {v:.4g}" for k, v in values.items()))

    for s in starts:
        print(f"seed {s['seed']} initial temperature {s['temperature_K_3N_minus_3']:.1f} K")

    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)

    print("WATER CHECKS", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
