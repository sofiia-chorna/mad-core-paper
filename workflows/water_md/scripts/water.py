import hashlib
import re
from pathlib import Path

import metatomic.torch as mta
import numpy as np
import torch
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from vesin.metatomic import compute_requested_neighbors


D = Path(__file__).resolve().parent.parent
VALID = D.parent.parent
LIPS = VALID / "md" / "lips_delta_10k"
DELTA_RUNS = VALID / "experiments" / "delta_lda_pbe_18k" / "runs" / "main"

SOURCE_URL = "https://atomistic-cookbook.org/_downloads/184bee2944e6bfa0af8580936f246718/thermostats.zip"
SOURCE_PAGE = "https://atomistic-cookbook.org/examples/thermostats/thermostats.html"
SOURCE_ZIP = D / "source" / "thermostats.zip"
SOURCE_PDB = D / "source" / "water_32.pdb"

# the Cookbook start with its molecular centres scaled into the cubic cell of 1.0 g/cm3, copied byte for byte from the 1.0 g/cm3 experiment
START = D / "source" / "water_32_1gcm3.extxyz"
TARGET_DENSITY_G_CM3 = 1.0

BASELINE_PT = LIPS / "models" / "pet-mad-m-v1.6.0.fixed.pt"
LDA_PT = DELTA_RUNS / "delta_lda--omat-s--madcore_shared18000--s42" / "model_best.pt"
PBE_PT = DELTA_RUNS / "delta_pbe--omat-s--madcore_shared18000--s42" / "model_best.pt"
FLOAT64 = D / "verification" / "float64"
BASELINE_OUTPUT = "energy"
DELTA_OUTPUT = "mtt::delta_energy"
DRIVER = D / "scripts" / "metatomic_output_driver.py"

# the sha256 values, source, baseline and driver as verified in the LiPS experiment, the corrections from the shared split runs (best epochs 102 and 101)
EXPECTED_SHA256 = {
    "source zip": "9a5aa16afcb5f5a61c84aff653b1abd93e1e8407c0fa49dc847841b5e30a9fdd",
    "source pdb": "6e87ba4bbb9df6aa330890c95d4573713f7003bee0c182ed8f6ecdb95aa0ee6b",
    "baseline export": "7e578c97879d169a63e6eeb89928246541deea9f1799539c269c939281bb36e1",
    "lda correction": "3ace01e32ef23f4aa5772cd0d15f39133a93e4c8879c026423c7a17b98796778",
    "pbe correction": "aebe2505067209e9d7a3a46caeaa556b286af96535a62bfdf0bbfb8799490d99",
    "driver": "760ccb143b7cef019bec485414f7675f479cc315a2f9133a4c990d4742f66e57",
    "start 1gcm3": "3c2c263c96f1aaa54aca0a03df356d52a3a2d65b48998f169e3e38fdaeb017c8",
}
HASHED_FILES = {
    "source zip": SOURCE_ZIP,
    "source pdb": SOURCE_PDB,
    "baseline export": BASELINE_PT,
    "lda correction": LDA_PT,
    "pbe correction": PBE_PT,
    "driver": DRIVER,
    "start 1gcm3": START,
}

CONDITIONS = {
    "baseline": [
        ("baseline", BASELINE_PT, BASELINE_OUTPUT, 1.0),
    ],
    "lda_corrected": [
        ("baseline", BASELINE_PT, BASELINE_OUTPUT, 1.0),
        ("correction", LDA_PT, DELTA_OUTPUT, -1.0),
    ],
    "pbe_corrected": [
        ("baseline", BASELINE_PT, BASELINE_OUTPUT, 1.0),
        ("correction", PBE_PT, DELTA_OUTPUT, -1.0),
    ],
}
CORRECTION_OF = {"baseline": None, "lda_corrected": "lda", "pbe_corrected": "pbe"}
LABELS = {"baseline": "PET-MAD-1.6-M", "lda_corrected": "approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k)", "pbe_corrected": "approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k)"}
COLORS = {"baseline": "C0", "lda_corrected": "C1", "pbe_corrected": "C2"}

TEMPERATURE_K = 300.0
NATOMS = 96
NMOLECULES = 32
STRIDE = 50
SEEDS = (0, 1, 2, 3, 4)
VELOCITY_SEED_BASE = 20261010
THERMOSTAT_SEED_BASE = 20261020

# folder of each phase under runs, the short pilot test runs share one folder
PHASE_DIRS = {
    "pilot_nve": "pilots/nve",
    "pilot_nve_half_dt": "pilots/nve_half_dt",
    "pilot_nvt": "pilots/nvt",
    "production": "production",
}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)

    return h.hexdigest()


# one model, one energy output, conservative forces as the negative energy gradient, one structure per call
class Potential:
    def __init__(self, path, output, device):
        self.model = mta.load_atomistic_model(str(path)).to(device)
        self.output = output
        self.device = device
        self.dtype = getattr(torch, self.model.capabilities().dtype)
        self.options = mta.ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={output: mta.ModelOutput(quantity="energy", unit="eV", per_atom=False)},
        )

    def __call__(self, numbers, positions, cell, pbc):
        pbc = np.asarray(pbc, dtype=bool)
        cell = np.array(cell, dtype=float)
        cell[~pbc] = 0.0

        positions = torch.tensor(np.asarray(positions, dtype=float), dtype=self.dtype, device=self.device, requires_grad=True)
        system = mta.System(
            types=torch.tensor(np.asarray(numbers), dtype=torch.int32, device=self.device),
            positions=positions,
            cell=torch.tensor(cell, dtype=self.dtype, device=self.device),
            pbc=torch.tensor(pbc, device=self.device),
        )
        compute_requested_neighbors(system, system_length_unit="angstrom", model=self.model)

        outputs = self.model([system], self.options, check_consistency=False)
        energy = outputs[self.output].block().values.sum()
        (gradient,) = torch.autograd.grad(energy, positions)

        return float(energy.detach().double().cpu()), -gradient.detach().double().cpu().numpy()


def atoms_call(potential, atoms):
    return potential(atoms.numbers, atoms.positions, atoms.cell.array, atoms.pbc)


# the scaled start with its 9.8555 A cell, ASE default masses
def source_atoms():
    atoms = read(START, format="extxyz")
    atoms.pbc = True
    return atoms


# the same positions for every seed, Maxwell-Boltzmann velocities at 300 K with the centre-of-mass momentum removed
def water_start(seed):
    atoms = source_atoms()
    rng = np.random.default_rng(VELOCITY_SEED_BASE + seed)

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)

    return atoms


# kinetic temperature over 3N minus 3 degrees of freedom, velocities in ASE units
def kinetic_temperature(masses, velocities):
    kinetic = 0.5 * (masses[:, None] * velocities**2).sum()
    return 2.0 * kinetic / ((3 * len(masses) - 3) * 8.617333262e-5)


# the segments of one i-PI output, oldest backup first, current file last
def segments(run, name):
    backups = sorted(run.glob(f"#{name}#*#"), key=lambda p: int(p.name.split("#")[2]))
    current = [run / name] if (run / name).exists() else []
    return backups + current


# frames of an i-PI extxyz output keyed by step, a later occurrence of a step replaces an earlier one
def read_ipi_xyz(paths, columns):
    frames = {}
    occurrences = 0
    damaged = 0
    symbols = None

    for path in paths:
        lines = path.read_text().splitlines()
        i = 0

        while i < len(lines):
            header = lines[i].strip()
            if not header.isdigit() or i + 1 >= len(lines) or "Step:" not in lines[i + 1]:
                i += 1
                continue

            natoms = int(header)
            rows = [r.split() for r in lines[i + 2:i + 2 + natoms]]

            # a frame cut by a killed segment is skipped, its step is written again after the restart
            if len(rows) < natoms or any(len(r) != columns for r in rows):
                damaged += 1
                i += 2
                continue

            comment = lines[i + 1]
            step = int(re.search(r"Step:\s*(\d+)", comment).group(1))
            lattice = re.search(r'Lattice="([^"]+)"', comment).group(1)
            cell = np.array(lattice.split(), dtype=float).reshape(3, 3)
            values = np.array([r[-3:] for r in rows], dtype=float)

            symbols = [r[0] for r in rows]
            frames[step] = (cell, values)
            occurrences += 1
            i += 2 + natoms

    return frames, symbols, {"occurrences": occurrences, "unique_steps": len(frames), "damaged_frames": damaged}


# property rows keyed by step, a later occurrence replaces an earlier one, rows cut by a kill are skipped
def read_properties(run):
    rows = {}
    names = None
    damaged = 0

    for path in segments(run, "sim.out"):
        text = path.read_text()
        names = [n.split("{")[0] for n in re.findall(r"--> (\S+)", text)]

        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue

            fields = line.split()
            if len(fields) != len(names):
                damaged += 1
                continue

            values = np.array(fields, dtype=float)
            rows[int(values[0])] = values

    steps = sorted(rows)
    table = np.array([rows[s] for s in steps])
    props = {name: table[:, k] for k, name in enumerate(names)}

    return props, {"property_rows": len(steps), "damaged_property_rows": damaged}


# positions, velocities and forces read separately and paired by step only
def read_trajectory(run):
    positions, symbols, pos_check = read_ipi_xyz(segments(run, "sim.pos_0.extxyz"), 4)
    velocities, _, vel_check = read_ipi_xyz(segments(run, "sim.vel_0.extxyz"), 7)
    forces, _, for_check = read_ipi_xyz(segments(run, "sim.for_0.extxyz"), 7)

    paired = sorted(set(positions) & set(velocities) & set(forces))
    every = set(positions) | set(velocities) | set(forces)
    last = paired[-1] if paired else 0
    expected = set(range(0, last + 1, STRIDE))

    check = {
        "positions": pos_check,
        "velocities": vel_check,
        "forces": for_check,
        "paired_frames": len(paired),
        "missing_step_count": len(expected - set(paired)),
        "unpaired_step_count": len(every - set(paired)),
        "steps_not_in_all_three": sorted(every - set(paired))[:20],
        "missing_steps": sorted(expected - set(paired))[:20],
        "unexpected_steps": sorted(set(paired) - expected)[:20],
    }

    steps = np.array(paired, dtype=int)
    cells = np.array([positions[s][0] for s in paired])
    pos = np.array([positions[s][1] for s in paired])
    vel = np.array([velocities[s][1] for s in paired])
    frc = np.array([forces[s][1] for s in paired])

    return {"steps": steps, "symbols": symbols, "cells": cells, "positions": pos, "velocities": vel, "forces": frc}, check
