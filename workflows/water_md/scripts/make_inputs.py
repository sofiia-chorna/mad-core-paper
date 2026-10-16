import json
import sys
from pathlib import Path

from ase import units
from ase.io import write
from ipi.utils.units import unit_to_internal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water import (  # noqa: E402
    CONDITIONS,
    D,
    DRIVER,
    PHASE_DIRS,
    STRIDE,
    TEMPERATURE_K,
    THERMOSTAT_SEED_BASE,
    VELOCITY_SEED_BASE,
    kinetic_temperature,
    water_start,
)


RUNS = D / "runs"
STARTS = D / "starts"
VENV = Path("${HOME}/venv-ipi-petmad/bin")
GEOMETRY_LOG_CALLS = 20
THERMOSTAT = {"mode": "svr", "tau_fs": 10.0, "fixcom": True}

PHASES = {
    "pilot_nve": {"ensemble": "nve", "timestep_fs": 0.25, "steps": 8000, "seeds": [0], "checkpoint_stride": 2000},
    "pilot_nve_half_dt": {"ensemble": "nve", "timestep_fs": 0.125, "steps": 16000, "seeds": [0], "checkpoint_stride": 4000},
    "pilot_nvt": {"ensemble": "nvt", "timestep_fs": 0.25, "steps": 8000, "seeds": [0], "checkpoint_stride": 1000},
    "production": {"ensemble": "nvt", "timestep_fs": 0.25, "steps": 200000, "seeds": [0, 1, 2, 3, 4], "checkpoint_stride": 2000},
}
PHASE_CODES = {"pilot_nve": "nve", "pilot_nve_half_dt": "nvh", "pilot_nvt": "nvt", "production": "prd"}
CONDITION_CODES = {"baseline": "b", "lda_corrected": "l", "pbe_corrected": "p"}

INPUT = """<simulation verbosity='low' safe_stride='{checkpoint_stride}'>
  <output prefix='sim'>
    <properties stride='{stride}' filename='out'> [ step, time{{picosecond}}, conserved{{electronvolt}}, temperature{{kelvin}}, temperature(H){{kelvin}}, temperature(O){{kelvin}}, kinetic_md{{electronvolt}}, potential{{electronvolt}}, pressure_md{{bar}}, {components} ] </properties>
    <trajectory filename='pos' stride='{stride}' format='ase'> positions </trajectory>
    <trajectory filename='vel' stride='{stride}' format='ase'> velocities </trajectory>
    <trajectory filename='for' stride='{stride}' format='ase'> forces </trajectory>
    <checkpoint stride='{checkpoint_stride}' filename='chk' overwrite='true'/>
  </output>
  <total_steps> {steps} </total_steps>
  <total_time> 0 </total_time>
  <prng><seed> {seed} </seed></prng>
{forcefields}
  <system>
    <initialize nbeads='1'>
      <file mode='ase'> init.extxyz </file>
      <momenta mode='manual'> {momenta} </momenta>
    </initialize>
    <forces>
{forces}
    </forces>
    <motion mode='dynamics'>
      <fixcom> {fixcom} </fixcom>
      <dynamics mode='{ensemble}'>
        <timestep units='femtosecond'> {timestep_fs} </timestep>
{thermostat}
      </dynamics>
    </motion>
    <ensemble>
      <temperature units='kelvin'> {temperature} </temperature>
    </ensemble>
  </system>
</simulation>
"""

FORCEFIELD = """  <ffsocket name='{name}' mode='unix' pbc='false'>
    <address> {address} </address>
    <latency> 1e-4 </latency>
    <timeout> 600 </timeout>
  </ffsocket>"""

RUN = """#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"

VENV={venv}
GPU=${{GPU:-0}}

# the step stored in an i-PI restart file, empty for an empty file
step_of() {{
  if [ -s "$1" ]; then
    grep -o "<step>[^<]*</step>" "$1" | head -n 1 | tr -dc '0-9'
  fi
}}

for f in RESTART sim.chk; do
  if [ -s "$f" ] && ! $VENV/python -c "import sys, xml.etree.ElementTree as E; E.parse(sys.argv[1])" "$f" 2>/dev/null; then
    mv "$f" "$f.unreadable.$(date +%s)"
    echo "$(date +%F_%T) $f unreadable, set aside" >> segments.log
  fi
done

# resume from whichever of the final and the periodic checkpoint is furthest, through a copy
restart_step=$(step_of RESTART)
chk_step=$(step_of sim.chk)
INPUT=input.xml
done_steps=0

if [ -n "$restart_step" ] && [ "$restart_step" -ge "${{chk_step:-0}}" ]; then
  cp RESTART resume.xml
  INPUT=resume.xml
  done_steps=$restart_step
elif [ -n "$chk_step" ]; then
  cp sim.chk resume.xml
  INPUT=resume.xml
  done_steps=$chk_step
fi

if [ "$done_steps" -ge {steps} ]; then
  echo "already finished at step $done_steps"
  exit 0
fi

# a fresh start over existing outputs would mix two trajectories, so it is refused
if [ "$INPUT" = input.xml ] && [ -s sim.out ]; then
  echo "$(date +%F_%T) outputs exist but no readable checkpoint, not restarted" >> segments.log
  exit 3
fi

# sockets left by an interrupted segment of this run would block i-PI
rm -f {sockets}

echo "$(date +%F_%T) start from $INPUT at step $done_steps on $(hostname) GPU $GPU job ${{SLURM_JOB_ID:-none}}" >> segments.log
$VENV/i-pi $INPUT >> ipi.log 2>&1 &
IPI=$!
sleep 15

{drivers}

wait $IPI
status=$?
echo "$(date +%F_%T) i-PI exit $status" >> segments.log

# the drivers only serve this i-PI process, they are stopped when it exits
kill $(jobs -p) 2>/dev/null
wait
exit $status
"""

DRIVER_LINE = (
    "CUDA_VISIBLE_DEVICES=$GPU $VENV/i-pi-py_driver -u -a {address} -m custom -P {driver} "
    "-o template=init.extxyz,model={model},device=cuda,output={output},geometry_log=geometry_{name}.jsonl,log_calls={log_calls} "
    ">> driver_{name}.log 2>&1 &"
)


def momenta_au(atoms):
    velocities_m_per_s = atoms.get_velocities() * units.fs * 1.0e5
    masses_au = atoms.get_masses()[:, None] * unit_to_internal("mass", "dalton", 1.0)
    return masses_au * velocities_m_per_s * unit_to_internal("velocity", "m/s", 1.0)


def write_start(seed):
    STARTS.mkdir(parents=True, exist_ok=True)
    atoms = water_start(seed)
    path = STARTS / f"water_seed{seed}.extxyz"

    if not path.exists():
        write(path, atoms, format="extxyz")

    return atoms


def make_run(phase, condition, seed):
    settings = PHASES[phase]
    run = RUNS / PHASE_DIRS[phase] / f"{condition}_seed{seed}"

    if (run / "input.xml").exists():
        raise SystemExit(f"{run} already has inputs, not overwritten")

    run.mkdir(parents=True, exist_ok=True)
    atoms = write_start(seed)
    write(run / "init.extxyz", atoms, format="extxyz")

    tag = f"{PHASE_CODES[phase]}{CONDITION_CODES[condition]}{seed}"
    forcefields, forces, drivers = [], [], []

    for name, model, output, weight in CONDITIONS[condition]:
        address = f"pk_{tag}_{name[:4]}"
        forcefields.append(FORCEFIELD.format(name=name, address=address))
        forces.append(f"      <force forcefield='{name}' weight='{weight}'/>")
        drivers.append(DRIVER_LINE.format(
            address=address,
            driver=DRIVER,
            model=model,
            output=output,
            name=name,
            log_calls=GEOMETRY_LOG_CALLS,
        ))

    components = ", ".join(f"pot_component_raw({i}){{electronvolt}}" for i in range(len(CONDITIONS[condition])))

    thermostat = ""
    if settings["ensemble"] == "nvt":
        thermostat = (
            f"        <thermostat mode='{THERMOSTAT['mode']}'>\n"
            f"          <tau units='femtosecond'> {THERMOSTAT['tau_fs']} </tau>\n"
            f"        </thermostat>"
        )

    text = INPUT.format(
        stride=STRIDE,
        checkpoint_stride=settings["checkpoint_stride"],
        components=components,
        steps=settings["steps"],
        seed=THERMOSTAT_SEED_BASE + seed,
        forcefields="\n".join(forcefields),
        forces="\n".join(forces),
        momenta="[ " + ", ".join(f"{x:.12e}" for x in momenta_au(atoms).ravel()) + " ]",
        fixcom=THERMOSTAT["fixcom"],
        ensemble=settings["ensemble"],
        timestep_fs=settings["timestep_fs"],
        thermostat=thermostat,
        temperature=TEMPERATURE_K,
    )
    (run / "input.xml").write_text(text)

    sockets = " ".join(f"/tmp/ipi_pk_{tag}_{name[:4]}" for name, _, _, _ in CONDITIONS[condition])
    (run / "run.sh").write_text(RUN.format(venv=VENV, drivers="\n".join(drivers), steps=settings["steps"], sockets=sockets))
    (run / "run.sh").chmod(0o755)

    meta = {
        "phase": phase,
        "condition": condition,
        "seed": seed,
        **settings,
        "temperature_K": TEMPERATURE_K,
        "thermostat": THERMOSTAT if settings["ensemble"] == "nvt" else None,
        "fixcom": THERMOSTAT["fixcom"],
        "velocity_seed": VELOCITY_SEED_BASE + seed,
        "thermostat_seed": THERMOSTAT_SEED_BASE + seed,
        "forces": [[n, str(m), o, w] for n, m, o, w in CONDITIONS[condition]],
        "initial_temperature_K_3N_minus_3": float(kinetic_temperature(atoms.get_masses(), atoms.get_velocities())),
    }
    (run / "run.json").write_text(json.dumps(meta, indent=1) + "\n")

    return run


# usage make_inputs.py phase, every condition and every seed of the phase
REUSED = {"baseline"}


def main():
    phase = sys.argv[1]

    for condition in CONDITIONS:
        if condition in REUSED:
            continue
        for seed in PHASES[phase]["seeds"]:
            print(make_run(phase, condition, seed))


if __name__ == "__main__":
    main()
