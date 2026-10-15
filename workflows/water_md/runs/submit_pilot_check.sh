#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:15:00
#SBATCH --job-name=water18k-pilot-check
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k
PY=${HOME}/venv-ipi-petmad/bin/python

# the water checks on the scaled start, then the independent evaluation on CPU, apart from the GPU clients that drove the pilots
cd $D
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/verify_water.py
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/pilot_check.py
