#!/bin/bash
#SBATCH --job-name=chk-failed
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/bor/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/bor
source ${HOME}/venv-phace/bin/activate
export PYTHONUNBUFFERED=1
python 02_check_failed_in_bor.py
