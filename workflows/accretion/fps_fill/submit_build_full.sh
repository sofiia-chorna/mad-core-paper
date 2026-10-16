#!/bin/bash
#SBATCH --job-name=full10m
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_10M/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/madcore_10M
source ${HOME}/venv-phace/bin/activate

python build_full_10M.py
