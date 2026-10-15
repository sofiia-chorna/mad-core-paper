#!/bin/bash
#SBATCH --job-name=dedup-pool
#SBATCH --account=aa002
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/logs/%x_%j.out
set -euo pipefail
source ${HOME}/venv-phace/bin/activate
python ${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/dedup_pool.py
