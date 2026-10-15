#!/bin/bash
#SBATCH --job-name=build-pool-prod
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/logs/%x_%j.out

set -euo pipefail

source ${HOME}/venv-phace/bin/activate

python ${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/build_pool.py
