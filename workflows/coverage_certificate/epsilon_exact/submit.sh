#!/bin/bash
#SBATCH --job-name=eps_exact
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact
source ${HOME}/venv-phace/bin/activate

echo "start $(date -Is)"
python recompute_epsilon.py
echo "finish $(date -Is)"
