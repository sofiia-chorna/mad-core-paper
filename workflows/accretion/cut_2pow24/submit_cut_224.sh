#!/bin/bash
#SBATCH --job-name=cut224
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/scripts
source ${HOME}/venv-phace/bin/activate

echo "start $(date -Is)"
python cut_coreset224.py
echo "finish $(date -Is)"

