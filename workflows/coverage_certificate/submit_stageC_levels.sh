#!/bin/bash
#SBATCH --job-name=cov_stageC_levels
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=300G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
source ${HOME}/venv-phace/bin/activate

echo "start $(date -Is)"
python stageC_levels.py
echo "finish $(date -Is)"
