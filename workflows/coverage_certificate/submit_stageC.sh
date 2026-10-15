#!/bin/bash
#SBATCH --job-name=cov_stageC
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
source ${HOME}/venv-phace/bin/activate

until [ -f out/selection_features_build.json ]; do echo "waiting for selection features $(date -Is)"; sleep 60; done

echo "start $(date -Is)"
srun --ntasks=4 python stageC_full_separation.py release
srun --ntasks=4 python stageC_full_separation.py selection
echo "finish $(date -Is)"
