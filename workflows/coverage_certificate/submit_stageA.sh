#!/bin/bash
#SBATCH --job-name=cov_stageA
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
source ${HOME}/venv-phace/bin/activate

until [ -f out/selection_features_build.json ]; do echo "waiting for selection features $(date -Is)"; sleep 60; done

echo "start $(date -Is)"
python stageA_assigned_bound.py 12
echo "finish $(date -Is)"
