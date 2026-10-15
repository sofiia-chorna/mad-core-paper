#!/bin/bash
#SBATCH --job-name=cov_stageA_split
#SBATCH --account=ab010
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=800G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/logs/%x_%A_%a.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
source ${HOME}/venv-phace/bin/activate
export UENV_WARN_MIGRATE=1

folders=$(cat out/stageA_split/task_${SLURM_ARRAY_TASK_ID}.txt)

echo "start $(date -Is) on $(hostname) task ${SLURM_ARRAY_TASK_ID}"
python stageA_assigned_bound.py 12 "$folders"
echo "finish $(date -Is)"
