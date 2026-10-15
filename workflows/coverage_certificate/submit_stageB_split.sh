#!/bin/bash
#SBATCH --job-name=cov_stageB_split
#SBATCH --account=ab010
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=450G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
source ${HOME}/venv-phace/bin/activate

n_done=$(ls out/stageA/*.npz | wc -l)
if [ "$n_done" -ne 455 ]; then echo "stage A incomplete: $n_done of 455 npz"; exit 1; fi

echo "start $(date -Is)"
python stageB_bruteforce_closure.py
echo "finish $(date -Is)"
