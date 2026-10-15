#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --job-name=final-redo-_v3
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
bash ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/fix/verify_fix.sh || exit 1
cd ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/evaluate_final.py
grep -h -o "Package directory: ${HOME}/venv-metatrain[a-z-]*" reports/final_checkpoints/*.log | sort | uniq -c
echo "FINAL REDO DONE"
