#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=05:00:00
#SBATCH --job-name=ft-aligned-20k
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/finetune_aligned_20k/runs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/finetune_aligned_20k
LAUNCH=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh
PY=${HOME}/venv-metatrain-fix/bin/python

# input checks again on the compute node, the run starts only if they pass
cd $D
if [ "$PWD" != "$D" ]; then
  echo "GUARD: wrong working directory $PWD"
  exit 1
fi
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/check_inputs.py || exit 1

# 150 epochs in one uninterrupted allocation on one GPU, corrected package verified by the launcher
bash $LAUNCH wait $D/runs omat-s--madcore_20k_aligned--s42 || exit 1

# selected checkpoint exported and evaluated on both frozen tests, then the three model comparison
cd $D
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/evaluate.py || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/compare.py
echo "ALIGNED FINE-TUNE DONE"
