#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:45:00
#SBATCH --job-name=ft-aligned-20k-eval
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/finetune_aligned_20k/runs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/finetune_aligned_20k
PY=${HOME}/venv-metatrain-fix/bin/python

# rerun of the evaluation step of runs/submit.sh after job 3628079 failed at checkpoint loading, training is not repeated
cd $D
if ! grep -q "Training finished!" runs/omat-s--madcore_20k_aligned--s42/train.log; then
  echo "GUARD: training not finished"
  exit 1
fi

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/evaluate.py || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/compare.py
echo "ALIGNED FINE-TUNE EVALUATION DONE"
