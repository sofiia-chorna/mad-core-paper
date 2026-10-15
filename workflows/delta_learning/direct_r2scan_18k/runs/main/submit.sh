#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --job-name=direct18k-main
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/direct_r2scan_shared18k/runs/main/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/direct_r2scan_shared18k
LAUNCH=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh
PY=${HOME}/venv-metatrain-fix/bin/python
RUN=direct_r2scan--omat-s--madcore_shared18000--s42

cd $D
if [ "$PWD" != "$D" ]; then
  echo "GUARD: wrong working directory $PWD"
  exit 1
fi

# the smoke stage first, two epochs on 256 structures with the evaluation and report path, so a broken path stops the job early
# the debug partition holds the water MD chain, so smoke and main share this one normal allocation
bash $LAUNCH wait $D/runs/smoke $RUN || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/evaluate.py --stage smoke || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/report.py --stage smoke || exit 1
echo "SMOKE DONE"

# 150 epochs on the shared 18,000 training structures in one allocation, one GPU
# a resubmission after a timeout resumes from the newest checkpoint, checkpoints every 25 epochs
bash $LAUNCH wait $D/runs/main $RUN || exit 1

if ! grep -q "Training finished!" $D/runs/main/$RUN/train.log 2>/dev/null; then
  echo "training not finished in this allocation, resubmit to resume"
  exit 1
fi

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/evaluate.py --stage main || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/report.py --stage main || exit 1
uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/audit_restart.py || exit 1
echo "MAIN DONE"
