#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:30:00
#SBATCH --job-name=dshared-main
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/delta_lda_pbe_shared/runs/main/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/delta_lda_pbe_shared
LAUNCH=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh
PY=${HOME}/venv-metatrain-fix/bin/python
RUNS="delta_lda--omat-s--madcore_shared18000--s42 delta_pbe--omat-s--madcore_shared18000--s42"
DONE=$D/runs/main/MAIN_DONE

# debug chain, both corrections train in the same allocation on one GPU each, 1 h 30 links with afterany dependencies
# every link resumes each run from the newest checkpoint of its own run directory, a later link finds the marker and leaves
if [ -f "$DONE" ]; then
  echo "already done, nothing to do"
  exit 0
fi

bash $LAUNCH wait $D/runs/main $RUNS || exit 1

for run in $RUNS
do
  if ! grep -q "Training finished!" $D/runs/main/$run/train.log 2>/dev/null; then
    echo "$run not finished in this allocation, the next chain job resumes"
    exit 0
  fi
done

cd $D
if [ "$PWD" != "$D" ]; then
  echo "GUARD: wrong working directory $PWD"
  exit 1
fi

# both corrections on the two test files, one GPU each
pids=()
gpu=0
for run in $RUNS
do
  CUDA_VISIBLE_DEVICES=$gpu uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/evaluate.py --stage main --model $run &
  pids+=($!)
  gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"
do
  wait "$pid" || status=1
done
if [ "$status" != "0" ]; then
  echo "evaluation failed"
  exit 1
fi

uenv run pytorch/v2.9.1:v2 --view=default -- $PY scripts/report.py --stage main || exit 1
touch $DONE
echo "MAIN DONE"
