#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --job-name=ssv3-eval
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
cd ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k
bash ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/fix/verify_fix.sh || exit 1
export EVAL_STAGE=scratch_scaling_v3 EVAL_RUNS=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/runs EVAL_OUT=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/reports/selected_checkpoints EVAL_MTT=${HOME}/venv-metatrain-fix/bin/mtt
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/evaluate.py --batch-size 16
cd ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3
if [ "$PWD" != "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3" ]; then
echo "GUARD: wrong working directory $PWD"
  exit 1
fi
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/evaluate_final.py
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/val_subsets.py
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/scripts/metatrain_metrics.py --round ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3 --prefix scratchv3-s
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/scripts/make_maps.py scaling --round ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3 --prefix scratchv3-s --sizes 5k,10k,20k --scale-from ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/reports/maps/panels_scaling_selected.npz
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/scripts/plot_tails.py --round ${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3 --prefix scratchv3-s --sizes 2k,5k,10k,20k --label "round 3 (+102 isolated-atom references per training set)" --validation "the common 1,250-structure validation set (1,000 random + 250 MAD-CORE)" --extra " Every training set carries the same 102 isolated-atom references in addition to the ordinary structures; the x axis counts the ordinary structures."
uenv run pytorch/v2.9.1:v2 --view=default -- ${HOME}/venv-metatrain-fix/bin/python scripts/check_eval_provenance.py guard || exit 1
echo "EVAL DONE"
