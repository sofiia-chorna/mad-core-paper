#!/bin/bash
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=ssv3d-populations
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1

ROUND=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3
PYTHON=${HOME}/venv-metatrain-fix/bin/python

cd "$ROUND"
if [ "$PWD" != "$ROUND" ]; then
  echo "GUARD: wrong working directory $PWD"
  exit 1
fi

# the two evaluation populations, selected from the frozen manifests only
uenv run pytorch/v2.9.1:v2 --view=default -- "$PYTHON" scripts/build_populations.py || exit 1

# per structure error sums from the stored predictions, checked against metrics_mtt.tsv
uenv run pytorch/v2.9.1:v2 --view=default -- "$PYTHON" scripts/component_stats.py || exit 1

# pooled metrics, learning curves, maps and tables of both versions
uenv run pytorch/v2.9.1:v2 --view=default -- "$PYTHON" scripts/population_figures.py || exit 1

# the vertical paper figure of each version
for population in balanced_4000 union_7847
do
  uenv run pytorch/v2.9.1:v2 --view=default -- "$PYTHON" scripts/paper_figure.py "$population" || exit 1
done

echo "POPULATIONS DONE"
