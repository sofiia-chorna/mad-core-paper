#!/bin/bash
#SBATCH --job-name=gfre-lfre-union
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:15:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/logs/%x_%j.out

set -euo pipefail
ROOT=${DEV_ROOT}
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=4

echo "STARTING $(date)"
python -u "$ROOT/paper/figures/mad1.5_gfre_lfre_compare/gfre_lfre_union.py" \
    --n-total 15000 \
    --n-test 4000 \
    --with-lfre \
    --lfre-n-jobs 32 \
    --pair-workers 7
echo "ENDING $(date)"
