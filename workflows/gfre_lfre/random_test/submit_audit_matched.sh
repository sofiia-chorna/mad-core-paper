#!/bin/bash
#SBATCH --job-name=audit-matched
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test/logs/%x_%j.out

set -euo pipefail
ROOT=${DEV_ROOT}
HERE=$ROOT/paper/figures/mad1.5_gfre_lfre_compare/random_test

source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=4

echo "STARTING $(date)"
cd $HERE
python -u audit_matched.py
echo "ENDING $(date)"
