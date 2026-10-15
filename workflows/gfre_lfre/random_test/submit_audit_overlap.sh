#!/bin/bash
#SBATCH --job-name=audit-overlap
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test/logs/%x_%j.out

set -euo pipefail
HERE=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test

source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=4

echo "STARTING $(date)"
cd $HERE
python -u audit_overlap.py
echo "ENDING $(date)"
