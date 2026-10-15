#!/bin/bash
#SBATCH --job-name=eval-r2scan
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
ROOT=${DEV_ROOT}
HERE=$ROOT/paper/figures/mad1.5_gfre_lfre_compare/random_test

export OMP_NUM_THREADS=8

echo "STARTING $(date) on $(hostname)"
cd $HERE
$ROOT/phace-atomic/huggingface/ecumetric_work/venv-ecumetric/bin/python -u eval_r2scan.py
echo "ENDING $(date)"
