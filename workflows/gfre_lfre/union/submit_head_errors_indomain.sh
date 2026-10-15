#!/bin/bash
#SBATCH --job-name=head-errors-indomain
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/logs/%x_%j.out

set -euo pipefail
ROOT=${DEV_ROOT}
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "STARTING $(date)"
python -u "$ROOT/paper/figures/mad1.5_gfre_lfre_compare/head_errors_indomain.py"
echo "ENDING $(date)"
