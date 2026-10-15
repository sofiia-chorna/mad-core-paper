#!/bin/bash
#SBATCH --job-name=phace-featurize-uniontest
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/logs/%x_%j.out

set -euo pipefail
ROOT=${DEV_ROOT}
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "STARTING $(date)"
python -u "$ROOT/paper/figures/mad1.5_gfre_lfre_compare/featurize_phace_union_test.py"
echo "ENDING $(date)"
