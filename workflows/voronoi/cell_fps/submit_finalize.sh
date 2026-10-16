#!/bin/bash
#SBATCH --job-name=cellfps-finalize
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-128}"

python finalize_dups.py
