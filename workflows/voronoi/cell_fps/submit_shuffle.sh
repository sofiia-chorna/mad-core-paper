#!/bin/bash
#SBATCH --job-name=cellfps-shuffle
#SBATCH --account=aa002
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

PLAN=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps/plan.json
total_tasks=$((SLURM_NNODES * 8))

srun -ul --ntasks="$total_tasks" bash -c \
    "python shuffle_chunk.py --plan $PLAN --task-id \$SLURM_PROCID --num-tasks $total_tasks"
