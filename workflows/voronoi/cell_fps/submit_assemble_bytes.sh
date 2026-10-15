#!/bin/bash
#SBATCH --job-name=cellfps-idb
#SBATCH --account=aa002
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=debug
#SBATCH --time=04:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps/binary
total_tasks=$((SLURM_NNODES * 8))
srun -ul --ntasks="$total_tasks" bash -c \
  "python assemble_ids.py --out $OUT --phase bytes --task-id \$SLURM_PROCID --num-tasks $total_tasks"
