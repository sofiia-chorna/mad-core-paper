#!/bin/bash
#SBATCH --job-name=cellfps-rowid
#SBATCH --account=aa002
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=debug
#SBATCH --time=00:22:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps
total_tasks=$((SLURM_NNODES * 16))

echo "start $(date -Is) tasks=$total_tasks"
srun -ul --ntasks="$total_tasks" bash -c \
    "python build_rowid_map.py --plan $OUT/plan.json --out $OUT/rowid \
     --task-id \$SLURM_PROCID --num-tasks $total_tasks"
echo "finish $(date -Is)"
