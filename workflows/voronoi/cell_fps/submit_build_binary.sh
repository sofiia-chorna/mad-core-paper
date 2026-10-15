#!/bin/bash
#SBATCH --job-name=cellfps-bin
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
OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps/binary
total_tasks=$((SLURM_NNODES * 16))
echo "start $(date -Is) apply=${APPLY:-0}"
if [ "${APPLY:-0}" = "1" ]; then
  srun -ul --ntasks="$total_tasks" bash -c \
    "python build_binary.py --task-id \$SLURM_PROCID --num-tasks $total_tasks --out $OUT --apply"
else
  srun -ul --ntasks="$total_tasks" bash -c \
    "python build_binary.py --task-id \$SLURM_PROCID --num-tasks $total_tasks --out $OUT"
fi
echo "finish $(date -Is)"
