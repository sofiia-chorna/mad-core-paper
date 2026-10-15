#!/bin/bash
#SBATCH --job-name=vor-assist
#SBATCH --account=aa002
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/scripts
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

total_tasks=$((SLURM_NNODES * 4))
echo "assist start $(date -Is) tasks=$total_tasks (reverse)"
srun -ul --ntasks="$total_tasks" --gpus-per-task=1 bash -c \
    "python assign_shards.py --task-id \$SLURM_PROCID --num-tasks $total_tasks --reverse"
echo "assist finish $(date -Is)"
