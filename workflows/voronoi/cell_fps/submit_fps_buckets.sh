#!/bin/bash
#SBATCH --job-name=cellfps-buckets
#SBATCH --account=aa002
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%A_%a.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps

echo "start $(date -Is) array=$SLURM_ARRAY_TASK_ID"
srun -ul --ntasks=4 --gpus-per-task=1 bash -c "
    bucket=\$((SLURM_ARRAY_TASK_ID * 4 + SLURM_PROCID))
    printf -v bname 'b%03d' \$bucket
    python run_cell_fps.py --bucket-dir $OUT/shuffle/buckets/\$bname --out-dir $OUT/fps
"
echo "finish $(date -Is)"
