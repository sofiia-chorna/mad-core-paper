#!/bin/bash
#SBATCH --job-name=cellfps-mad1r
#SBATCH --account=aa002
#SBATCH --array=0-31
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=01:00:00
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
    python rerun_ordinary_mad1.py --bucket-dir $OUT/shuffle/buckets/\$bname \
        --out-dir $OUT/mad1_removal/fps_rerun"
echo "finish $(date -Is)"
