#!/bin/bash
#SBATCH --job-name=cellfps-dense
#SBATCH --account=aa002
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/cell_fps
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps
CKPT_EVERY="${CKPT_EVERY:-2000000}"
total_tasks=$((SLURM_NNODES * 4))

port=29700
for cell in $CELLS; do
    export MASTER_PORT=$((port++))
    srun -ul --ntasks="$total_tasks" --gpus-per-task=1 \
        python run_cell_fps_sharded.py --cell "$cell" --dense \
        --parts-root "$OUT/shuffle/cells" \
        --out-dir "$OUT/fps" \
        --checkpoint-root "$OUT/checkpoints" \
        --checkpoint-every "$CKPT_EVERY"
done
