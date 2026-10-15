#!/bin/bash
#SBATCH --job-name=cellfps-megabench
#SBATCH --account=aa002
#SBATCH --nodes=1
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
export CORESETS_FPS_SHARD_XSEL=1

OUT=${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps
BENCH_CELLS="16448618 16595942 16488709"

echo "start $(date -Is)"
port=29510
for cell in $BENCH_CELLS; do
    echo "=== cell $cell $(date -Is)"
    export MASTER_PORT=$((port++))
    srun -ul --ntasks=4 --gpus-per-task=1 \
        python run_cell_fps_sharded.py --cell "$cell" \
        --parts-root "$OUT/shuffle/cells" \
        --out-dir "$OUT/fps" \
        --checkpoint-root "$OUT/checkpoints"
done
echo "finish $(date -Is)"
