#!/bin/bash
#SBATCH --job-name=fps-10m
#SBATCH --account=aa002
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_10M/logs/%x_%j.out

set -euo pipefail

proj=${DEV_ROOT}/phace-atomic
run=$proj/accrete/runs/madcore_10M

source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun -ul --ntasks="$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))" --gpus-per-task=1 \
    python "$proj/fps/run_fps_continue.py" \
        --manifest       "$run/fps_manifest.txt" \
        --scaler         "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl" \
        --K              "$K_TARGET" \
        --seed-cids      "$proj/accrete/runs/madcore_production/out/accrete.parquet" \
        --checkpoint-dir "$run/out/fpsckpt10m" \
        --checkpoint-every 50000 \
        --output         "$run/out/coreset_10M.parquet"

