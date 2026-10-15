#!/bin/bash
#SBATCH --job-name=mcloud_shard
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=04:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/mcloud_shard/logs/%x_%j.out

set -euo pipefail

export MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_shard
export WORK=$MAD_CORE/work/mcloud_rescan
export MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
export BATCH="${BATCH:-8}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

mkdir -p "$RUN/logs" "$WORK/shards/merged"
rm -rf "$WORK/shards/merged/.claims"   # clear stale shard claims from any prior run

# 16 GPUs (4 nodes x 4), each claims shards dynamically -> the slow dense-MOF shards spread evenly
srun -ul --gpus-per-task=1 bash "$RUN/worker.sh"
