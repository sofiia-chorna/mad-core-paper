#!/bin/bash
#SBATCH --job-name=mcloud_tarzip
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/mcloud_archive_tarzip/logs/%x_%j.out

set -euo pipefail

export MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_archive_tarzip

export MANIFEST="${MANIFEST:-$RUN/manifest.txt}"
export WORK="${WORK:-$MAD_CORE/work/mcloud_tierc}"
export EX=$WORK/extract
export STRUCT_DIR=$WORK/structures
export FEAT_DIR=$WORK/features
export MERGED_DIR=$WORK/merged
export MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
mkdir -p "$STRUCT_DIR" "$FEAT_DIR" "$MERGED_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

rm -rf "$MERGED_DIR/.claims"   # clear stale dataset claims from any prior run

# one task per GPU across all nodes; tasks claim datasets dynamically (see worker.sh)
srun -ul --gpus-per-task=1 bash "$RUN/worker.sh"
