#!/bin/bash
#SBATCH --job-name=nc-OC25-rej025
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/OC25_train/logs/%x_%j.out
# OC25_train rej025 (rejected@0.35 passed@0.25) -> filtered_rej_passes_at_025/OC25_train (FPS both-variants rule)
set -euo pipefail
MAD=${DEV_ROOT}/mad-core
MAN=$MAD/inference/nc_ingest/runs/OC25_train/manifest_rej025.txt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks=$SLURM_NTASKS_PER_NODE --gpus-per-task=1 bash -c '
  L=$((SLURM_LOCALID+1)); ROW=$(sed -n "${L}p" "'"$MAN"'"); [ -z "$ROW" ] && exit 0
  set -- $ROW; bash "'"$MAD"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3"
'
