#!/bin/bash
#SBATCH --job-name=nc-OC20-rej025
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/OC20_S2EF_train_all/logs/%x_%j.out
# OC20_S2EF_train_all rej025 (rejected@0.35 passed@0.25, 192 structs from the tar) -> rej025 variant (both-variants rule)
set -euo pipefail
MAD=${DEV_ROOT}/mad-core
ROW=$(cat $MAD/inference/nc_ingest/runs/OC20_S2EF_train_all/manifest_rej025.txt)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks=1 --gpus-per-task=1 bash -c "bash $MAD/inference/nc_ingest/_nc_group_one.sh $ROW"
