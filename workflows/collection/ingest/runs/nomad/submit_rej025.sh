#!/bin/bash
#SBATCH --job-name=nc-nomad-rej025
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/nomad/logs/%x_%j.out
# nomad rej025 = rejected@0.35 but passed@0.25 (10,864 structs) -> features under
# filtered_rej_passes_at_025/nomad, so the nomad FPS pool can include both variants (colabfit-consistent).
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
SRC=${STORE_ROOT}/dataset_collection/download_nomad/filtered_0.25_passed/part_0000.xyz
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks=1 --gpus-per-task=1 bash "$MAD_CORE/inference/nc_ingest/_nc_one.sh" nomad filtered_rej_passes_at_025 "$SRC"
