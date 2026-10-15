#!/bin/bash
#SBATCH --job-name=nc-ODAC_train
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --array=0-159%16
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/ODAC_train/logs/%x_%A_%a.out
# ODAC (odac25) train_passed: 160 aselmdb (1.1T), periodic MOF+adsorbate, E+F. native id=row.unique_id. out=model_292/filtered/ODAC_train/. self-contained: `sbatch submit.sh`. aselmdb reader, metatomic/vesin 0.5.8, lean cols.
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/ODAC_train/manifest.txt
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
PER_NODE=$SLURM_NTASKS_PER_NODE; OFFSET=$(( SLURM_ARRAY_TASK_ID * PER_NODE ))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export CHUNK_ROWS=2000 BATCH_SIZE=16
srun -ul --ntasks="$PER_NODE" --gpus-per-task=1 bash -c '
    set -e; LINE=$(( '"$OFFSET"' + SLURM_LOCALID + 1 ))
    ROW=$(sed -n "${LINE}p" "'"$MANIFEST"'"); [ -z "$ROW" ] && exit 0
    set -- $ROW
    bash "'"$MAD_CORE"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3" "${4:-}"
'
