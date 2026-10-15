#!/bin/bash
#SBATCH --job-name=nc-feat-dbg
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/logs/slurm/%x_%j.out

set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST="${MANIFEST:?set MANIFEST}"
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
NTASKS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "=== nc-feat-dbg  nodes=$SLURM_NNODES ranks=$NTASKS  manifest=$MANIFEST ==="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks="$NTASKS" --gpus-per-task=1 bash -c '
    set -e
    LINE=$(( SLURM_PROCID + 1 ))
    ROW=$(sed -n "${LINE}p" "'"$MANIFEST"'")
    if [ -z "$ROW" ]; then echo "[rank $SLURM_PROCID] line $LINE empty, exit"; exit 0; fi
    set -- $ROW
    echo "[rank $SLURM_PROCID] line $LINE: dataset=$1 variant=$2 src=$3 stem=${4:-}"
    bash "'"$MAD_CORE"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3" "${4:-}"
'
echo "finish: $(date -Is)"
