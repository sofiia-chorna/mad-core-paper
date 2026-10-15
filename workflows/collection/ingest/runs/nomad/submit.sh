#!/bin/bash
#SBATCH --job-name=nc-nomad
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --array=0-25%16
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/nomad/logs/%x_%A_%a.out
# nomad PhACE features: filtered (passed) shards = download_nomad/filtered_xyz/<shard>/part_0001.xyz
# 104 shards, ~5.4M structs, geometry-only (no DFT E/F), mixed pbc. metatomic NL (vesin 0.5.8).
# self-contained like fps/runs/<pool>/submit.sh -> just `sbatch submit.sh`.
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/nomad/manifest.txt
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
PER_NODE=$SLURM_NTASKS_PER_NODE
OFFSET=$(( SLURM_ARRAY_TASK_ID * PER_NODE ))
echo "=== nc-nomad task $SLURM_ARRAY_TASK_ID  lines $((OFFSET+1))..$((OFFSET+PER_NODE)) ==="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks="$PER_NODE" --gpus-per-task=1 bash -c '
    set -e
    LINE=$(( '"$OFFSET"' + SLURM_LOCALID + 1 ))
    ROW=$(sed -n "${LINE}p" "'"$MANIFEST"'")
    [ -z "$ROW" ] && { echo "[rank $SLURM_LOCALID] line $LINE empty, exit"; exit 0; }
    set -- $ROW
    echo "[rank $SLURM_LOCALID] line $LINE: $1 $2 $3 ${4:-}"
    bash "'"$MAD_CORE"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3" "${4:-}"
'
echo "finish: $(date -Is)"
