#!/bin/bash
#SBATCH --job-name=nc-nomad-missing6
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:45:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/nomad/logs/%x_%j.out
# Featurize ONLY the 6 nomad shards the original part-number bug missed (part_0002/3/4).
# Single node, 4 ranks; each rank strides over its share of the 6-line manifest
# (rank r does lines r, r+4, ...). Idempotent (_nc_one.sh skips existing outputs).
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/nomad/manifest_missing6.txt
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
PER_NODE=$SLURM_NTASKS_PER_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks="$PER_NODE" --gpus-per-task=1 bash -c '
    set -e
    NLINES=$(grep -c . "'"$MANIFEST"'")
    i=$SLURM_LOCALID
    while [ $i -lt "$NLINES" ]; do
        LINE=$(( i + 1 ))
        ROW=$(sed -n "${LINE}p" "'"$MANIFEST"'")
        i=$(( i + '"$PER_NODE"' ))
        [ -z "$ROW" ] && continue
        set -- $ROW
        echo "[rank $SLURM_LOCALID] line $LINE: $1 $2 $3 ${4:-}"
        bash "'"$MAD_CORE"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3" "${4:-}"
    done
'
echo "finish: $(date -Is)"
