#!/bin/bash
#SBATCH --job-name=nc-Alexandria
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/Alexandria/logs/%x_%j.out
# Alexandria pbesol+pressure PhACE features. 12 passed files -> one 3-node debug job
# (12 ranks, ~3 min, within debug's 90 node-min / 1:30 limits). metatomic/vesin 0.5.8.
# pbesol_filtered + pressure_filtered (*_passed.json.bz2); single "filtered" variant.
# self-contained: `sbatch submit.sh`.
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/Alexandria/manifest.txt
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
NTASKS=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "=== nc-Alexandria  nodes=$SLURM_NNODES ranks=$NTASKS ==="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
srun -ul --ntasks="$NTASKS" --gpus-per-task=1 bash -c '
    set -e
    LINE=$(( SLURM_PROCID + 1 ))
    ROW=$(sed -n "${LINE}p" "'"$MANIFEST"'")
    [ -z "$ROW" ] && { echo "[rank $SLURM_PROCID] line $LINE empty, exit"; exit 0; }
    set -- $ROW
    echo "[rank $SLURM_PROCID] line $LINE: $1 $2 $3 ${4:-}"
    bash "'"$MAD_CORE"'/inference/nc_ingest/_nc_one.sh" "$1" "$2" "$3" "${4:-}"
'
echo "finish: $(date -Is)"
