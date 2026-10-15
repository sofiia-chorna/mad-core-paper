#!/bin/bash
#SBATCH --job-name=phace-atomic
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/logs/%x_%j.out

set -euo pipefail

PROJ=${DEV_ROOT}/phace-atomic
MAD_CORE=${DEV_ROOT}/mad-core
VENV_PY=${HOME}/venv-phace/bin/python
DATA_DIR=${DATA_DIR:-$PROJ/data/atomic_combined}
OPTS=${OPTS:-$PROJ/options.yaml}

mkdir -p "$PROJ/logs" "$PROJ/wandb"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export MPICH_GPU_SUPPORT_ENABLED=0
export TRITON_HOME=/dev/shm/
export CUDA_CACHE_DISABLE=1
export WANDB_DIR=$PROJ/wandb
export WANDB_MODE=${WANDB_MODE:-online}

echo "start: $(date -Is)  data=$DATA_DIR  opts=$OPTS"

srun -ul --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE --gpus-per-task=1 bash -c '
    set -euo pipefail
    if [[ $SLURM_LOCALID -eq 0 ]]; then
        echo "[$(hostname)] local rank 0: copying '"$DATA_DIR"' -> /tmp/train"
        rm -f /tmp/finish-signal
        rm -rf /tmp/train
        cp -rL "'"$DATA_DIR"'" /tmp/train
        echo "[$(hostname)] local rank 0: copy done"
        touch /tmp/finish-signal
    fi
    while [[ ! -f /tmp/finish-signal ]]; do sleep 1; done

    source ${HOME}/venv-phace/bin/activate
    cd "'"$PROJ"'"
    "'"$VENV_PY"'" -m metatrain train "'"$OPTS"'" --restart auto
'
echo "finish: $(date -Is)"
