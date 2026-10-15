#!/bin/bash
#SBATCH --job-name=phace-dbgchain
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=450G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/logs/%x_%j.out

set -uo pipefail

PROJ=${DEV_ROOT}/phace-atomic
SELF="$PROJ/submit_debug_chain.sh"
DONE="$PROJ/TRAINING_DONE"
VENV_PY=${HOME}/venv-phace/bin/python
DATA_DIR="$PROJ/data/atomic_combined"
OPTS="$PROJ/options.yaml"
mkdir -p "$PROJ/logs"

if [ -f "$DONE" ]; then echo "TRAINING_DONE present -> chain complete, exiting."; exit 0; fi

# Auto-stop: exit 0 from should_stop.py = val-loss plateau or hard epoch cap -> end chain.
if "$VENV_PY" "$PROJ/pre/should_stop.py"; then touch "$DONE"; echo "stop condition met -> chain ending."; exit 0; fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$SELF" && echo "queued successor" \
    || echo "WARN: successor submit failed (debug queue limit?) -- chain may stop after this job"

export MASTER_ADDR=$(hostname) MASTER_PORT=39591 WORLD_SIZE=$SLURM_NPROCS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8 MPICH_GPU_SUPPORT_ENABLED=0
export TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1 WANDB_DIR="$PROJ/wandb" WANDB_MODE=online

echo "start: $(date -Is)"
srun -ul --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE --gpus-per-task=1 bash -c '
    set -uo pipefail
    if [[ $SLURM_LOCALID -eq 0 ]]; then
        rm -f /tmp/finish-signal
        rm -rf /tmp/train
        cp -rL "'"$DATA_DIR"'" /tmp/train
        touch /tmp/finish-signal
    fi
    while [[ ! -f /tmp/finish-signal ]]; do sleep 1; done
    source ${HOME}/venv-phace/bin/activate
    cd "'"$PROJ"'"
    "'"$VENV_PY"'" -m metatrain train "'"$OPTS"'" --restart auto
'
rc=$?
echo "srun rc=$rc"
if [ "$rc" -eq 0 ]; then touch "$DONE"; echo "marked TRAINING_DONE"; fi
echo "finish: $(date -Is)"
