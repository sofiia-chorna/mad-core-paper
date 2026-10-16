#!/bin/bash
#SBATCH --job-name=fps-madunion-meanstd
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/fps/runs/mad_union_meanstd/logs/%x_%j.out
# full FPS over the MAD union with the mean, std (512-d) phace-atomic descriptor
set -euo pipefail

PROJ=${DEV_ROOT}/phace-atomic
RUN=$PROJ/fps/runs/mad_union_meanstd
VENV_PY=${HOME}/venv-phace/bin/python
mkdir -p "$RUN/logs" "$RUN/checkpoints"
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun -ul --ntasks="$SLURM_NTASKS_PER_NODE" --gpus-per-task=1 \
    "$VENV_PY" "$PROJ/fps/run_fps.py" \
        --manifest         "$PROJ/fps/manifests/mad_union_meanstd.txt" \
        --scaler           "$PROJ/fps/scaler_mad15_phaceatomic_meanstd.pkl" \
        --seed             0 \
        --output           "$RUN/fps.parquet" \
        --checkpoint-dir   "$RUN/checkpoints/ckpt" \
        --checkpoint-every 100000 \
        --notes            "mad_union_meanstd_full"
