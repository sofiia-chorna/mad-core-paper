#!/bin/bash
#SBATCH --job-name=smap-2p20
#SBATCH --account=aa002
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ind_$SLURM_JOB_ID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29517
export OMP_NUM_THREADS=32

echo "start $(date -Is)  nodes=$SLURM_NNODES master=$MASTER_ADDR"
srun --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 bash -c '
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0
python fit_dist.py --tag p20d8 --level 20 --n-components 8 \
    --sigma 1.24 --a-high 4 --b-high 2 --a-low 2 --b-low 2 \
    --wgamma 0.25 --maxiter 400 --tf32'
echo "finish $(date -Is)"
