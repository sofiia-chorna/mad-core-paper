#!/bin/bash
# the fit of the map with the record populations of the level 20 cells
#SBATCH --job-name=smap-rec-fit
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29517
export OMP_NUM_THREADS=32
# tasks get them back
export TASK_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
env -u LD_LIBRARY_PATH srun --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 bash -c '
export LD_LIBRARY_PATH=$TASK_LD_LIBRARY_PATH
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0
python fit_dist.py --tag p20d8_rec --level 20 --n-components 8 \
    --populations ${DEV_ROOT}/phace-atomic/voronoi/out/analysis/populations_level20_record.npy \
    --sigma 1.24 --a-high 4 --b-high 2 --a-low 2 --b-low 2 \
    --wgamma 0.25 --maxiter 400 --tf32'
