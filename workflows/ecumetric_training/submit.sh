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

# the data is copied to the local disk of the node first, options.yaml reads it from /tmp/train
cd ${DEV_ROOT}/phace-atomic
source ${HOME}/venv-phace/bin/activate

export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export WORLD_SIZE=$SLURM_NPROCS
export OMP_NUM_THREADS=8
export WANDB_DIR=$PWD/wandb

srun -ul --ntasks=$SLURM_NTASKS --gpus-per-task=1 bash -c '
    if [ $SLURM_LOCALID = 0 ]; then
        cp -rL data/atomic_combined /tmp/train
        touch /tmp/copied
    fi
    while [ ! -f /tmp/copied ]; do
        sleep 1
    done
    python -m metatrain train options.yaml --restart auto
'
