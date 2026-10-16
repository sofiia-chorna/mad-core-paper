#!/bin/bash
#SBATCH --job-name=fps-family
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# FPS of at most K structures from one family of the collection, sbatch submit_fps_family.sh gdb13 10000000
family=$1
k=${2:-1000000}

proj=${DEV_ROOT}/phace-atomic
store=${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92
core=${STORE_LP133}/mad-core/coresets/phace_atomic_ep92/$family

mkdir -p $core/checkpoints
find $store/$family -name '*.parquet' | sort > $core/manifest.txt

source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

resume=""
if [ -f $core/checkpoints/ckpt_latest.txt ]; then
    resume="--resume-from $core/checkpoints/ckpt"
fi

srun -ul --ntasks=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) --gpus-per-task=1 \
    python $proj/fps/run_fps.py \
        --manifest $core/manifest.txt \
        --scaler $proj/fps/scaler_mad15_phaceatomic_meanstd.pkl \
        --K $k \
        --max-force 200 --max-energy-per-atom 50 --max-atomic-energy 100 \
        --emit-features \
        --output $core/fps.parquet \
        --checkpoint-dir $core/checkpoints/ckpt --checkpoint-every 100000 $resume
