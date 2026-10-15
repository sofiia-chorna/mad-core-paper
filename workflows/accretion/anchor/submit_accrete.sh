#!/bin/bash
#SBATCH --job-name=accrete-prod
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/logs/%x_%j.out
set -euo pipefail

proj=${DEV_ROOT}/phace-atomic
run=$proj/accrete/runs/madcore_production
n_target=${N_TARGET:?set N_TARGET}
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CORESETS_BISECT_SEED=1 CORESETS_FILL_CHUNK=256 CORESETS_LAZY_FORWARD=1 CORESETS_BISECT_LANES=4 CORESETS_LANES_MIN_STEPS=768

echo "start $(date -Is)  nodes=$SLURM_NNODES  n_target=$n_target"
echo "CORESETS: SEED=${CORESETS_BISECT_SEED:-} LAZY=${CORESETS_LAZY_FORWARD:-} LANES=${CORESETS_BISECT_LANES:-} LANES_MIN=${CORESETS_LANES_MIN_STEPS:-} FILL=${CORESETS_FILL_CHUNK:-} GALLOP_DEEP=${CORESETS_BISECT_GALLOP_DEEP:-unset}"
srun -ul --ntasks="$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))" --gpus-per-task=1 \
    python "$proj/accrete/run_accrete.py" \
        --manifest        "$run/pool_manifest.txt" \
        --scaler          "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl" \
        --anchor-features "${STORE_ROOT}/mad-core/mad_union/anchor.parquet" \
        --anchor-fps      "${STORE_ROOT}/mad-core/mad_union/fps.parquet" \
        --n-target        "$n_target" \
        --out-dir         "$run/out" \
        --checkpoint-dir  "$run/out/ckpt" --checkpoint-every 50000 --resume
echo "finish $(date -Is)"
