#!/bin/bash
# refit with the record populations, step 2: the sketch-map fit, as in July but with the
# record level-20 populations and the July sigma 1.24; stops if the record populations
# change the sigma derived from the pair-weighted histogram
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
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/ind_$SLURM_JOB_ID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29517
export OMP_NUM_THREADS=32
SIGMA_JULY=$(python -c "import json; print(json.load(open('../results/rec/sigma_july.json'))['sigma'])")
SIGMA=$(python -c "import json; print(json.load(open('../results/rec/sigma_record.json'))['sigma'])")
echo "sigma: July populations $SIGMA_JULY, record populations $SIGMA"
# the July fit used sigma 1.24 (the paper value) although the histogram gives 1.26 for both
# populations; keep 1.24 unless the record populations move the derived sigma
if [ "$SIGMA_JULY" != "$SIGMA" ]; then
    echo "the record populations change the derived sigma, not fitting"; exit 1
fi
echo "start $(date -Is)  nodes=$SLURM_NNODES master=$MASTER_ADDR"
# srun itself must not see the uenv libraries (its plugins need the system OpenSSL), the
# tasks get them back
export TASK_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
env -u LD_LIBRARY_PATH srun --ntasks=2 --ntasks-per-node=1 --gpus-per-node=4 bash -c '
export LD_LIBRARY_PATH=$TASK_LD_LIBRARY_PATH
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0
python fit_dist.py --tag p20d8_rec --level 20 --n-components 8 \
    --populations ${DEV_ROOT}/phace-atomic/voronoi/out/analysis/populations_level20_record.npy \
    --sigma 1.24 --a-high 4 --b-high 2 --a-low 2 --b-low 2 \
    --wgamma 0.25 --maxiter 400 --tf32'
echo "finish $(date -Is)"
