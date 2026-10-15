#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --job-name=water18k-pilots
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=8
export VESIN_CUDA_MAX_PAIRS_PER_POINT=16384
export PYTHONDONTWRITEBYTECODE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k
R=$D/runs
PY=${HOME}/venv-ipi-petmad/bin/python
RUNS="lda_corrected_seed0 pbe_corrected_seed0"
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# node 0 runs the NVE pilots at 0.25 fs on GPUs 0 and 1 and the water checks on GPU 3
srun --nodes=1 --ntasks=1 --cpus-per-task=288 --nodelist=${NODES[0]} bash -c "
  CUDA_VISIBLE_DEVICES=3 uenv run pytorch/v2.9.1:v2 --view=default -- $PY $D/scripts/verify_water.py > $D/logs/verify_water_gpu.log 2>&1 &
  bash $R/run_node.sh pilots/nve $RUNS
  wait
" &

# node 1 runs the NVT pilots in segments killed every 200 s
srun --nodes=1 --ntasks=1 --cpus-per-task=288 --nodelist=${NODES[1]} bash -c "
  bash $R/run_segments.sh pilots/nvt lda_corrected_seed0 1 200 &
  bash $R/run_segments.sh pilots/nvt pbe_corrected_seed0 2 200 &
  wait
" &

# node 2 runs the NVE pilots at 0.125 fs
srun --nodes=1 --ntasks=1 --cpus-per-task=288 --nodelist=${NODES[2]} bash $R/run_node.sh pilots/nve_half_dt $RUNS &

wait
echo "WATER PILOTS DONE"
