#!/bin/bash
#SBATCH --job-name=rand2p20
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/scripts
source ${HOME}/venv-phace/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 SAMPLE_NPROC=192
echo "start $(date -Is)"
python random_sample_2p20.py
echo "finish $(date -Is)"
