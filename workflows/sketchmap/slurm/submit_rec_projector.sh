#!/bin/bash
# refit with the record populations, step 3: the SiLU projector on the new map
#SBATCH --job-name=smap-rec-projector
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
export PYTHONUNBUFFERED=1
python train_projector.py --tag p20d8_rec_silu --embedding-tag p20d8_rec --d 8 --activation silu
