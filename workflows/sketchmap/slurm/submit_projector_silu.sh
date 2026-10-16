#!/bin/bash
#SBATCH --job-name=projector_silu
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
python train_projector.py --tag p20d8_silu --embedding-tag p20d8 --d 8 --activation silu
