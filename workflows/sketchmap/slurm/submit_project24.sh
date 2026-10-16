#!/bin/bash
#SBATCH --job-name=project24
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
python apply_projector.py --tag p20d8 --d 8
