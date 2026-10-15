#!/bin/bash
#SBATCH --job-name=vor-comp
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/voronoi/scripts
source ${HOME}/venv-phace/bin/activate
echo "start $(date -Is)"
python make_composition_l20.py
echo "finish $(date -Is)"
