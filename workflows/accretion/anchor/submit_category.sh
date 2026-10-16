#!/bin/bash
#SBATCH --job-name=category-map
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

cd $SLURM_SUBMIT_DIR
source ${HOME}/venv-phace/bin/activate

python build_category_map.py
