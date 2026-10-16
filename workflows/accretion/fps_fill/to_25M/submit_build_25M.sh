#!/bin/bash
#SBATCH --job-name=build-25M
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# the geometry and feature table of the 25 M ordering, then the light table with the categories
cd $SLURM_SUBMIT_DIR
source ${HOME}/venv-phace/bin/activate
export GATHER_NPROC=64

python build_full_25M.py
python build_light_25M.py
