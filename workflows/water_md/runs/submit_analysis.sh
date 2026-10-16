#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --job-name=water18k-analysis

python=${HOME}/venv-ipi-petmad/bin/python
cd $SLURM_SUBMIT_DIR/..

uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/analyse.py
