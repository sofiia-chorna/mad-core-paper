#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00
#SBATCH --job-name=ssv3-eval

python=${HOME}/venv-metatrain-fix/bin/python
cd $SLURM_SUBMIT_DIR/..

uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/evaluate_final.py
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/val_subsets.py
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/build_populations.py
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/population_figures.py
