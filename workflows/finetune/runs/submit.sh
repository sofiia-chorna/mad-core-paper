#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=05:00:00
#SBATCH --job-name=ft-aligned-20k

python=${HOME}/venv-metatrain-fix/bin/python
here=$SLURM_SUBMIT_DIR

cd $here/omat-s--madcore_20k_aligned--s42
CUDA_VISIBLE_DEVICES=0 uenv run pytorch/v2.9.1:v2 --view=default -- $python -m metatrain train options.yaml --restart auto >> train.log 2>&1

cd $here/..
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/evaluate.py
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/compare.py
