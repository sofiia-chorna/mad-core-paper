#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --job-name=direct18k-main

# 150 epochs on the shared 18,000 training structures on one GPU, then the evaluation
python=${HOME}/venv-metatrain-fix/bin/python
here=$SLURM_SUBMIT_DIR
run=direct_r2scan--omat-s--madcore_shared18000--s42

cd $here/$run
CUDA_VISIBLE_DEVICES=0 uenv run pytorch/v2.9.1:v2 --view=default -- $python -m metatrain train options.yaml --restart auto >> train.log 2>&1

cd $here/../..
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/evaluate.py --stage main
uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/report.py --stage main
