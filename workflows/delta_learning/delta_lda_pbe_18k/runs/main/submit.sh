#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:30:00
#SBATCH --job-name=dshared-main

# the two corrections on two GPUs, then their evaluation on the two test files
python=${HOME}/venv-metatrain-fix/bin/python
here=$SLURM_SUBMIT_DIR
runs="delta_lda--omat-s--madcore_shared18000--s42 delta_pbe--omat-s--madcore_shared18000--s42"

gpu=0
for run in $runs; do
    cd $here/$run
    CUDA_VISIBLE_DEVICES=$gpu uenv run pytorch/v2.9.1:v2 --view=default -- $python -m metatrain train options.yaml --restart auto >> train.log 2>&1 &
    gpu=$((gpu + 1))
done

wait
cd $here/../..

for run in $runs; do
    uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/evaluate.py --stage main --model $run
done

uenv run pytorch/v2.9.1:v2 --view=default -- $python scripts/report.py --stage main
