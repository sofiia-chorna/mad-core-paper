#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --job-name=ssv3-node_b

# four models on the four GPUs of one node, resumed from the last checkpoint when requeued
python=${HOME}/venv-metatrain-fix/bin/python
runs=$SLURM_SUBMIT_DIR

gpu=0
for run in scratchv3-s--random_2k--s42 scratchv3-s--random_5k--s42 scratchv3-s--random_10k--s42 scratchv3-s--random_20k--s42; do
    cd $runs/$run
    CUDA_VISIBLE_DEVICES=$gpu uenv run pytorch/v2.9.1:v2 --view=default -- $python -m metatrain train options.yaml --restart auto >> train.log 2>&1 &
    gpu=$((gpu + 1))
done

wait
