#!/bin/bash
#SBATCH --job-name=dpa31-omat24
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 12
#SBATCH --time 12:00:00 
#SBATCH --gres=gpu:1
#SBATCH --get-user-env
#SBATCH --mem=64G
#SBATCH --account=cosmo
#SBATCH --partition=h100

echo STARTING AT `date`

cd ${KUMA_WORK_ROOT}

source ~/.bashrc
conda activate dpa

export DP_VARIANT=cuda
export CUDAToolkit_ROOT=$CUDA_HOME
python3 scripts/feats/mad15/mad15_dpa31_feats.py Omat24 LL

echo FINISHED at `date`
