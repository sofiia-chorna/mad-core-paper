#!/bin/bash
#SBATCH --job-name=omol
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

cd ${KUMA_WORK_ROOT}

source ~/.bashrc
conda activate dpa-omol

export DP_VARIANT=cuda
export CUDAToolkit_ROOT=$CUDA_HOME
python3 scripts/0_feats/mad15/mad15_dpa_omol_feats.py

