#!/bin/bash
#SBATCH --job-name=petmad-s-1.5
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

mkdir -p data/mad-1.5/{dpa,pet,mace,uma,univ}
source ~/.bashrc
conda activate pet-feats
python3 scripts/feats/mad15_pet_feats.py mad-s-v1.5.0 LL

echo FINISHED at `date`
