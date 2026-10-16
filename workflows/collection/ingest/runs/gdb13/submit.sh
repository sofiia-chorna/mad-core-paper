#!/bin/bash
#SBATCH --job-name=gdb13
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# one chunk of 2 million SMILES per array task, split -l 2000000 raw/13.smi chunks/
smiles=$1
stem=$(basename ${smiles%.smi})

work=${DEV_ROOT}/mad-core/work/gdb13

source ${DEV_ROOT}/mad-core/venv-phace/bin/activate
cd $SLURM_SUBMIT_DIR/../..

python ingest/preprocess/smiles2extxyz.py $smiles -o $work/xyz/$stem.xyz -j $SLURM_CPUS_PER_TASK

python -m ingest.convert --dataset gdb13 --inputs $work/xyz/$stem.xyz --out $work/structures/$stem.parquet

rm $work/xyz/$stem.xyz
