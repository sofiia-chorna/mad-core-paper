#!/bin/bash
#SBATCH --job-name=ingest
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# one manifest line per array task, sbatch --array=0-N submit.sh runs/nomad/manifest.txt
manifest=$1
line=$(( SLURM_ARRAY_TASK_ID + 1 ))

source ${DEV_ROOT}/mad-core/venv-phace/bin/activate
cd $SLURM_SUBMIT_DIR/..

set -- $(sed -n "${line}p" $manifest)
dataset=$1
variant=$2
stem=$3
shift 3

out=${STORE_USER_ROOT}/nc_sources/$variant/$dataset/nc_passed/$stem.parquet

python -m ingest.convert --dataset $dataset --inputs "$@" --out $out
