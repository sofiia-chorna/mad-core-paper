#!/bin/bash
#SBATCH --job-name=feat-atomic
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/fps/logs/%x_%j.out

# features of one MAD subset with the ecumetric model, sbatch submit_featurize.sh mad_next
subset=$1
proj=${DEV_ROOT}/phace-atomic
mad_core=${DEV_ROOT}/mad-core

case $subset in
    mad15) in=$mad_core/data/mad-1.5/model_292/mad_1.5_with_phace.parquet ;;
    mad_next) in=$mad_core/data/mad-next/model_292/mad_next.parquet ;;
    mad_sol) in=$mad_core/data/mad-next/model_292/mad_sol.parquet ;;
esac

source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -ul python $proj/inference/run_inference.py \
    --parquet $in \
    --model $proj/pre/phace_atomic_ep92.pt \
    --keep-cols configuration_id,dataset_id,atomic_numbers,positions,cell,pbc \
    --output $proj/fps/features/${subset}_phaceatomic_meanstd.parquet \
    --nl-backend metatomic --batch-size 32 --meanstd
