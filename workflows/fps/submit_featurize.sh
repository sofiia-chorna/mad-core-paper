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
# Re-featurize a MAD subset with the new atom-level PhACE checkpoint -> structure-mean features.
#   SUBSET=mad15|mad_next|mad_sol  sbatch --job-name=feat-$SUBSET --export=ALL,SUBSET=$SUBSET submit_featurize.sh
set -euo pipefail

PROJ=${DEV_ROOT}/phace-atomic
MAD_CORE=${DEV_ROOT}/mad-core
VENV_PY=${HOME}/venv-phace/bin/python
SUBSET=${SUBSET:?set SUBSET=mad15|mad_next|mad_sol}
case "$SUBSET" in
    mad15)    IN="$MAD_CORE/data/mad-1.5/model_292/mad_1.5_with_phace.parquet" ;;
    mad_next) IN="$MAD_CORE/data/mad-next/model_292/mad_next.parquet" ;;
    mad_sol)  IN="$MAD_CORE/data/mad-next/model_292/mad_sol.parquet" ;;
    *) echo "bad SUBSET=$SUBSET"; exit 1 ;;
esac
mkdir -p "$PROJ/fps/logs" "$PROJ/fps/features"
EXTRA=""
SUF=""
if [ "${MEANSTD:-0}" = "1" ]; then EXTRA="$EXTRA --meanstd"; SUF="_meanstd"; fi
OUT="$PROJ/fps/features/${SUBSET}_phaceatomic${SUF}.parquet"
if [ "${SAVE_ATOMIC:-0}" = "1" ]; then EXTRA="$EXTRA --save-atomic $PROJ/fps/features/${SUBSET}_phaceatomic_atomic.npy"; fi
source ${HOME}/venv-phace/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

echo "start $(date -Is)  SUBSET=$SUBSET  IN=$IN  OUT=$OUT  EXTRA=$EXTRA"
srun -ul "$VENV_PY" "$PROJ/inference/run_inference.py" \
    --parquet   "$IN" \
    --model     "$PROJ/pre/phace_atomic_ep92.pt" \
    --keep-cols configuration_id,dataset_id,atomic_numbers,positions,cell,pbc \
    --output    "$OUT" \
    --nl-backend metatomic --batch-size 32 $EXTRA
echo "finish $(date -Is)"
