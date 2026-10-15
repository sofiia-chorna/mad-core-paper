#!/bin/bash
#SBATCH --job-name=phace-mad-sol
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/mad_sol/logs/%x_%j.out

set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
SRC=$MAD_CORE/data/mad-next/mad-sol_no_monomers.xyz
OUT_DIR=$MAD_CORE/data/mad-next/model_292
STRUCT=$OUT_DIR/mad_sol.parquet
FEAT=$OUT_DIR/mad_sol_with_phace.parquet
MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
PY=$MAD_CORE/venv-phace/bin/python
[ -f "$SRC" ]   || { echo "missing source xyz: $SRC" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }
mkdir -p "$OUT_DIR"
source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
cd "$MAD_CORE"
echo "start: $(date -Is)"

echo "== [1/2] ingest (xyz -> structures parquet) =="
"$PY" -m inference.nc_ingest.convert --dataset mad_sol --inputs "$SRC" --out "$STRUCT"

echo "== [2/2] featurize (lean: ids + phace predicted energy/forces + embeddings) =="
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$PY" inference/run_inference.py --parquet "$STRUCT" --output "$FEAT" --model "$MODEL" \
        --nl-backend metatomic --batch-size 32 --dtype float32 \
        --keep-cols configuration_id,dataset_id,nsites

N=$("$PY" -c "import pyarrow.parquet as pq; print(pq.read_metadata('$FEAT').num_rows)")
NAN=$("$PY" -c "
import numpy as np, pyarrow.parquet as pq
f=pq.read_table('$FEAT', columns=['phace_features_bb_mean']).column(0)
a=np.stack(f.to_pylist())
print(int((~np.isfinite(a)).any(axis=1).sum()))
")
echo "finish: $(date -Is)  rows=$N  nonfinite_feature_rows=$NAN"
