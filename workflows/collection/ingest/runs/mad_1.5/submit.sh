#!/bin/bash
#SBATCH --job-name=phace-mad-1.5
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/mad_1.5/logs/%x_%j.out

# Give the MAD-1.5 anchor a real content-addressed configuration_id (MAD-1.5:<hash>) like every
# other dataset, instead of the bare row index. Same pipeline as mad_next: convert (3 split xyz ->
# one structures parquet in train|val|test order, ASE-parsed, energy=get_potential_energy()) then
# featurize (lean: ids + phace predicted energy/forces + embeddings). Re-featurizing also re-derives
# the features with the current (fixed) vesin, independent of the old clariden npy.
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
OUT_DIR=$MAD_CORE/data/mad-1.5/model_292
TRAIN=$MAD_CORE/data/mad-1.5/mad-1.5-r2scan-train.xyz
VAL=$MAD_CORE/data/mad-1.5/mad-1.5-r2scan-val.xyz
TEST=$MAD_CORE/data/mad-1.5/mad-1.5-r2scan-test.xyz
STRUCT=$OUT_DIR/mad_1.5.parquet
FEAT=$OUT_DIR/mad_1.5_with_phace.parquet
MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
PY=$MAD_CORE/venv-phace/bin/python
for f in "$TRAIN" "$VAL" "$TEST" "$MODEL"; do [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }; done
mkdir -p "$OUT_DIR"
source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
cd "$MAD_CORE"
echo "start: $(date -Is)"

echo "== [1/2] ingest (train|val|test xyz -> structures parquet, content-addressed ids) =="
"$PY" -m inference.nc_ingest.convert --dataset mad_1.5 --inputs "$TRAIN" "$VAL" "$TEST" --out "$STRUCT"

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
