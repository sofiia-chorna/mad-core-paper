#!/bin/bash
# Fused convert+featurize worker for a GROUP of source files -> one shard (OC20: 26k tiny
# extxyz.xz files grouped ~100/shard). usage: _nc_group_one.sh <DATASET> <VARIANT> <STEM> <FILE>...
# 1) convert all FILEs -> one normalized parquet  NC_SRC_ROOT/<variant>/<dataset>/nc_passed/<stem>.parquet
# 2) featurize -> OUT_ROOT/<variant>/<dataset>/<stem>_with_phace.parquet   (lean cols, metatomic)
# Both steps skip if their output exists (idempotent).
set -euo pipefail
[ $# -ge 4 ] || { echo "usage: $0 <DATASET> <VARIANT> <STEM> <FILE>..." >&2; exit 2; }
DATASET="$1"; VARIANT="$2"; STEM="$3"; shift 3
INPUTS=("$@")
MAD_CORE=${DEV_ROOT}/mad-core
NC_SRC_ROOT=${STORE_USER_ROOT}/nc_sources
OUT_ROOT=${STORE_USER_ROOT}/model_292
MODEL="$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt"
NORM="$NC_SRC_ROOT/$VARIANT/$DATASET/nc_passed/$STEM.parquet"
FEAT="$OUT_ROOT/$VARIANT/$DATASET/${STEM}_with_phace.parquet"
source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"
if [ -f "$FEAT" ]; then echo "[nc_group] feature exists, skip: $FEAT"; exit 0; fi
if [ ! -f "$NORM" ]; then
    echo "[nc_group] convert $DATASET/$VARIANT $STEM (${#INPUTS[@]} files) -> $NORM"
    python -m inference.nc_ingest.convert --dataset "$DATASET" --inputs "${INPUTS[@]}" --out "$NORM"
fi
KEEP_COLS="${KEEP_COLS:-configuration_id,dataset_id,nsites}"
echo "[nc_group] featurize $NORM -> $FEAT"
python "$MAD_CORE/inference/run_inference.py" \
    --parquet "$NORM" --output "$FEAT" --model "$MODEL" \
    --nl-backend "${NL_BACKEND:-metatomic}" --batch-size "${BATCH_SIZE:-32}" --dtype "${DTYPE:-float32}" \
    --keep-cols "$KEEP_COLS"
