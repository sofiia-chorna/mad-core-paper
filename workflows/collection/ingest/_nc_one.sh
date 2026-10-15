#!/bin/bash

set -euo pipefail

[ $# -ge 3 ] || { echo "usage: $0 <DATASET> <VARIANT> <SOURCE_FILE> [STEM]" >&2; exit 2; }
DATASET="$1"; VARIANT="$2"; SRC="$3"; STEM_OVERRIDE="${4:-}"

MAD_CORE=${DEV_ROOT}/mad-core
NC_SRC_ROOT=${STORE_USER_ROOT}/nc_sources
OUT_ROOT=${STORE_USER_ROOT}/model_292
MODEL="$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt"

[ -f "$SRC" ]   || { echo "missing source: $SRC" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }


if [ -n "$STEM_OVERRIDE" ]; then STEM="$STEM_OVERRIDE"; else STEM="$(basename "$SRC")"; STEM="${STEM%%.*}"; fi
NORM="$NC_SRC_ROOT/$VARIANT/$DATASET/nc_passed/$STEM.parquet"
FEAT="$OUT_ROOT/$VARIANT/$DATASET/${STEM}_with_phace.parquet"

source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"

if [ -f "$FEAT" ]; then
    echo "[nc_one] feature exists, skip: $FEAT"
    exit 0
fi
if [ ! -f "$NORM" ]; then
    echo "[nc_one] convert $DATASET/$VARIANT $SRC -> $NORM"
    python -m inference.nc_ingest.convert --dataset "$DATASET" --inputs "$SRC" --out "$NORM" \
        --chunk-rows "${CHUNK_ROWS:-20000}"
fi
echo "[nc_one] featurize $NORM -> $FEAT"
# lean feature file: ids only from the input; phace_energy + features + predicted
# forces are always appended

KEEP_COLS="${KEEP_COLS:-configuration_id,dataset_id,nsites}"

python "$MAD_CORE/inference/run_inference.py" \
    --parquet "$NORM" --output "$FEAT" --model "$MODEL" \
    --nl-backend "${NL_BACKEND:-metatomic}" --batch-size "${BATCH_SIZE:-32}" --dtype "${DTYPE:-float32}" \
    --chunk-rows "${CHUNK_ROWS:-50000}" --keep-cols "$KEEP_COLS"
