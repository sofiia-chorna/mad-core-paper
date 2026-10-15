#!/usr/bin/env bash
# Kuma-side convert + featurize for the 12 OPTIMADE Materials Cloud datasets (CC BY 4.0).
#
# PREREQS on Kuma:
#   1. `git pull` the mad-core repo  (needs the structure_id parser in
#      inference/nc_ingest/readers/_extxyz_common.py + the 12 convert.py DATASETS entries).
#   2. transfer/optimade_sources_to_kuma.sh already landed the sources under $SRC_BASE.
#   3. venv-phace (or equivalent) with metatrain/metatomic/vesin + the m292 checkpoint present.
#
# WHAT IT DOES, per dataset (single part_0000.xyz each; jk-9v is 11.3 GB / 3.6M frames):
#   convert   xyz -> $NC_SRC_ROOT/filtered/<ds>/nc_passed/part_0000.parquet   (structures + ids)
#   featurize     -> $OUT_ROOT/filtered/<ds>/part_0000_with_phace.parquet      (LEAN: ids+phace)
# Idempotent: run_inference skips if output exists; convert skips if normalized parquet exists.
#
# rej025: none of these have a 0.25 bucket -> filtered-only is correct (both-variants rule).
#
# AFTER this finishes, rsync results back to CSCS — SAME flow as the 2601-feature
# reconciliation (fps/manifests/kuma_computed_features_to_cscs.txt). Outputs are staged
# under $NROOT/<full-cscs-path>, so one rsync rooted at $NROOT lands them canonically:
#   rsync -aP -e "ssh -J ${USER}@ela.cscs.ch" "$NROOT/" ${USER}@clariden.cscs.ch:/
# -> features at  ${STORE_ROOT}/.../${USER}/model_292/{filtered,filtered_rej_passes_at_025}/<ds>/
#    structures at ${STORE_ROOT}/.../${USER}/nc_sources/{filtered,...}/<ds>/nc_passed/
# Then build fps/runs/optimade_mcloud/manifest.txt from the CSCS feature shards and FPS.
set -euo pipefail

# ---- EDIT these for your Kuma layout ----
REPO="${REPO:-${KUMA_WORK_ROOT}/mad-core}"   # mad-core checkout on Kuma
PY="${PY:-$REPO/venv-phace/bin/python}"                 # venv python on Kuma
MODEL="${MODEL:-$REPO/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt}"
SRC_BASE="${SRC_BASE:-${TRANSFER_ROOT}/mad-core/sources${STORE_ROOT}/dataset_collection/download_optimade/filtered_xyz}"
# rej025 sources (rejected@0.35, passed@0.25), extracted per-dataset from the combined
# 0.25 file by extract_rej025.py. Only mcloudarchive__m0-zg has any (3 structs).
REJ025_BASE="${REJ025_BASE:-${TRANSFER_ROOT}/mad-core/sources${STORE_USER_ROOT}/optimade_rej025}"
# Stage under the SAME Kuma->CSCS convention as the other 2601 Kuma-computed features:
# ${TRANSFER_ROOT}/mad-core/features/ + the full CSCS abs path. Then one rsync
# (NROOT=${TRANSFER_ROOT}/mad-core/features, dest clariden:/) lands them canonically.
NROOT="${NROOT:-${TRANSFER_ROOT}/mad-core/features}"
OUT_ROOT="${OUT_ROOT:-$NROOT${STORE_USER_ROOT}/model_292}"     # features
NC_SRC_ROOT="${NC_SRC_ROOT:-$NROOT${STORE_USER_ROOT}/nc_sources}"  # structures
NL_BACKEND="${NL_BACKEND:-metatomic}"   # Kuma vesin 0.5.8 (verified) -> metatomic OK (NL bug fixed, faster). Use ase if vesin<0.5.8.
# -----------------------------------------

MANIFEST="$(cd "$(dirname "$0")" && pwd)/manifest.txt"
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
[ -x "$PY" ] || { echo "python not found: $PY (set PY=)" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL (set MODEL=)" >&2; exit 2; }
cd "$REPO"

# one dataset+variant: convert (xyz->parquet) then featurize (->_with_phace.parquet)
feat_one() {
    local ds="$1" variant="$2" src="$3"
    [ -f "$src" ] || return 0   # no source for this variant -> skip silently
    local norm="$NC_SRC_ROOT/$variant/$ds/nc_passed/part_0000.parquet"
    local feat="$OUT_ROOT/$variant/$ds/part_0000_with_phace.parquet"
    echo "  [$variant] $src"
    "$PY" -m inference.nc_ingest.convert --dataset "$ds" --inputs "$src" --out "$norm" --strict
    "$PY" inference/run_inference.py --parquet "$norm" --output "$feat" --model "$MODEL" \
          --nl-backend "$NL_BACKEND" --keep-cols configuration_id,dataset_id,nsites
}

n=$(grep -cve '^[[:space:]]*$' "$MANIFEST"); i=0
while read -r ds; do
    [ -z "$ds" ] && continue
    i=$((i+1))
    echo "=== [$i/$n] $ds ==="
    [ -f "$SRC_BASE/$ds/part_0000.xyz" ] || { echo "  SKIP: missing filtered source" >&2; continue; }
    feat_one "$ds" filtered                    "$SRC_BASE/$ds/part_0000.xyz"
    feat_one "$ds" filtered_rej_passes_at_025  "$REJ025_BASE/$ds/part_0000.xyz"   # only m0-zg has this
done < "$MANIFEST"
echo "DONE: $i OPTIMADE MC datasets (+ m0-zg rej025) -> $OUT_ROOT/  (now rsync features + nc_sources back to CSCS)"
