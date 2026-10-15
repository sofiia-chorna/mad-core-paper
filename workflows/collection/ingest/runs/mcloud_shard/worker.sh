#!/bin/bash
# One GPU. Dynamic-claim over the input shards in $IN (atomic mkdir claim, so whenever a GPU frees
# up it grabs the next shard -> all GPUs stay busy even though one dataset is much slower). Per shard
# the FULL pipeline runs (filter -> featurize -> merge+ZBL) so the slow per-row ZBL loop is sharded
# across all GPUs too, not left for a single-threaded finalize. Output = one merged shard parquet.
source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"

RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_shard
IN=$WORK/shards/in
OUT=$WORK/shards/merged
CLAIMS=$OUT/.claims
mkdir -p "$OUT" "$CLAIMS"

for shard in "$IN"/*.parquet; do
    stem=$(basename "$shard" .parquet)
    merged="$OUT/${stem}_merged.parquet"
    [ -f "$merged" ] && continue
    mkdir "$CLAIMS/$stem" 2>/dev/null || continue

    filt="$OUT/${stem}_filt.parquet"
    feat="$OUT/${stem}_feat.parquet"

    nrem=$(python "$RUN/shard.py" filter "$shard" "$filt") || { echo "FAIL filter $stem"; continue; }
    [ "${nrem:-0}" -gt 0 ] || { echo "skip $stem (0 rows after filter)"; rm -f "$filt"; continue; }

    python "$MAD_CORE/inference/run_inference.py" \
        --parquet "$filt" --output "$feat" --model "$MODEL" \
        --nl-backend metatomic --batch-size "${BATCH:-8}" --dtype float32 \
        --chunk-rows 20000 --keep-cols configuration_id,dataset_id,nsites \
        || { echo "FAIL featurize $stem"; rm -f "$filt"; continue; }

    python -m inference.nc_ingest.merge_dataset --structures "$filt" --features "$feat" --out "$merged" \
        || { echo "FAIL merge $stem"; rm -f "$filt" "$feat" "$merged"; continue; }

    rm -f "$filt" "$feat"
done
