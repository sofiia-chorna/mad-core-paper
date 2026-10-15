#!/bin/bash

set -uo pipefail
cd ${DEV_ROOT}/mad-core
RUNS=inference/nc_ingest/runs
FEAT=${STORE_USER_ROOT}/model_292
LOOP=inference/nc_ingest/submit_nc_debug_loop.sh


ENTRIES=(
  "$RUNS/nomad/manifest.txt|$FEAT/filtered/nomad|nomad"
  "$RUNS/OC25_train/manifest.txt|$FEAT/filtered/OC25_train|OC25train"
  "$RUNS/OC25_val/manifest.txt|$FEAT/filtered/OC25_val|OC25val"
  "$RUNS/ODAC_val/manifest.txt|$FEAT/filtered/ODAC_val|ODACval"
)

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r man outdir tag <<< "$entry"
  nlines=$(grep -c . "$man" 2>/dev/null || echo 0)
  echo "[driver] === $tag : $nlines shards -> $outdir ==="

  for attempt in 1 2 3 4 5 6; do
    have=$(ls "$outdir"/*_with_phace.parquet 2>/dev/null | wc -l)

    if [ "$have" -ge "$nlines" ]; then echo "[driver] $tag complete ($have/$nlines)"; break; fi
    echo "[driver] $tag attempt $attempt: $have/$nlines done -> submit debug loop"
    jid=$(MANIFEST="$PWD/$man" sbatch --parsable --job-name="nc-dbg-$tag" "$LOOP" 2>&1)

    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then echo "[driver] submit failed: $jid ; retry in 120s"; sleep 120; continue; fi
    echo "[driver] $tag job $jid submitted; waiting"

    while squeue -h -j "$jid" -t PENDING,RUNNING,CONFIGURING,COMPLETING 2>/dev/null | grep -q .; do sleep 30; done

  done
  final=$(ls "$outdir"/*_with_phace.parquet 2>/dev/null | wc -l)
  echo "[driver] $tag FINAL $final/$nlines"

done
echo "[driver] all small datasets done $(date -Is)"
