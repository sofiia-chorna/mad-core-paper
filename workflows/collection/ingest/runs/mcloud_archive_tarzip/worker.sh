#!/bin/bash
# Runs on ONE GPU (launched once per task by submit.sh). All tasks scan the SAME manifest and grab
# datasets dynamically: a task claims a dataset with an atomic `mkdir` of a lock dir; only one task
# wins each. So whenever a GPU frees up it takes the next un-done dataset -> no GPU sits idle while
# work remains (fixes the static-striding imbalance where one big dataset left other GPUs idle).
#
# Per dataset, three steps:
#   1. convert   extracted files (.db/xyz/cif/POSCAR) -> structures parquet   (archive reader)
#   2. featurize structures -> PhACE features parquet                          (run_inference)
#   3. merge     structures + features + ZBL columns -> ONE merged parquet     (merge_dataset)
# Any failure is logged and skipped (never wedges the task).

source "$MAD_CORE/venv-phace/bin/activate"
export PYTHONPATH="$MAD_CORE"

CLAIMS="$MERGED_DIR/.claims"
mkdir -p "$CLAIMS"

mapfile -t DATASETS < "$MANIFEST"
for line in "${DATASETS[@]}"; do
    ds=${line%%[[:space:]]*}
    [ -z "$ds" ] && continue

    merged="$MERGED_DIR/$ds.parquet"
    [ -f "$merged" ] && continue                 # already done
    mkdir "$CLAIMS/$ds" 2>/dev/null || continue  # another task got it first

    src="$EX/${ds#mcloudarchive__}"
    norm="$STRUCT_DIR/$ds.parquet"
    feat="$FEAT_DIR/${ds}_with_phace.parquet"
    [ -f "$src/.done" ] || { echo "skip $ds (not extracted)"; continue; }

    # 1. extracted files -> structures parquet
    if [ ! -f "$norm" ]; then
        echo "convert $ds"
        python -m inference.nc_ingest.convert --dataset "$ds" --inputs "$src" --out "$norm" --chunk-rows 20000 \
            || { echo "FAIL convert $ds"; rm -f "$norm"; continue; }
    fi
    [ -f "$norm" ] || { echo "skip $ds (no structures)"; continue; }

    # 1b. drop structures the model/pipeline cannot handle, in place, before featurize (same row-filter
    #     spirit as the >1000-atom rule): >1000 atoms (slow/OOM), ghost atoms (Z=0, model rejects them),
    #     degenerate periodic cell (neighbor list fails). Skip the dataset if nothing remains.
    nrem=$(python - "$norm" <<'PY'
import sys

import numpy as np
import pyarrow.parquet as pq

path = sys.argv[1]
t = pq.read_table(path)
n = t.num_rows

keep = np.asarray(t.column("nsites")) <= 1000

an = t.column("atomic_numbers").combine_chunks()
flat = np.asarray(an.values)
off = np.asarray(an.offsets)
nonempty = np.diff(off) > 0
row_min = np.zeros(n, dtype=np.int64)
if nonempty.any():
    row_min[nonempty] = np.minimum.reduceat(flat, off[:-1][nonempty])
keep &= row_min > 0

cells = np.asarray(t.column("cell").to_pylist(), dtype=float).reshape(n, 3, 3)
periodic = np.asarray(t.column("pbc").to_pylist()).any(axis=1)
degenerate = np.abs(np.linalg.det(cells)) < 1e-6
keep &= ~(periodic & degenerate)

t = t.filter(keep)
pq.write_table(t, path)
print(t.num_rows)
PY
) || { echo "FAIL filter $ds"; continue; }
    [ "${nrem:-0}" -gt 0 ] || { echo "skip $ds (no structures <=1000 atoms)"; continue; }
    echo "filtered $ds -> $nrem structures (<=1000 atoms)"

    # 2. structures -> PhACE features
    echo "featurize $ds"
    python "$MAD_CORE/inference/run_inference.py" \
        --parquet "$norm" --output "$feat" --model "$MODEL" \
        --nl-backend metatomic --batch-size "${BATCH:-32}" --dtype float32 \
        --chunk-rows "${CHUNK_ROWS:-50000}" --keep-cols configuration_id,dataset_id,nsites \
        || { echo "FAIL featurize $ds"; continue; }

    # 3. geometry + features + ZBL columns -> one merged parquet
    echo "merge $ds"
    python -m inference.nc_ingest.merge_dataset --structures "$norm" --features "$feat" --out "$merged" \
        || { echo "FAIL merge $ds"; rm -f "$merged"; continue; }

    rm -f "$norm" "$feat"                         # keep only the final merged parquet
done
