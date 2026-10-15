#!/bin/bash
# After the GPU job: each dataset's merged shards (geometry + features + ZBL columns, already
# computed per shard by worker.sh) just need concatenating into the one final merged parquet.
#   bash finalize.sh > logs/finalize.log 2>&1
set -euo pipefail
export MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_shard
WORK=$MAD_CORE/work/mcloud_rescan
OUT=$WORK/shards/merged
MERGED=$WORK/merged
mkdir -p "$MERGED"

run() { uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "export PYTHONPATH=$MAD_CORE && source $MAD_CORE/venv-phace/bin/activate && $1"; }

for ds in 286v9-64865 kbsns-73a82 nv386-6m953 w7hen-cme26 2y89r-m1q15 vf417-5pb37 nsp7v-2dk91; do
    full=mcloudarchive__$ds
    [ -f "$MERGED/$full.parquet" ] && { echo "skip $full (merged exists)"; continue; }
    run "python $RUN/shard.py concat '$OUT/${full}@s*_merged.parquet' $MERGED/$full.parquet"
done

echo "FINALIZE_DONE merged=$(ls "$MERGED"/*.parquet | wc -l)"
