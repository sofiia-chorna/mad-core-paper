#!/bin/bash
# Login-node prep for the sharded heavy-MOF featurize (CPU only, no GPU). Run detached:
#   setsid nohup bash prep.sh > logs/prep.log 2>&1 &
# Splits the 6 datasets that already have a structures parquet into row-shards, and converts
# nsp7v's 324k cif files into structure shards in parallel. All shards land in $IN as one
# parquet each, ready for the GPU featurize job (submit.sh).
set -euo pipefail
export MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_shard
WORK=$MAD_CORE/work/mcloud_rescan
SD=$WORK/structures
IN=$WORK/shards/in
ROWS=20000
NSP_GROUPS=64
mkdir -p "$IN"

run() { uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "export PYTHONPATH=$MAD_CORE && source $MAD_CORE/venv-phace/bin/activate && $1"; }

# 1. datasets with an existing structures parquet -> row shards (no filter here; worker filters)
for ds in 286v9-64865 kbsns-73a82 nv386-6m953 w7hen-cme26 2y89r-m1q15 vf417-5pb37; do
    full=mcloudarchive__$ds
    run "python $RUN/shard.py split --norm $SD/$full.parquet --ds $full --out-dir $IN --rows $ROWS"
done

# 2. nsp7v: 324k cif files -> structure shards, converted in parallel (strided slices)
NSP=mcloudarchive__nsp7v-2dk91
run "seq 0 $((NSP_GROUPS-1)) | xargs -P 32 -I{} python $RUN/shard.py convert-cifs \
        --extract $WORK/extract/nsp7v-2dk91 --dataset $NSP \
        --out $IN/${NSP}@s{}.parquet --num $NSP_GROUPS --idx {}"

echo "PREP_DONE shards=$(ls "$IN"/*.parquet | wc -l)"
