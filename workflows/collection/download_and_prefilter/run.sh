#!/bin/bash

# close contact filter of every shard of one dataset
dataset=$1
out=$2
scale=${3:-0.35}

mkdir -p $out/co_passed $out/co_rejected

for shard in $dataset/co/*.parquet; do
    name=$(basename $shard .parquet)
    python pyykko_filter_parquet.py $shard -o $out/co_passed/${name}_passed.parquet --rejected $out/co_rejected/${name}_rejected.parquet --scale $scale --workers 8
done
