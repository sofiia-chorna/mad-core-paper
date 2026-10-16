#!/bin/bash

# close contact filter of every aselmdb file of one folder, 16 files at a time
in=$1
out=$2
rejected=$3
scale=${4:-0.35}

mkdir -p $out $rejected

for file in $in/*.aselmdb; do
    name=$(basename $file)
    python filter_lmdb.py $file -o $out/$name --rejected $rejected/$name --scale $scale &
    if [ $(jobs -r | wc -l) -ge 16 ]; then
        wait -n
    fi
done

wait
