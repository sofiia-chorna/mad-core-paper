#!/bin/bash

# records whose files are not read directly go through src2extxyz first
source ${DEV_ROOT}/mad-core/venv-phace/bin/activate
cd $(dirname $0)/../..

src=${DEV_ROOT}/mad-core/mc_archive_survey/tierA_dl
work=${WORK}/mcloud_archive_extras

while read -r dataset method; do

    id=${dataset#mcloudarchive__}
    inputs=$(ls $src/${id}___*)

    if [ $method != direct ]; then
        python ingest/preprocess/src2extxyz.py $inputs -o $work/xyz/$dataset.xyz
        inputs=$work/xyz/$dataset.xyz
    fi

    python -m ingest.convert --dataset $dataset --inputs $inputs --out $work/structures/$dataset.parquet

done < $(dirname $0)/manifest.txt
