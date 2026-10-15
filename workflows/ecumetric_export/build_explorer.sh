#!/bin/bash
set -euo pipefail

folder=${DEV_ROOT}/phace-atomic/huggingface
work_dir=${DEV_ROOT}/phace-atomic/huggingface/ecumetric_work
venv=$work_dir/venv-ecumetric
release=${STORE_ROOT}/mad-core/release

export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=4
export PYTHONWARNINGS=ignore

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    "source $venv/bin/activate && python $folder/build_explorer.py \
        --ecumetric-model $release/ecumetric_model.pt \
        --scaler $release/ecumetric_scaler.npz \
        --projector $release/ecumetric_sketchmap.pt \
        --metadata $folder/explorer_metadata.yaml \
        --output $work_dir/ecumetric_explorer.pt"

sha256sum "$release/ecumetric_model.pt" "$work_dir/ecumetric_explorer.pt"
