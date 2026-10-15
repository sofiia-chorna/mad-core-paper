#!/bin/bash
set -euo pipefail

folder=${DEV_ROOT}/phace-atomic/huggingface
work_dir=${DEV_ROOT}/phace-atomic/huggingface/ecumetric_work
venv=$work_dir/venv-ecumetric
metatrain_clone=${HOME}/src/metatrain

checkpoint=${STORE_ROOT}/mad-core/release/ecumetric_model.ckpt
model=$work_dir/ecumetric_model.pt
stripped_checkpoint=$work_dir/ecumetric_model_stripped.ckpt

export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=1

# the export compiles the metatrain code that is checked out, so record which one
git -C "$metatrain_clone" rev-parse --abbrev-ref HEAD
git -C "$metatrain_clone" rev-parse HEAD
git -C "$metatrain_clone" diff | sha256sum

cd "$work_dir"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    "source $venv/bin/activate && mtt export $checkpoint -m $folder/model_metadata.yaml -o $model"

# the stripped checkpoint gets the same metadata and must give the model exported above
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    "source $venv/bin/activate && python $folder/strip_checkpoint.py \
        --checkpoint $checkpoint \
        --metadata $folder/model_metadata.yaml \
        --reference-model $model \
        --output $stripped_checkpoint"

sha256sum "$checkpoint" "$model" "$stripped_checkpoint"
