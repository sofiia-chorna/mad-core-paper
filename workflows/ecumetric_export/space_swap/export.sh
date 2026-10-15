#!/bin/bash
build_dir=${DEV_ROOT}/phace-atomic/release/build/ecumetric_space_swap
checkpoint=${DEV_ROOT}/phace-atomic/model_92_space.ckpt
staging_dir=$build_dir/staging

mkdir -p "$staging_dir"
cd "$staging_dir" || exit 1

export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=4

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    "source ${HOME}/venv-metatrain/bin/activate && mtt export $checkpoint -o $staging_dir/ecumetric_model.pt"

sha256sum "$checkpoint" "$staging_dir/ecumetric_model.pt"
