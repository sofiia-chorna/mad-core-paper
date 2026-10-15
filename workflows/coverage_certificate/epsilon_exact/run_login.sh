#!/bin/bash
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact

echo "start $(date -Is) on $(hostname)"
export CUDA_VISIBLE_DEVICES=2
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python recompute_epsilon.py'
echo "finish $(date -Is)"
