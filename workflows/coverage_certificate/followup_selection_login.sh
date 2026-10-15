#!/bin/bash
set -uo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/epsilon_exact

until [ -f ../coverage_cert/out/selection_features_build.json ]; do sleep 30; done
echo "selection features ready $(date -Is) on $(hostname)"
cat ../coverage_cert/out/selection_features_build.json
export CUDA_VISIBLE_DEVICES=2
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python recompute_epsilon.py selection'
/usr/bin/python3 check_levels_direct.py selection
echo "followup finish $(date -Is)"
