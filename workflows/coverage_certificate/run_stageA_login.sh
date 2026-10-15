#!/bin/bash
set -uo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
echo "start $(date -Is) on $(hostname)"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageA_assigned_bound.py 6'
echo "finish $(date -Is)"
