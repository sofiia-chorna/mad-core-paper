#!/bin/bash
set -uo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
d=$(ls -t logs/stageC_login_driver_*.out | head -1)
until grep -q "finish selection" "$d"; do sleep 60; done
echo "start levels $(date -Is) on $(hostname)"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageC_levels.py'
echo "finish levels $(date -Is)"
