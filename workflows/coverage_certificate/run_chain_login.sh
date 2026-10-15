#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
export UENV_WARN_MIGRATE=1
echo "chain start $(date -Is) on $(hostname)"

./run_stageC_login.sh selection
echo "selection done $(date -Is)"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageC_levels.py' > logs/stageC_levels_login_$(date +%H%M%S).out 2>&1
echo "levels done $(date -Is)"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageA_assigned_bound.py 4' > logs/stageA_login_full_$(date +%H%M%S).out 2>&1
echo "stageA done $(date -Is)"
echo "chain finish $(date -Is)"
