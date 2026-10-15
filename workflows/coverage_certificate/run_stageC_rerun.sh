#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
export UENV_WARN_MIGRATE=1
./run_stageC_login.sh release
./run_stageC_login.sh selection
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageC_levels.py' > logs/stageC_levels_rawf64_$(date +%H%M%S).out 2>&1
echo "levels exit $? $(date -Is)"
/usr/bin/python3 build_release_changeset.py > logs/changeset_rawf64_$(date +%H%M%S).out 2>&1
echo "rerun finish $(date -Is)"
