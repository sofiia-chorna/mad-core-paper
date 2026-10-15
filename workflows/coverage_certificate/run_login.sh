#!/bin/bash
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert

script="$1"
echo "start $(date -Is) on $(hostname) $script"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source ${HOME}/venv-phace/bin/activate && python $script"
echo "finish $(date -Is)"
