#!/bin/bash
set -uo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
variant="$1"
echo "start $variant $(date -Is) on $(hostname)"

for rank in 0 1 2 3; do
    COV_RANK=$rank COV_WORLD=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uenv run pytorch/v2.9.1:v2 --view=default -- bash -c "source ${HOME}/venv-phace/bin/activate && python stageC_full_separation.py $variant" > logs/stageC_login_${variant}_rank${rank}.out 2>&1 &
    sleep 20
done
wait
echo "finish $variant $(date -Is)"
