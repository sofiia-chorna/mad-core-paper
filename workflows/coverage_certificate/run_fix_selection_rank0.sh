#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
export UENV_WARN_MIGRATE=1
echo "rank0 redo start $(date -Is) on $(hostname)"
COV_RANK=0 COV_WORLD=4 COV_DEVICE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageC_full_separation.py selection' > logs/stageC_login_selection_rank0_redo.out 2>&1
echo "rank0 redo exit $? $(date -Is)"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python stageC_levels.py' > logs/stageC_levels_login_redo.out 2>&1
echo "levels redo exit $? $(date -Is)"
