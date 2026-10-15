#!/bin/bash
# Login-node featurizer for OMol25_train rej025 (train_rejected_0.35_passed_0.25).
# The rej025 source is tiny (200 aselmdb x 64KB) so this runs on a Clariden GPU
# login node instead of waiting in the SLURM queue. Single GPU, sequential over
# the same manifest as submit.sh. Idempotent (_nc_one.sh skips existing outputs).
# Run detached:
#   uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
#     'setsid nohup inference/nc_ingest/runs/OMol25_train_rej025/run_login.sh \
#       > inference/nc_ingest/runs/OMol25_train_rej025/logs/login_$(date +%Y%m%d_%H%M%S).out 2>&1 &'
set -uo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/OMol25_train_rej025/manifest.txt
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

echo "=== rej025 login featurize start: $(date -Is) host=$(hostname) GPU=$CUDA_VISIBLE_DEVICES ==="
n=0; ok=0; fail=0
total=$(grep -c . "$MANIFEST")
while read -r DS VAR SRC EXTRA; do
    [ -z "${DS:-}" ] && continue
    n=$((n+1))
    echo "--- [$n/$total] $DS $VAR $(basename "$SRC") ---"
    if bash "$MAD_CORE/inference/nc_ingest/_nc_one.sh" "$DS" "$VAR" "$SRC" "${EXTRA:-}"; then
        ok=$((ok+1))
    else
        fail=$((fail+1)); echo "!!! FAILED: $SRC"
    fi
done < "$MANIFEST"
echo "=== done: $(date -Is)  ok=$ok fail=$fail total=$total ==="
