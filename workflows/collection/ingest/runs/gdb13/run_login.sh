#!/bin/bash
# Add the ZBL energy columns to the GDB-13 merged shards ON THE LOGIN NODE, in parallel,
# detached. Used because the SLURM `normal` partition is blocked by maintenance and the
# work is CPU-only: read each work/gdb13/merged shard, compute zbl_energy_per_atom and
# zbl_max_atomic_energy from the stored geometry, rewrite the shard in place.
#
# NUM_TASKS python processes each own shards[i::NUM_TASKS] and parallelise the per-row
# ZBL energy across WORKERS row-group workers. NUM_TASKS*WORKERS stays well under the 288
# login cores so we are a good neighbour on the shared node. Restartable: shards that
# already carry the columns are skipped, so just re-run this if it is interrupted.
#
# Launch (survives logout / login-node rotation):
#   setsid nohup bash inference/nc_ingest/runs/gdb13/run_login.sh \
#       > inference/nc_ingest/runs/gdb13/logs/zbl_energy_login.launch.log 2>&1 < /dev/null &
#
# Monitor:  tail -f inference/nc_ingest/runs/gdb13/logs/zbl_energy_login_*.log
# Impact:   uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
#             "source venv-phace/bin/activate && \
#              python inference/nc_ingest/runs/gdb13/add_zbl_energy.py \
#                --merged-dir work/gdb13/merged --summarize"
set -euo pipefail
export MAD_CORE=${DEV_ROOT}/mad-core
export GDB=$MAD_CORE/work/gdb13
export RUN=$MAD_CORE/inference/nc_ingest/runs/gdb13
export LOGS=$RUN/logs
export NUM_TASKS=8        # parallel processes
export WORKERS=16         # row-group workers per process  (8*16 = 128 of 288 cores)
mkdir -p "$LOGS"

echo "start $(date -Is)  host=$(hostname)  num_tasks=$NUM_TASKS workers=$WORKERS"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c '
  set -euo pipefail
  source "$MAD_CORE/venv-phace/bin/activate"
  for i in $(seq 0 $((NUM_TASKS - 1))); do
    python "$RUN/add_zbl_energy.py" \
        --merged-dir "$GDB/merged" \
        --task-id "$i" --num-tasks "$NUM_TASKS" --workers "$WORKERS" \
        > "$LOGS/zbl_energy_login_$i.log" 2>&1 &
  done
  wait
'
echo "all $NUM_TASKS login tasks finished $(date -Is)"
