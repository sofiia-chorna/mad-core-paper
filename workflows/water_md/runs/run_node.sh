#!/bin/bash
set -uo pipefail
D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k

phase=$1
shift

# one run per GPU of this node
pids=()
gpu=0
for run in "$@"; do
  GPU=$gpu uenv run pytorch/v2.9.1:v2 --view=default -- bash $D/runs/$phase/$run/run.sh &
  pids+=($!)
  gpu=$((gpu + 1))
done

i=0
for run in "$@"; do
  wait "${pids[$i]}"
  echo "$(date +%F_%T) $(hostname) $phase/$run run.sh exit $?"
  i=$((i + 1))
done
