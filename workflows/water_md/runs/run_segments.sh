#!/bin/bash
set -uo pipefail
D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k

phase=$1
run=$2
gpu=$3
seconds=$4

# each segment is killed without warning after the given seconds, as a walltime kill would, and resumed from its checkpoint
for k in $(seq 1 20); do
  out=$(GPU=$gpu timeout -s KILL $seconds uenv run pytorch/v2.9.1:v2 --view=default -- bash $D/runs/$phase/$run/run.sh)

  if echo "$out" | grep -q "already finished"; then
    echo "$(date +%F_%T) $phase/$run finished after $((k - 1)) segments"
    exit 0
  fi
done

echo "$(date +%F_%T) $phase/$run not finished after 20 segments"
exit 1
