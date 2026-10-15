#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert

until grep -q "stageC " logs/submit_when_open.out; do sleep 60; done
jA=$(grep "stageA " logs/submit_when_open.out | awk '{print $3}')
jC=$(grep "stageC " logs/submit_when_open.out | awk '{print $3}')
jB=$(sbatch --parsable --dependency=afterok:$jA submit_stageB.sh)
jL=$(sbatch --parsable --dependency=afterok:$jC submit_stageC_levels.sh)
echo "$(date -Is) stageB $jB (after $jA)  stageC_levels $jL (after $jC)" >> logs/submit_when_open.out
