#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert

until j0=$(sbatch --parsable submit_stage0.sh 2>/dev/null); do echo "$(date -Is) submit blocked, retry in 300s"; sleep 300; done
echo "$(date -Is) stage0 $j0"
jA=$(sbatch --parsable --dependency=afterok:$j0 submit_stageA.sh)
echo "$(date -Is) stageA $jA"
jC=$(sbatch --parsable submit_stageC.sh)
echo "$(date -Is) stageC $jC"
