#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --job-name=water18k-prod
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/logs/%x-%j.out
set -uo pipefail
export UENV_WARN_MIGRATE=1
export OMP_NUM_THREADS=8
export VESIN_CUDA_MAX_PAIRS_PER_POINT=16384
export PYTHONDONTWRITEBYTECODE=1

D=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k
PY=${HOME}/venv-ipi-petmad/bin/python
CHAIN=$D/logs/production_chain.log
MAX_SEGMENTS=14

INCOMPLETE=($(uenv run pytorch/v2.9.1:v2 --view=default -- $PY $D/scripts/check_runs.py production --incomplete))
STEPS=$(uenv run pytorch/v2.9.1:v2 --view=default -- $PY $D/scripts/check_runs.py production --steps)

# the chain ends here when every run is complete, and the analysis job is queued
if [ ${#INCOMPLETE[@]} -eq 0 ]; then
  echo "$(date +%F_%T) job $SLURM_JOB_ID all production runs complete, chain ends" | tee -a $CHAIN
  analysis=$(sbatch --parsable $D/runs/submit_analysis.sh)
  echo "$(date +%F_%T) job $SLURM_JOB_ID queued analysis job $analysis" | tee -a $CHAIN
  exit 0
fi

# the next segment is queued first, unless the last segment made no progress or the segment cap is reached
segments=$(grep -c "segment start" $CHAIN 2>/dev/null)
previous=$(grep "segment start" $CHAIN 2>/dev/null | tail -n 1 | awk '{print $NF}')
echo "$(date +%F_%T) job $SLURM_JOB_ID segment start, ${#INCOMPLETE[@]} incomplete runs, steps $STEPS" >> $CHAIN
stop_reason=""

if [ -n "$previous" ] && [ "$STEPS" -le "$previous" ]; then
  stop_reason="no progress since the last segment"
elif [ "$segments" -ge "$MAX_SEGMENTS" ]; then
  stop_reason="segment cap $MAX_SEGMENTS reached"
elif next=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $D/runs/submit_debug_production.sh); then
  echo "$(date +%F_%T) job $SLURM_JOB_ID queued next segment $next" >> $CHAIN
else
  echo "$(date +%F_%T) job $SLURM_JOB_ID could not queue the next segment" | tee -a $CHAIN
fi

# a chain stopped early still analyses what exists, incomplete runs are reported as such
if [ -n "$stop_reason" ]; then
  echo "$(date +%F_%T) job $SLURM_JOB_ID $stop_reason, no segment run, analysis queued" | tee -a $CHAIN
  analysis=$(sbatch --parsable $D/runs/submit_analysis.sh)
  echo "$(date +%F_%T) job $SLURM_JOB_ID queued analysis job $analysis" | tee -a $CHAIN
  exit 0
fi

# one run per GPU across the three nodes, ten corrected runs, the baseline runs are complete symlinks
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
for n in "${!NODES[@]}"; do
  share=("${INCOMPLETE[@]:$((4 * n)):4}")
  if [ ${#share[@]} -gt 0 ]; then
    srun --nodes=1 --ntasks=1 --cpus-per-task=288 --nodelist=${NODES[$n]} bash $D/runs/run_node.sh production "${share[@]}" &
  fi
done
wait
