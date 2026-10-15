#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --job-name=ssv3-node_a
#SBATCH --nice=2000
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/runs/%x-%j.out
set -uo pipefail
# every allocation queues its successor at start (afterany) while a run of this node is unfinished, and resumes
# every unfinished run from its newest checkpoint; per run locks keep two allocations off the same run
G=${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling_v3/runs
unfinished=0
for run in scratchv3-s--madcore_2k--s42 scratchv3-s--madcore_5k--s42 scratchv3-s--madcore_10k--s42 scratchv3-s--madcore_20k--s42; do
  grep -q "Training finished!" $G/$run/train.log 2>/dev/null || unfinished=1
done
if [ "$unfinished" = "1" ]; then
  sbatch --dependency=afterany:$SLURM_JOB_ID $G/submit_node_a.sh >> $G/chain_node_a.log 2>&1 || true
fi
${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts/launch_runs_fix.sh wait $G scratchv3-s--madcore_2k--s42 scratchv3-s--madcore_5k--s42 scratchv3-s--madcore_10k--s42 scratchv3-s--madcore_20k--s42
