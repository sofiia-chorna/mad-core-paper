#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH --mem=800G
#SBATCH --job-name=water18k-dft-snapshots
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/diagnostics/dft_snapshots/logs/%x-%j.out
#SBATCH --uenv prgenv-gnu/24.11:v1
#SBATCH --view default

set -uo pipefail

HERE=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/diagnostics/dft_snapshots
source $HERE/run_aims_frames.sh

RANKS=72
STEP_MEM=195G

# r2scan first, then pw-lda, then pbe, four slots on one node, finished pairs are skipped on a resubmission
WORKS=()
for functional in r2scan pw-lda pbe; do
  for frame in $(ls -d $HERE/aims/*/ | xargs -n1 basename | sort); do
    WORKS+=($HERE/aims/$frame/$functional)
  done
done

run_list 4 "${WORKS[@]}"

echo "all pairs finished at $(date +%H:%M:%S)"
