#!/bin/bash
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH --mem=800G
#SBATCH --job-name=water18k-dft-probe
#SBATCH --output=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/diagnostics/dft_snapshots/logs/%x-%j.out
#SBATCH --uenv prgenv-gnu/24.11:v1
#SBATCH --view default

set -uo pipefail

HERE=${DEV_ROOT}/phace-atomic/mad-core-valid/md/petmad_m_delta_water_shared18k/diagnostics/dft_snapshots
source $HERE/run_aims_frames.sh

RANKS=72
STEP_MEM=195G

# one r2scan single point first, to show whether the libxc functional works in this build
run_list 1 $HERE/aims/event_during/r2scan

echo "probe finished at $(date +%H:%M:%S)"
