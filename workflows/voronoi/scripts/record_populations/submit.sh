#!/usr/bin/env bash
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=populations-level20-record
#SBATCH --output=${DEV_ROOT}/phace-atomic/voronoi/logs/populations-level20-record-%j.log

set -euo pipefail

cd ${DEV_ROOT}/phace-atomic/voronoi/scripts/record_populations

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

srun --ntasks=1 --cpus-per-task=288 /usr/bin/python3.11 populations_level20_record.py
