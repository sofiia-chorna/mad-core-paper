#!/bin/bash
#SBATCH --job-name=normalize-zbl
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --array=1-315%16
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/normalize_zbl/logs/%A_%a.out


set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/normalize_zbl
VENV_PY=$MAD_CORE/venv-phace/bin/python

FAMILY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN/families.txt")
[ -n "$FAMILY" ] || { echo "no family at index $SLURM_ARRAY_TASK_ID"; exit 0; }

source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=1

APPLY_FLAG=""
[ "${APPLY:-0}" = "1" ] && APPLY_FLAG="--apply"
echo "start $FAMILY ($([ -n "$APPLY_FLAG" ] && echo APPLY || echo DRY-RUN)) $(date -Is)"
srun -ul "$VENV_PY" "$RUN/normalize.py" --family "$FAMILY" --workers "${WORKERS:-96}" $APPLY_FLAG
echo "finish $FAMILY $(date -Is)"
