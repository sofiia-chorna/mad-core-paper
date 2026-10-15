#!/bin/bash
#SBATCH --job-name=gdb13-zbl
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --array=0-7
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/gdb13/logs/zbl_%A_%a.out
# Phase 2: merge structures + phace features and add ZBL max_force, into ONE parquet per shard.
# Reuses Jonathan's zbl_filter.py merge-mode: for each structures/<stem>.parquet it joins
# features/<stem>_with_phace.parquet by configuration_id and appends max_force + passed (<200).
# Output: work/gdb13/merged/<stem>.parquet  (geometry + phace energy/features/forces +
# max_force + passed). CPU only, resume-safe (skips shards already merged). num-tasks must match array.
# Inputs on scratch (work/gdb13) after the store-quota incident moved them off aa002.
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
GDB=$MAD_CORE/work/gdb13
ZBL=${STORE_ROOT}/zbl_filtering/zbl_filter.py
VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"

srun -ul --ntasks=1 "$VENV_PY" "$ZBL" \
    --root        "$GDB/structures" \
    --pattern     '*.parquet' \
    --feat-root   "$GDB/features" \
    --feat-suffix _with_phace \
    --id-col      configuration_id \
    --out-root    "$GDB/merged" \
    --threshold   200 --cutoff 5.0 \
    --task-id     "$SLURM_ARRAY_TASK_ID" --num-tasks 8 \
    --workers     "$SLURM_CPUS_PER_TASK"
