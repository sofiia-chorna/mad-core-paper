#!/usr/bin/env bash
set -euo pipefail

cd ${DEV_ROOT}/phace-atomic
work=${DEV_ROOT}/phace-atomic/matcloud-upload/mad-cosmo-v1/production-v2
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${work}/code"
unset PYTHONOPTIMIZE

exec /usr/bin/python3.11 -B "${work}/recovery-20261005/code/recover_production.py" --run
