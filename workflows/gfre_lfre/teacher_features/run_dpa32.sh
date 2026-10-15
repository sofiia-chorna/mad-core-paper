#!/bin/bash
# Freeze DPA-3.2-5M branches locally (CPU, fast), then submit inference jobs.
#
# Usage: bash scripts/feats/mad15/run_dpa32.sh

set -e
cd ${KUMA_WORK_ROOT}

source ~/.bashrc
conda activate dpa-omol

CHECKPOINT=data/models/dpa/DPA-3.2-5M.pt

echo "--- freezing OMat24 branch ---"
dp --pt freeze -c $CHECKPOINT -o data/models/dpa/DPA-3.2-5M-OMat24.pth --model-branch OMat24

echo "--- freezing OMol25 branch ---"
dp --pt freeze -c $CHECKPOINT -o data/models/dpa/DPA-3.2-5M-OMol25.pth --model-branch OMol25

echo "--- submitting inference jobs ---"
OMAT_JOB=$(sbatch --parsable scripts/feats/mad15/run_dpa32_omat24.sh)
echo "OMat24 job: $OMAT_JOB"

OMOL_JOB=$(sbatch --parsable scripts/feats/mad15/run_dpa32_omol25.sh)
echo "OMol25 job: $OMOL_JOB"

echo "done — OMat24 and OMol25 running in parallel"
