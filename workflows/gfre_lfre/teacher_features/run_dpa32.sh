#!/bin/bash
# freeze DPA-3.2-5M branches locally (CPU, fast), then submit inference jobs

set -e
cd ${KUMA_WORK_ROOT}

source ~/.bashrc
conda activate dpa-omol

CHECKPOINT=data/models/dpa/DPA-3.2-5M.pt

dp --pt freeze -c $CHECKPOINT -o data/models/dpa/DPA-3.2-5M-OMat24.pth --model-branch OMat24

dp --pt freeze -c $CHECKPOINT -o data/models/dpa/DPA-3.2-5M-OMol25.pth --model-branch OMol25

OMAT_JOB=$(sbatch --parsable scripts/feats/mad15/run_dpa32_omat24.sh)
echo "OMat24 job: $OMAT_JOB"

OMOL_JOB=$(sbatch --parsable scripts/feats/mad15/run_dpa32_omol25.sh)
echo "OMol25 job: $OMOL_JOB"

echo "done — OMat24 and OMol25 running in parallel"
