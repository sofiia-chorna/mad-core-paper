#!/usr/bin/env bash
#SBATCH --job-name=optimade_mc_feat
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=inference/nc_ingest/runs/optimade_mcloud/logs/%x_%j.out
#SBATCH --error=inference/nc_ingest/runs/optimade_mcloud/logs/%x_%j.err
#
# Single-GPU convert + PhACE-m292 featurize for the 12 OPTIMADE Materials Cloud
# datasets (CC BY 4.0) + the m0-zg rej025 (3 structs). run_inference.py is plain
# single-GPU inference (cuda:0) — no NCCL, no CUDA-ext JIT, so no gcc/cuda modules.
#
# jk-9v (11.3 GB / 3.6M structs) dominates wall time (~hours) and is last in the
# manifest; the other 11 finish in seconds-to-minutes. 12h is generous headroom.
# Idempotent: convert skips if normalized parquet exists, run_inference skips if
# the _with_phace output exists — safe to resubmit if it times out.
#
# Submit from the repo root:
#   cd ${KUMA_WORK_ROOT}/mad-core
#   sbatch inference/nc_ingest/runs/optimade_mcloud/submit_kuma.sh
set -euo pipefail

REPO=${KUMA_WORK_ROOT}/mad-core
cd "$REPO"
mkdir -p inference/nc_ingest/runs/optimade_mcloud/logs

source venv-phace/bin/activate

# run_kuma.sh reads its own EDIT-marked defaults (REPO/PY/MODEL/SRC_BASE/REJ025_BASE/
# NC_SRC_ROOT/OUT_ROOT/NL_BACKEND), all already correct for this Kuma layout.
exec inference/nc_ingest/runs/optimade_mcloud/run_kuma.sh
