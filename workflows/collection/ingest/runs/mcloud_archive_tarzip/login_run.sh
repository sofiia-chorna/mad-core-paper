#!/bin/bash
# Finish the rescan featurize+merge on the LOGIN node's 4 GPUs (no SLURM wall). Reuses worker.sh
# (dynamic claim, <=1000 filter), one instance per GPU via CUDA_VISIBLE_DEVICES. Idempotent.
#   setsid nohup bash login_run.sh > logs/rescan_login.log 2>&1 &
export MAD_CORE=${DEV_ROOT}/mad-core
RUN=$MAD_CORE/inference/nc_ingest/runs/mcloud_archive_tarzip
export WORK=$MAD_CORE/work/mcloud_rescan
export EX=$WORK/extract STRUCT_DIR=$WORK/structures FEAT_DIR=$WORK/features MERGED_DIR=$WORK/merged
export MANIFEST=$MAD_CORE/mc_archive_survey/rescan_manifest.txt
export MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
export MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1 OMP_NUM_THREADS=18
mkdir -p "$STRUCT_DIR" "$FEAT_DIR" "$MERGED_DIR"
rm -rf "$MERGED_DIR/.claims"

uenv run pytorch/v2.9.1:v2 --view=default -- bash -c '
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$g bash "'"$RUN"'/worker.sh" &
    done
    wait
'
echo "LOGIN_RUN_DONE"
