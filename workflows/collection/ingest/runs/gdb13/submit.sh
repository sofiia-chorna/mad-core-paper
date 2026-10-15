#!/bin/bash
#SBATCH --job-name=gdb13
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/gdb13/logs/%x_%A_%a.out
# GDB-13: SMILES -> 3D (RDKit ETKDGv3+MMFF) -> convert(gdb13, variant=filtered) -> lean PhACE features.
# Fused per chunk: smiles2xyz (CPU) then convert + featurize (GPU); the .xyz is transient (removed).
# ALL outputs land under scratch work/gdb13 (full features ~3TB exceed the aa002 store quota).
# Idempotent (skips a chunk whose feature shard exists). Short --time so the scheduler backfills fast.
#   build a per-tranche chunk manifest (see NOTES.txt), then:
#   sbatch --array=0-$(( (NCHUNKS+3)/4 - 1 ))%96 submit.sh <chunk_manifest>
# manifest rows: "<smi_chunk_path> <stem>"
set -euo pipefail
export MAD_CORE=${DEV_ROOT}/mad-core
MANIFEST="${1:?usage: sbatch --array=0-N%M submit.sh <chunk_manifest>}"
[ -f "$MANIFEST" ] || { echo "missing manifest: $MANIFEST" >&2; exit 2; }
export MANIFEST

# outputs on SCRATCH: full GDB-13 features ~3TB overran the aa002 store quota (incident 2026-06-16);
# store only gets the final coreset. structures/features were moved here from the store.
GDB=$MAD_CORE/work/gdb13
export STRUCT_DIR=$GDB/structures
export FEAT_DIR=$GDB/features
export XYZ_DIR=$GDB/xyz
export MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
mkdir -p "$STRUCT_DIR" "$FEAT_DIR" "$XYZ_DIR"

PER_NODE=$SLURM_NTASKS_PER_NODE
export OFFSET=$(( SLURM_ARRAY_TASK_ID * PER_NODE ))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

srun -ul --ntasks="$PER_NODE" --gpus-per-task=1 bash -c '
    set -e
    LINE=$(( OFFSET + SLURM_LOCALID + 1 ))
    ROW=$(sed -n "${LINE}p" "$MANIFEST"); [ -z "$ROW" ] && exit 0
    set -- $ROW
    SMI="$1"; STEM="$2"
    XYZ="$XYZ_DIR/$STEM.xyz"
    NORM="$STRUCT_DIR/$STEM.parquet"
    FEAT="$FEAT_DIR/${STEM}_with_phace.parquet"
    [ -f "$FEAT" ] && { echo "[gdb13] skip $STEM (feature exists)"; exit 0; }

    source "$MAD_CORE/venv-phace/bin/activate"
    export PYTHONPATH="$MAD_CORE"

    if [ ! -f "$NORM" ]; then
        if [ ! -f "$XYZ" ]; then
            echo "[gdb13] smiles2xyz $SMI -> $XYZ"
            python "$MAD_CORE/inference/nc_ingest/preprocess/smiles2extxyz.py" "$SMI" -o "$XYZ" -j "$SLURM_CPUS_PER_TASK"
        fi
        echo "[gdb13] convert -> $NORM"
        python -m inference.nc_ingest.convert --dataset gdb13 --inputs "$XYZ" --out "$NORM" --chunk-rows 20000
    fi
    echo "[gdb13] featurize -> $FEAT"
    python "$MAD_CORE/inference/run_inference.py" \
        --parquet "$NORM" --output "$FEAT" --model "$MODEL" \
        --nl-backend metatomic --batch-size 32 --dtype float32 \
        --chunk-rows 50000 --keep-cols configuration_id,dataset_id,nsites
    rm -f "$XYZ"
'
