#!/bin/bash
#SBATCH --job-name=mcloud_extras
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/mcloud_archive_extras/logs/%x_%j.out

set -euo pipefail
export MAD_CORE=${DEV_ROOT}/mad-core
export MANIFEST=$MAD_CORE/inference/nc_ingest/runs/mcloud_archive_extras/manifest.txt
export SRC=$MAD_CORE/mc_archive_survey/tierA_dl

OUT=$MAD_CORE/work/mcloud_archive_extras
export STRUCT_DIR=$OUT/structures
export FEAT_DIR=$OUT/features
export XYZ_DIR=$OUT/xyz
export MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
mkdir -p "$STRUCT_DIR" "$FEAT_DIR" "$XYZ_DIR"

export NTASKS=$SLURM_NTASKS_PER_NODE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1

srun -ul --ntasks="$NTASKS" --gpus-per-task=1 bash -c '
    # no set -e here: one malformed dataset must not kill the whole rank (heterogeneous sources)
    source "$MAD_CORE/venv-phace/bin/activate"
    export PYTHONPATH="$MAD_CORE"
    NLINES=$(wc -l < "$MANIFEST")

    LINE=$(( SLURM_LOCALID + 1 ))
    while [ "$LINE" -le "$NLINES" ]; do
        ROW=$(sed -n "${LINE}p" "$MANIFEST")
        LINE=$(( LINE + NTASKS ))
        [ -z "$ROW" ] && continue
        set -- $ROW; DS="$1"; METHOD="$2"
        RID=${DS#mcloudarchive__}

        NORM="$STRUCT_DIR/$DS.parquet"
        FEAT="$FEAT_DIR/${DS}_with_phace.parquet"
        [ -f "$FEAT" ] && { echo "[extras] skip $DS (feature exists)"; continue; }

        if [ ! -f "$NORM" ]; then
            shopt -s nullglob nocaseglob
            SRCFILES=( "$SRC/${RID}___"* )
            shopt -u nullglob nocaseglob
            [ "${#SRCFILES[@]}" -gt 0 ] || { echo "[extras] WARN no inputs for $DS"; continue; }

            if [ "$METHOD" = "direct" ]; then
                INPUTS=( "${SRCFILES[@]}" )
            else
                STAGE="$XYZ_DIR/$DS.xyz"
                echo "[extras] src2extxyz ($METHOD, ${#SRCFILES[@]} files) -> $STAGE"
                python "$MAD_CORE/inference/nc_ingest/preprocess/src2extxyz.py" "${SRCFILES[@]}" -o "$STAGE"
                INPUTS=( "$STAGE" )
            fi

            echo "[extras] convert $DS ($METHOD, ${#INPUTS[@]} files) -> $NORM"
            python -m inference.nc_ingest.convert --dataset "$DS" --inputs "${INPUTS[@]}" --out "$NORM" --chunk-rows 20000 \
                || { echo "[extras] FAIL convert $DS"; rm -f "$NORM"; continue; }
        fi

        [ -f "$NORM" ] || { echo "[extras] FAIL no structures for $DS"; continue; }
        echo "[extras] featurize -> $FEAT"
        python "$MAD_CORE/inference/run_inference.py" \
            --parquet "$NORM" --output "$FEAT" --model "$MODEL" \
            --nl-backend metatomic --batch-size 32 --dtype float32 \
            --chunk-rows 50000 --keep-cols configuration_id,dataset_id,nsites
    done
'
