#!/bin/bash
#SBATCH --job-name=optimade-oqmd
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/mad-core/inference/nc_ingest/runs/optimade_oqmd/logs/%x_%j.out
# OPTIMADE oqmd__oqmd (CC BY 4.0, oqmd.org) -- the one remaining license-sure optimade with data.
# 11,189 structures; rejected@0.35 = 0 -> NO rej025 bucket -> filtered-only (both-variants rule
# trivially satisfied). Bulk crystals (pbc TTT) -> metatomic NL (vesin 0.5.8: correct + faster).
# Whole pipeline on CSCS (source is local; tiny): ingest xyz -> featurize (lean) -> FPS (K=N).
# Writes straight into the post-reorg layout: phace_m292/oqmd__oqmd/{structures,features,coreset}.
# (nmd__nmd HELD pending Jonathan's redundancy-with-nomad Q; psdi__pchprop source is empty.)
#   sbatch submit.sh   -- idempotent (convert & run_inference skip if output exists).
set -euo pipefail
MAD_CORE=${DEV_ROOT}/mad-core
SRC=${STORE_ROOT}/dataset_collection/download_optimade/filtered_xyz/oqmd__oqmd/part_0000.xyz
FAM=${STORE_USER_ROOT}/phace_m292/oqmd__oqmd
STRUCT=$FAM/structures/oqmd__oqmd/filtered/part_0000.parquet
FEAT=$FAM/features/oqmd__oqmd/filtered/part_0000_with_phace.parquet
MANIFEST=$MAD_CORE/inference/nc_ingest/runs/optimade_oqmd/manifest.txt
SCALER=$MAD_CORE/mad-1.5/model_292/mad_features_clariden_bb_mean_scaler.pkl
MODEL=$MAD_CORE/models/training_20260401_221450_outputs_2026-04-01_22-15-30_model_292.pt
PY=$MAD_CORE/venv-phace/bin/python
DC=${STORE_ROOT}/dataset_collection
[ -f "$SRC" ] || { echo "missing source xyz: $SRC"; exit 2; }
mkdir -p "$(dirname "$STRUCT")" "$(dirname "$FEAT")" "$FAM/coreset"
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
cd "$MAD_CORE"

echo "== [1/3] ingest (xyz -> structures parquet) =="
"$PY" -m inference.nc_ingest.convert --dataset oqmd__oqmd --inputs "$SRC" --out "$STRUCT" --strict

echo "== [2/3] featurize (lean: ids + phace) =="
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$PY" inference/run_inference.py --parquet "$STRUCT" --output "$FEAT" --model "$MODEL" \
        --nl-backend metatomic --keep-cols configuration_id,dataset_id,nsites

echo "$FEAT" > "$MANIFEST"
N=$("$PY" -c "import pyarrow.parquet as pq; print(pq.read_metadata('$FEAT').num_rows)")
echo "== [3/3] fps (K=N=$N, full nested ordering) =="
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$PY" fps/run_fps.py --manifest "$MANIFEST" --scaler "$SCALER" --K "$N" --seed 0 \
        --output "$FAM/coreset/fps.parquet" \
        --notes "oqmd__oqmd_Kfull_optimade_CCBY_filtered_only"

echo "== README + structures.txt =="
cat > "$FAM/README.txt" <<EOF
# phace_m292/oqmd__oqmd

PhACE m292 features + FPS coreset for the \`oqmd__oqmd\` pool. Everything joins by \`configuration_id\`.

  coreset/fps.parquet                       nested FPS coreset; first K' rows by \`fps_order\` = size-K' coreset
  coreset/manifest.txt, fps.parquet.meta.json
  features/<dataset>/<variant>/*.parquet    256-d phace_features_bb_mean + predicted energy/forces, keyed by configuration_id
  structures/<dataset>/<variant>/*.parquet  geometry + DFT energy/forces/stress (non-colabfit datasets)
  structures.txt                            pointer to colabfit source structures (colabfit datasets)

variant: filtered = passed@0.35 ; filtered_rej_passes_at_025 = rejected@0.35 then passed@0.25

Consume: read coreset/fps.parquet -> configuration_id where fps_order < K' -> join to features/ and/or structures/.
EOF
cat > "$FAM/structures.txt" <<EOF
# Source structures / provenance for family \`oqmd__oqmd\`.
# Join materialized data to coreset by \`configuration_id\`.

oqmd__oqmd  [non-colabfit] -- materialized structures (geometry+DFT, joinable): structures/oqmd__oqmd/
    raw upstream in Jonathan dataset_collection:
        filtered:    $DC/download_optimade/filtered_xyz/oqmd__oqmd
EOF
echo "finish: $(date -Is)  rows=$N"
