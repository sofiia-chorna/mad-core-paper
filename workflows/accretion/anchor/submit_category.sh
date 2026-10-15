#!/bin/bash
#SBATCH --job-name=category-map
#SBATCH --account=aa002
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production/logs/%x_%j.out
set -euo pipefail
source ${HOME}/venv-phace/bin/activate
P=${DEV_ROOT}/phace-atomic
R=$P/accrete/runs/madcore_production
python $R/build_category_map.py
python $P/fps/analysis/decomposition.py --fps $R/out/accrete.parquet --manifest $R/pool_manifest.txt \
    --ks 256 512 1024 2048 4096 8192 \
    --subset-map $R/picks_category_map.json --colors $R/picks_category_colors.json \
    --output $R/decomposition_categories.png
python $R/decompose_categories_log.py
