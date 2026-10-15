#!/bin/bash
#SBATCH --job-name=build25m
#SBATCH --account=aa002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=750G
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --output=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_25M/logs/%x_%j.out

set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/accrete/runs/madcore_25M/scripts
source ${HOME}/venv-phace/bin/activate
export GATHER_NPROC=64

echo "start $(date -Is)"
cid_index=${DEV_ROOT}/phace-atomic/accrete/runs/madcore_25M/out/cid_index_25M.parquet
if [ -s "$cid_index" ]; then
    echo "[1/8] cid_index_25M (exists, skip)"
else
    echo "[1/8] cid_index_25M"; python build_cid_index_25M.py
fi
echo "[2/8] full_25M (geometry+features)"; python build_full_25M.py
echo "[3/8] category_map_25M";     python build_category_map_25M.py
echo "[4/8] element_map_25M";      python element_map_25M.py
echo "[5/8] decompose_categories"; python decompose_categories_log_25M.py
echo "[6/8] decompose_families";   python decompose_families_25M.py
echo "[7/8] mad_retention";        python plot_mad_retention_25M.py
echo "[8/8] gdb13 count";          python count_gdb13_25M.py
echo "finish $(date -Is)"
