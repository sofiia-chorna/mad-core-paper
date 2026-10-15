#!/bin/bash
# refit with the record populations, step 1: check the level-20 populations and derive sigma
#SBATCH --job-name=smap-rec-sigma
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/logs/%x_%j.out
set -euo pipefail
cd ${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/src
source ${HOME}/venv-phace/bin/activate
export PYTHONUNBUFFERED=1
A=${DEV_ROOT}/phace-atomic/voronoi/out/analysis
python - <<'PY'
import numpy as np
A = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
n = np.load(f"{A}/populations_level20_record.npy"); o = np.load(f"{A}/populations_level20.npy")
print("record level-20:", n.shape, n.dtype, "sum", int(n.sum()), "row0", int(n[0]), "min", int(n.min()),
      "all <= July", bool((n <= o).all()), "rows differing", int((n != o).sum()))
assert n.shape == (2 ** 20,) and n.dtype == np.int64 and int(n.sum()) == 1_694_434_380
assert int(n[0]) == 6275 and n.min() > 0 and (n <= o).all() and int((n != o).sum()) == 126_851
PY
python derive_params_2p20.py --sigma-out ../results/rec/sigma_july.json
python derive_params_2p20.py --populations $A/populations_level20_record.npy --sigma-out ../results/rec/sigma_record.json
