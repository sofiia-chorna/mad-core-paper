#!/bin/bash
cd ${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert
export UENV_WARN_MIGRATE=1 COV_MMAP=1
echo "probe start $(date -Is) on $(hostname)"
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c 'source ${HOME}/venv-phace/bin/activate && python - <<PY
import time, stageA_assigned_bound as A
A.OUT = A.OUT + "_probe"
import numpy as np
A.scale = A.load_scale()
A.centers_release = np.memmap(f"{A.RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(A.N_CORE, 512))
A.centers_selection = np.load(f"{A.HERE}/out/selection_features_madcore.npy", mmap_mode="r")
t = time.time(); A.run_folder(("anchor", "mad-next-coreset")); print("anchor probe seconds", round(time.time() - t), flush=True)
PY'
echo "probe finish $(date -Is)"
