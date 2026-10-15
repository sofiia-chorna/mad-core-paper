"""Data of the GFRE/LFRE heatmap (fig:gfre_lfre).

The two JSON files in data/ are the GFRE and LFRE results over the MAD-NEXT test split,
computed by workflows/gfre_lfre/union/gfre_lfre_union.py (featurization and the error
computation run as Slurm jobs, see the submit scripts in that folder). This script copies
the current results into data/.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "workflows" / "gfre_lfre" / "union" / "results_union"

for name in ("union_gfre_seed0.json", "union_lfre_seed0.json"):
    shutil.copy2(RESULTS / name, HERE / "data" / name)
    print("copied", name)
