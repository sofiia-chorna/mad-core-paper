import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "workflows" / "gfre_lfre" / "union" / "results_union"

for name in ("union_gfre_seed0.json", "union_lfre_seed0.json"):
    shutil.copy2(RESULTS / name, HERE / "data" / name)
    print("copied", name)
