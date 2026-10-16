import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, os.pardir, os.pardir, "workflows", "gfre_lfre", "random_test", "results_random_test")
DATA = os.path.join(HERE, "data")
FILES = [
    "random_gfre_seed0.json",
    "random_lfre_seed0.json",
    "summary_seed0.json",
]

os.makedirs(DATA, exist_ok=True)

for name in FILES:
    shutil.copy2(os.path.join(SOURCE, name), os.path.join(DATA, name))
    print(f"copied {name}")
