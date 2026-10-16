import csv
import sys

import numpy as np

SEED = 20260908
SAMPLE_SIZE = 1000

# the 5,847 random test structures with converged PBE and LDA labels and forces below 100 eV/A
parent = sys.argv[1]
out = sys.argv[2]

with open(parent) as fh:
    rows = list(csv.DictReader(fh, delimiter="\t"))

rows.sort(key=lambda r: r["configuration_id"])

rng = np.random.default_rng(SEED)
picks = rng.choice(len(rows), size=SAMPLE_SIZE, replace=False)

picked = sorted((rows[i] for i in picks), key=lambda r: r["configuration_id"])

for rank, row in enumerate(picked):
    row["sample_rank"] = rank

with open(out, "w") as fh:
    writer = csv.DictWriter(fh, fieldnames=["sample_rank"] + list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(picked)
