import glob
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJ = "${DEV_ROOT}/phace-atomic"
OLD = f"{PROJ}/fps/features/mad_next_phaceatomic_meanstd.parquet"
SUBSETS = f"{PROJ}/mad/mad_next/src/.subsets_seq.txt"
CHUNKS = f"{PROJ}/mad/mad_next/src/dimers_trimers/features"
OUT = f"{PROJ}/mad/mad_next/dataset.parquet"
BACKUP = f"{PROJ}/mad/_backup/mad_next_pre_dimtri.parquet"
DROP = {"moldimers", "moltrimers"}
COLS = ["configuration_id", "dataset_id", "atomic_numbers", "positions", "cell", "pbc",
        "phace_energy", "phace_features_bb_mean", "phace_non_conservative_forces"]

sub = np.array([ln.strip() for ln in open(SUBSETS)])
keep = ~np.isin(sub, list(DROP))

old = pq.read_table(OLD).select(COLS)
assert old.num_rows == len(sub), f"row/subset mismatch: {old.num_rows} != {len(sub)}"
kept = old.filter(pa.array(keep))

new_files = sorted(glob.glob(f"{CHUNKS}/*.parquet"))
new = pa.concat_tables([pq.read_table(f).select(COLS) for f in new_files])
merged = pa.concat_tables([kept, new])

os.makedirs(os.path.dirname(BACKUP), exist_ok=True)
if not os.path.exists(BACKUP):
    shutil.copy(OLD, BACKUP)
tmp = OUT + ".tmp"
pq.write_table(merged, tmp, compression="snappy")
os.replace(tmp, OUT)
print(f"kept {kept.num_rows:,} (dropped {int((~keep).sum()):,}) + new {new.num_rows:,} "
      f"= {merged.num_rows:,}  ->  {OUT}")
