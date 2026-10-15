"""Merge MAD-1.5 structures + features into ONE parquet, like colabfit's _with_phace (full copy +
phace). The two were written in the same row order (featurizer preserves it), so after asserting the
configuration_id columns line up row-for-row we just append the phace columns. Output overwrites
mad_1.5_with_phace.parquet so it becomes the single full+phace file (geometry + DFT energy/forces +
phace embedding + predicted energy/forces), joinable by configuration_id.
"""
import os
import pyarrow.parquet as pq

D = "${DEV_ROOT}/mad-core/data/mad-1.5/model_292"
STRUCT = f"{D}/mad_1.5.parquet"
FEAT = f"{D}/mad_1.5_with_phace.parquet"

struct = pq.read_table(STRUCT)
feat = pq.read_table(FEAT)
assert struct.column("configuration_id").to_pylist() == feat.column("configuration_id").to_pylist(), \
    "configuration_id order mismatch between structures and features"

PHACE_COLS = ["phace_energy", "phace_features_bb_mean", "phace_non_conservative_forces"]
merged = struct
for c in PHACE_COLS:
    merged = merged.append_column(c, feat.column(c))

tmp = f"{FEAT}.tmp"
pq.write_table(merged, tmp)
os.replace(tmp, FEAT)
print(f"merged -> {FEAT}")
print(f"rows: {merged.num_rows}")
print(f"cols: {merged.column_names}")
