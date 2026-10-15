import sys

import pyarrow.parquet as pq

p = sys.argv[1]
t = pq.read_table(p)
cols = t.column_names
if "max_force" in cols and "zbl_max_force" not in cols:
    t = t.rename_columns(["zbl_max_force" if c == "max_force" else c for c in cols])
    pq.write_table(t, p)
    print("renamed", p.split("/")[-1])
