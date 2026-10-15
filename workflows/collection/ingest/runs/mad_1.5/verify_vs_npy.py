"""Verify the re-featurized MAD-1.5 features (id-keyed parquet) against the old clariden npy.

Both are raw bb_mean in train|val|test frame order, so row i should correspond to the same frame.
Periodic frames should match (confirms the row<->frame ordering); non-periodic frames may differ if
the old npy was built with the buggy vesin (in which case the re-featurize is the fix).
"""
import numpy as np
import pyarrow.parquet as pq

D = "${DEV_ROOT}/mad-core/data/mad-1.5/model_292"

t = pq.read_table(f"{D}/mad_1.5_with_phace.parquet", columns=["configuration_id", "phace_features_bb_mean"])
cid = t.column("configuration_id").to_pylist()
new = t.column("phace_features_bb_mean").combine_chunks()
new = new.values.to_numpy(zero_copy_only=False).reshape(len(cid), -1).astype(np.float64)
old = np.load(f"{D}/mad_features_clariden_bb_mean.npy").astype(np.float64)
print(f"new {new.shape}  old {old.shape}  cid[0]={cid[0]}")

pbc = pq.read_table(f"{D}/mad_1.5.parquet", columns=["pbc"]).column(0).to_pylist()
periodic = np.array([bool(np.any(p)) for p in pbc])

d = np.abs(new - old).max(axis=1)


def report(mask, name):
    dd = d[mask]
    if not len(dd):
        print(f"  {name}: n=0"); return
    print(f"  {name}: n={int(mask.sum()):>7d}  maxabsdiff p50/p99/max = "
          f"{np.percentile(dd,50):.2e}/{np.percentile(dd,99):.2e}/{dd.max():.2e}  "
          f"match(<1e-3)={int((dd<1e-3).sum())}/{int(mask.sum())}")


print("max-abs feature diff vs old npy:")
report(np.ones(len(d), bool), "ALL        ")
report(periodic, "periodic   ")
report(~periodic, "non-period.")
