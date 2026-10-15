import pickle
import sys

import numpy as np
import pyarrow.parquet as pq
from skmatter.preprocessing import StandardFlexibleScaler

MANIFEST = sys.argv[1] if len(sys.argv) > 1 else "mad/union/manifest.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "mad/scaler_madunion_phaceatomic_meanstd.pkl"

files = [ln.strip() for ln in open(MANIFEST) if ln.strip()]
parts = []
for f in files:
    col = pq.read_table(f, columns=["phace_features_bb_mean"]).column(0).combine_chunks()
    parts.append(col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32))
X = np.concatenate(parts)
print(f"loaded {X.shape} from {len(files)} MAD datasets", flush=True)

scaler = StandardFlexibleScaler()
scaler.fit(X)
mean_ = np.asarray(scaler.mean_, dtype=np.float32)
scale_ = float(scaler.scale_)
print(f"fitted: ||mean_||={np.linalg.norm(mean_):.6f}  scale_={scale_:.6f}")

with open(OUT, "wb") as fh:
    pickle.dump({"mean_": mean_, "scale_": scale_}, fh)
print(f"saved -> {OUT}")
