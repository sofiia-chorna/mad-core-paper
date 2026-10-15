"""Fit a StandardFlexibleScaler on phace_features_bb_mean and save it as the
{mean_, scale_} dict that run_fps.py expects."""

import argparse
import pickle

import numpy as np
import pyarrow.parquet as pq
from skmatter.preprocessing import StandardFlexibleScaler

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feat", default="phace_features_bb_mean")
args = ap.parse_args()

t = pq.read_table(args.parquet, columns=[args.feat]).column(0).combine_chunks()
n = len(t)
X = t.values.to_numpy(zero_copy_only=False).reshape(n, -1).astype(np.float32)
print(f"loaded features: {X.shape}")

scaler = StandardFlexibleScaler()
scaler.fit(X)
mean_ = np.asarray(scaler.mean_, dtype=np.float32)
scale_ = float(scaler.scale_)
print(f"fitted: ||mean_||={np.linalg.norm(mean_):.6f}  scale_={scale_:.6f}")

with open(args.out, "wb") as f:
    pickle.dump({"mean_": mean_, "scale_": scale_}, f)
print(f"saved -> {args.out}")
