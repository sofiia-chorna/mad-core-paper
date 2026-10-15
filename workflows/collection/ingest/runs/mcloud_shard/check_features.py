import pickle

import numpy as np
import pyarrow.parquet as pq

sc = pickle.load(open("${STORE_ROOT}/mad-core/mad1.5/scaler.pkl", "rb"))
mean = np.asarray(sc["mean_"], dtype=np.float64)
scale = float(sc["scale_"])

print("MAD baselines (from memory): pool scaled-norm p99 ~1.59, MAD-anchor max ~6.94\n")
hdr = ["dataset", "n", "featNaN", "norm_med", "p99", "max", "mf_max", "zbl_naninf"]
print(f"{hdr[0]:16s}{hdr[1]:>9s}{hdr[2]:>8s}{hdr[3]:>9s}{hdr[4]:>7s}{hdr[5]:>8s}{hdr[6]:>8s}{hdr[7]:>11s}")

MD = "work/mcloud_rescan/merged"
DS = ["286v9-64865", "kbsns-73a82", "nv386-6m953", "w7hen-cme26",
      "2y89r-m1q15", "vf417-5pb37", "nsp7v-2dk91"]

for ds in DS:
    p = f"{MD}/mcloudarchive__{ds}.parquet"
    try:
        t = pq.read_table(p, columns=["phace_features_bb_mean", "max_force",
                                      "zbl_energy_per_atom", "zbl_max_atomic_energy"])
    except Exception:
        print(f"{ds:16s} (not merged yet)")
        continue

    X = np.array(t.column("phace_features_bb_mean").to_pylist(), dtype=np.float64)
    n = X.shape[0]
    nnan = int((~np.isfinite(X).all(axis=1)).sum())

    Xs = (X - mean) / scale
    norms = np.linalg.norm(np.where(np.isfinite(Xs), Xs, 0.0), axis=1)

    import pandas as pd
    mf = pd.to_numeric(t.column("max_force").to_pandas(), errors="coerce").to_numpy(dtype=np.float64)
    epa = pd.to_numeric(t.column("zbl_energy_per_atom").to_pandas(), errors="coerce").to_numpy(dtype=np.float64)
    mae = pd.to_numeric(t.column("zbl_max_atomic_energy").to_pandas(), errors="coerce").to_numpy(dtype=np.float64)
    zbad = int((~np.isfinite(mf)).sum() + (~np.isfinite(epa)).sum() + (~np.isfinite(mae)).sum())
    mff = mf[np.isfinite(mf)]
    mfmax = mff.max() if len(mff) else 0.0

    print(f"{ds:16s}{n:>9d}{nnan:>8d}{np.median(norms):>9.2f}"
          f"{np.percentile(norms, 99):>7.2f}{norms.max():>8.1f}{mfmax:>8.1f}{zbad:>11d}")
