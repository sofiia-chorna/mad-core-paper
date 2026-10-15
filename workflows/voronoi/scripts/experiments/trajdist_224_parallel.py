import glob
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

scaler = pickle.load(open("${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl", "rb"))
mean = np.asarray(scaler["mean_"], dtype=np.float64)
scale = float(scaler["scale_"])

light = pq.read_table("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/coreset224/out/coreset_224_light.parquet",
                      columns=["configuration_id", "family"])
picks = set(pc.filter(light.column("configuration_id"), pc.equal(light.column("family"), "Alexandria")).to_pylist())
del light
print(f"coreset224 Alexandria picks: {len(picks):,}", flush=True)

shards = [s for s in glob.glob(
    "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92/Alexandria/colabfit_Alexandria_geometry_optimization_paths_PBE_3D/**/*.parquet",
    recursive=True) if not s.endswith("index.parquet")]
agm_re = re.compile(r"id_(agm\d+)")


def scan_shard(shard):
    table = pq.read_table(shard, columns=["configuration_id", "names", "phace_features_bb_mean"])
    cids = table.column("configuration_id").to_pylist()
    names = table.column("names").to_pylist()
    feats = table.column("phace_features_bb_mean")
    hits = []
    for j, cid in enumerate(cids):
        if cid not in picks:
            continue
        match = agm_re.search(str(names[j]))
        if match:
            hits.append((match.group(1), cid, np.asarray(feats[j].as_py(), dtype=np.float32)))
    return hits


groups = defaultdict(dict)
duplicate_rows = 0
with Pool(64) as pool:
    for hits in pool.imap_unordered(scan_shard, shards, chunksize=4):
        for agm, cid, feat in hits:
            if cid in groups[agm]:
                duplicate_rows += 1
            else:
                groups[agm][cid] = feat
print(f"scanned {len(shards):,} shards; duplicate same-cid rows skipped: {duplicate_rows:,}", flush=True)

sizes = np.array([len(v) for v in groups.values()])
singles = int((sizes == 1).sum())
print(f"materials: {len(groups):,}  single-frame: {singles:,} ({100 * singles / len(groups):.1f}%)  "
      f"mean {sizes.mean():.3f}  max {sizes.max()}", flush=True)

all_d = []
worst = []
for agm, members in groups.items():
    if len(members) < 2:
        continue
    X = (np.stack(list(members.values())).astype(np.float64) - mean) / scale
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    d = D[np.triu_indices(len(members), 1)]
    all_d.append(d)
    worst.append((float(d.min()), len(members), agm))
all_d = np.concatenate(all_d)
worst.sort()
print(f"\nFULL PRODUCTION SCAN — same-material distinct pick pairs: {len(all_d):,}")
print(f"materials with >=2 frames: {int((sizes >= 2).sum()):,}")
print(f"min pairwise distance:  {all_d.min():.4f}   (eps(2^24)=0.2179)")
print(f"pairs == 0: {(all_d == 0).sum()}   pairs < 0.01: {(all_d < 0.01).sum()}   pairs < 0.2179: {(all_d < 0.2179).sum()}")
print(f"percentiles: p1={np.percentile(all_d, 1):.3f}  p50={np.percentile(all_d, 50):.3f}  p99={np.percentile(all_d, 99):.3f}")
print("5 closest pairs:", worst[:5], flush=True)
