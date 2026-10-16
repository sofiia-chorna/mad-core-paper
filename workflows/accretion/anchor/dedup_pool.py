import glob
import os

from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


small_root = "${STORE_LP133}/mad-core/accrete_pools/madcore_prod/small"
fps_root = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
run_dir = "${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production"
big_families = ["OC20-IS2RES", "Open_Molecular_Crystals_2025_OMC25", "ANI-1", "ANI-2x", "Alexandria",
                "OC20_S2EF_train_all", "ODAC", "OMat24", "OMol25", "gdb13"]


def dedup_shards(task):
    kind, fam, shards = task
    seen = set()
    raw = kept = 0
    rewrites = []
    for shard in shards:
        cids = pq.read_table(shard, columns=["configuration_id"]).column(0).to_pylist()
        n = len(cids)
        raw += n
        mask = np.fromiter((c not in seen and not seen.add(c) for c in cids), dtype=bool, count=n)
        k = int(mask.sum())
        kept += k
        if k < n:
            rewrites.append((shard, mask))
    for shard, mask in rewrites:
        table = pq.read_table(shard)
        deduped = table.filter(pa.array(mask))
        if deduped.num_rows == 0:
            os.remove(shard)
        else:
            tmp = shard + ".tmp"
            pq.write_table(deduped, tmp, compression="snappy")
            os.replace(tmp, shard)
    return kind, fam, raw, kept


tasks = []
for fam_dir in sorted(d for d in glob.glob(f"{small_root}/*") if os.path.isdir(d)):
    tasks.append(("small", os.path.basename(fam_dir), sorted(glob.glob(f"{fam_dir}/*.parquet"))))
for fam in big_families:
    tasks.append(("big", fam, sorted(glob.glob(f"{fps_root}/{fam}/with_features/*.parquet"))))

print(f"deduping {len(tasks)} families (small + big) ...", flush=True)
tot = {"small": [0, 0], "big": [0, 0]}
changed = []
with Pool(24) as pool:
    for kind, fam, raw, kept in pool.imap_unordered(dedup_shards, tasks):
        tot[kind][0] += raw
        tot[kind][1] += kept
        if kept < raw:
            changed.append((raw - kept, raw, kind, fam))

for kind in ("big", "small"):
    r, k = tot[kind]
    print(f"{kind:5}: raw={r:,} kept={k:,} removed={r - k:,} ({100 * (r - k) / max(r, 1):.2f}%)", flush=True)
print("families changed (top 25):", flush=True)
for dup, raw, kind, fam in sorted(changed, reverse=True)[:25]:
    print(f"  {kind:5} {fam:42s} -{dup:>9,} / {raw:>10,}  ({100 * dup / raw:.0f}%)", flush=True)

# rebuild manifest (deduped big + deduped small)
pool_shards = []
for fam in big_families:
    pool_shards += sorted(glob.glob(f"{fps_root}/{fam}/with_features/*.parquet"))
for fam_dir in sorted(d for d in glob.glob(f"{small_root}/*") if os.path.isdir(d)):
    pool_shards += sorted(glob.glob(f"{fam_dir}/*.parquet"))
with open(f"{run_dir}/pool_manifest.txt", "w") as f:
    f.write("\n".join(pool_shards) + "\n")

big_k, small_k = tot["big"][1], tot["small"][1]
print(f"\nrebuilt pool_manifest: {len(pool_shards):,} shards", flush=True)
print(f"POOL now: big={big_k:,} + small={small_k:,} = {big_k + small_k:,} unique rows", flush=True)
