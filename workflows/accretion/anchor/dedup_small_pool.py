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


def dedup_family(fam_dir):
    """Keep first occurrence of each configuration_id across a family's shards (in place)."""
    shards = sorted(glob.glob(f"{fam_dir}/*.parquet"))
    seen = set()
    raw = kept = 0
    for shard in shards:
        table = pq.read_table(shard)
        n = table.num_rows
        raw += n
        cids = table.column("configuration_id").to_pylist()
        mask = np.fromiter((c not in seen and not seen.add(c) for c in cids), dtype=bool, count=n)
        k = int(mask.sum())
        kept += k
        if k == n:
            continue                                  # no dups in this shard -> leave it
        if k == 0:
            os.remove(shard)                          # entirely duplicate shard
        else:
            deduped = table.filter(pa.array(mask))
            tmp = shard + ".tmp"
            pq.write_table(deduped, tmp, compression="snappy")
            os.replace(tmp, shard)
    return os.path.basename(fam_dir), raw, kept


fam_dirs = sorted(d for d in glob.glob(f"{small_root}/*") if os.path.isdir(d))
print(f"deduping {len(fam_dirs)} small families ...", flush=True)

raw_total = kept_total = 0
changed = []
with Pool(24) as pool:
    for fam, raw, kept in pool.imap_unordered(dedup_family, fam_dirs):
        raw_total += raw
        kept_total += kept
        if kept < raw:
            changed.append((raw - kept, raw, fam))

print(f"\nsmall pool: raw={raw_total:,} kept={kept_total:,} removed={raw_total - kept_total:,} "
      f"({100 * (raw_total - kept_total) / max(raw_total, 1):.2f}%)", flush=True)
print("families changed (top 25 by rows removed):", flush=True)
for dup, raw, fam in sorted(changed, reverse=True)[:25]:
    print(f"  {fam:44s} -{dup:>9,} / {raw:>10,}  ({100 * dup / raw:.0f}%)", flush=True)

# ---- rebuild the pool manifest (big coresets + deduped small; mad-1.5/mad-next already deleted) ----
pool_shards = []
for fam in big_families:
    pool_shards += sorted(glob.glob(f"{fps_root}/{fam}/with_features/*.parquet"))
for fam_dir in fam_dirs:
    pool_shards += sorted(glob.glob(f"{fam_dir}/*.parquet"))

with open(f"{run_dir}/pool_manifest.txt", "w") as f:
    f.write("\n".join(pool_shards) + "\n")

big_rows = kept_total  # small kept; big below
big_n = 0
for fam in big_families:
    big_n += sum(pq.ParquetFile(p).metadata.num_rows for p in sorted(glob.glob(f"{fps_root}/{fam}/with_features/*.parquet")))
print(f"\nrebuilt pool_manifest: {len(pool_shards):,} shards", flush=True)
print(f"POOL now: big={big_n:,} + small(dedup)={kept_total:,} = {big_n + kept_total:,} rows", flush=True)
