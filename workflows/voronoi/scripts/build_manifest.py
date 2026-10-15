import glob
import os
from multiprocessing import Pool

import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
store_roots = [
    "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92",
    "${STORE_LP133}/mad-core/datasets/phace_atomic_ep92",
]
anchor_parquet = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"


def shard_info(path):
    try:
        parquet_file = pq.ParquetFile(path)
        names = parquet_file.schema_arrow.names
        has_features = "phace_features_bb_mean" in names
        return path, parquet_file.metadata.num_rows, has_features
    except Exception as error:
        return path, -1, str(error)


shard_paths = []
for root in store_roots:
    for family_dir in sorted(glob.glob(f"{root}/*")):
        if not os.path.isdir(family_dir):
            continue
        for path in sorted(glob.glob(f"{family_dir}/**/*.parquet", recursive=True)):
            if not path.endswith("index.parquet") and "/_pilot/" not in path:
                shard_paths.append(path)
shard_paths.append(anchor_parquet)
print(f"scanning {len(shard_paths):,} parquet files", flush=True)

with Pool(64) as pool:
    results = pool.map(shard_info, shard_paths, chunksize=16)

total_rows = 0
n_bad = 0
with open(f"{voronoi_dir}/out/manifest_all.tsv", "w") as tsv, \
     open(f"{voronoi_dir}/out/manifest_all.txt", "w") as txt:
    tsv.write("shard\trows\thas_features\n")
    for path, rows, has_features in results:
        if rows < 0 or has_features is not True:
            n_bad += 1
            print(f"SKIP {path}: rows={rows} features={has_features}", flush=True)
            continue
        total_rows += rows
        tsv.write(f"{path}\t{rows}\t{has_features}\n")
        txt.write(path + "\n")

print(f"manifest: {len(shard_paths) - n_bad:,} shards, {total_rows:,} rows total, skipped={n_bad}", flush=True)
