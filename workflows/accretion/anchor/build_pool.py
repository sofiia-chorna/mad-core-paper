import glob
import os

from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


coreset_root = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
store_root = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
run_dir = "${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production"
small_pool_root = "${STORE_LP133}/mad-core/accrete_pools/madcore_prod/small"
ahmetcik_entries_path = f"{run_dir}/../no_ahm/ahmetcik_entries.txt"

feature_columns = ["configuration_id", "dataset_id", "phace_features_bb_mean"]
filter_columns = feature_columns + ["zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy"]

max_force = 200.0
max_energy_per_atom = 50.0
max_atomic_energy = 100.0

big_families = [
    "OC20-IS2RES", "Open_Molecular_Crystals_2025_OMC25", "ANI-1", "ANI-2x", "Alexandria",
    "OC20_S2EF_train_all", "ODAC", "OMat24", "OMol25", "gdb13",
]
big_families_in_store = set(big_families) - {"gdb13"}

ahmetcik_entries = set(line.strip() for line in open(ahmetcik_entries_path) if line.strip())


def zbl_keep_mask(table):
    feature_lists = table.column("phace_features_bb_mean").combine_chunks()
    feature_matrix = feature_lists.values.to_numpy(zero_copy_only=False).reshape(len(feature_lists), -1)

    features_finite = np.isfinite(feature_matrix).all(axis=1)

    force_ok = table.column("zbl_max_force").to_numpy() < max_force

    energy_ok = (table.column("zbl_energy_per_atom").to_numpy() <= max_energy_per_atom) \
        & (table.column("zbl_max_atomic_energy").to_numpy() <= max_atomic_energy)

    return features_finite & force_ok & energy_ok


def not_ahmetcik_mask(table):
    entry_ids = (cid.split(":", 1)[-1] for cid in table.column("configuration_id").to_pylist())
    return np.fromiter((entry_id not in ahmetcik_entries for entry_id in entry_ids),
                       dtype=bool, count=table.num_rows)


def filter_small_shard(task):
    source_shard, destination_part = task

    table = pq.read_table(source_shard, columns=filter_columns)

    if table.num_rows == 0:
        return 0, 0

    raw = table.num_rows

    keep = zbl_keep_mask(table)

    if "/nomad/" in source_shard:
        keep = keep & not_ahmetcik_mask(table)

    kept = table.select(feature_columns).filter(pa.array(keep))

    if kept.num_rows:
        pq.write_table(kept, destination_part, compression="snappy")

    return raw, kept.num_rows


small_shards_by_family = defaultdict(list)

for path in glob.glob(f"{store_root}/**/*.parquet", recursive=True):
    if "/_pilot/" in path or path.endswith("index.parquet"):
        continue

    family = path[len(store_root) + 1:].split("/")[0]

    if family not in big_families_in_store:
        small_shards_by_family[family].append(path)


small_tasks = []

for family in small_shards_by_family:
    out_dir = f"{small_pool_root}/{family}"

    if glob.glob(f"{out_dir}/*.parquet"):
        continue

    os.makedirs(out_dir, exist_ok=True)

    for shard_index, shard in enumerate(sorted(small_shards_by_family[family])):
        small_tasks.append((shard, f"{out_dir}/part_{shard_index:05d}.parquet"))


raw_total = kept_total = 0

if small_tasks:
    with Pool(64) as pool:
        for raw, kept in pool.imap_unordered(filter_small_shard, small_tasks, chunksize=1):
            raw_total += raw
            kept_total += kept


pool_shards = []
big_rows = 0

for family in big_families:
    fam_shards = sorted(glob.glob(f"{coreset_root}/{family}/with_features/*.parquet"))
    pool_shards += fam_shards
    big_rows += sum(pq.ParquetFile(p).metadata.num_rows for p in fam_shards)

small_rows = 0

for family in sorted(small_shards_by_family):
    fam_shards = sorted(glob.glob(f"{small_pool_root}/{family}/*.parquet"))
    pool_shards += fam_shards
    small_rows += sum(pq.ParquetFile(p).metadata.num_rows for p in fam_shards)

with open(f"{run_dir}/pool_manifest.txt", "w") as manifest_file:
    manifest_file.write("\n".join(pool_shards) + "\n")

print(f"big families (full FPS with_features): {big_rows:,} rows across 10 families", flush=True)
print(f"small shards filtered this run: raw={raw_total:,} kept={kept_total:,} dropped={raw_total - kept_total:,}", flush=True)
print(f"small pool rows (all families): {small_rows:,}", flush=True)
print(f"POOL TOTAL rows: {big_rows + small_rows:,}   manifest shards: {len(pool_shards):,}", flush=True)
