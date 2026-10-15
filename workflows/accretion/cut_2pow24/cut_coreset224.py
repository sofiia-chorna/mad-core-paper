import json
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

K = 2 ** 24
runs_dir = "${DEV_ROOT}/phace-atomic/accrete/runs"
source_out = f"{runs_dir}/madcore_25M/out"
target_out = f"{runs_dir}/coreset224/out"
coreset_25m_path = f"{runs_dir}/madcore_10M/out/coreset_25M.parquet"


def cut_small(source_path, target_path):
    table = pq.read_table(source_path)
    kept = table.filter(pc.less(table.column("fps_order"), K))
    pq.write_table(kept, target_path, compression="zstd")
    print(f"{target_path.split('/')[-1]}: {kept.num_rows:,} rows", flush=True)
    return kept


coreset = cut_small(coreset_25m_path, f"{target_out}/coreset_224.parquet")
kept_cids = set(coreset.column("configuration_id").to_pylist())
assert len(kept_cids) == K
del coreset

light = cut_small(f"{source_out}/coreset_25M_light.parquet", f"{target_out}/coreset_224_light.parquet")
assert light.num_rows == K

families = light.column("family").to_pylist()
with open(f"{target_out}/family_counts_224.txt", "w") as fh:
    counts = Counter(families)
    fh.write(f"dataset counts in the 2^24 coreset (K={K:,}), {len(counts)} datasets\n\n")
    for name, count in counts.most_common():
        fh.write(f"{name:60s} {count:>10,}   {100*count/K:7.3f}%\n")
print(f"family_counts_224.txt: {len(counts)} datasets", flush=True)
del light, families

category_map = json.load(open(f"{source_out}/picks_category_map_25M.json"))
category_map_224 = {cid: category for cid, category in category_map.items() if cid in kept_cids}
json.dump(category_map_224, open(f"{target_out}/picks_category_map_224.json", "w"))
with open(f"{target_out}/category_counts_224.txt", "w") as fh:
    counts = Counter(category_map_224.values())
    fh.write(f"category counts in the 2^24 coreset (K={K:,})\n\n")
    for name, count in counts.most_common():
        fh.write(f"{name:25s} {count:>10,}   {100*count/K:7.3f}%\n")
print(f"picks_category_map_224.json: {len(category_map_224):,} entries; categories: {dict(counts)}", flush=True)
del category_map, category_map_224

colors = json.load(open(f"{source_out}/picks_category_colors_25M.json"))
json.dump(colors, open(f"{target_out}/picks_category_colors_224.json", "w"), indent=2)

cid_index = pq.read_table(f"{source_out}/cid_index_25M.parquet")
mask = pc.is_in(cid_index.column("configuration_id"), value_set=pa.array(sorted(kept_cids), pa.string()))
cid_index_224 = cid_index.filter(mask)
pq.write_table(cid_index_224, f"{target_out}/cid_index_224.parquet", compression="zstd")
print(f"cid_index_224.parquet: {cid_index_224.num_rows:,} records", flush=True)
del cid_index, cid_index_224

source_full = pq.ParquetFile(f"{source_out}/coreset_25M_full.parquet")
writer = pq.ParquetWriter(f"{target_out}/coreset_224_full.parquet", source_full.schema_arrow, compression="zstd")
rows_written = 0
for batch in source_full.iter_batches(batch_size=65536):
    table = pa.Table.from_batches([batch])
    kept = table.filter(pc.less(table.column("fps_order"), K))
    if kept.num_rows:
        writer.write_table(kept)
        rows_written += kept.num_rows
        if rows_written % 2_000_000 < kept.num_rows:
            print(f"  full: {rows_written:,}/{K:,}", flush=True)
writer.close()
print(f"coreset_224_full.parquet: {rows_written:,} rows  {'OK' if rows_written == K else '*** FAIL ***'}", flush=True)
