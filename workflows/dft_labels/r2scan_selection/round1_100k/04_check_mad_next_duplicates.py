import collections
import hashlib

import numpy as np
import pyarrow.parquet as pq

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
mad_next_path = f"{project_dir}/mad/mad_next/dataset.parquet"
decimals = 3


def structure_hash(atomic_numbers, positions, cell):
    scale = 10 ** decimals
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    rounded_positions = np.round(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3) * scale).astype(np.int64)
    rounded_cell = np.round(
        np.asarray(cell, dtype=np.float64).reshape(-1) * scale).astype(np.int64)

    return hashlib.blake2b(
        numbers.tobytes() + rounded_positions.tobytes() + rounded_cell.tobytes(),
        digest_size=16).digest()


parquet_file = pq.ParquetFile(mad_next_path)
ids_by_hash = collections.defaultdict(list)
n_scanned = 0

for row_group in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(
        row_group, columns=["configuration_id", "atomic_numbers", "positions", "cell"])
    configuration_ids = table.column("configuration_id").to_pylist()

    for row in range(table.num_rows):
        digest = structure_hash(table.column("atomic_numbers")[row].as_py(),
                                table.column("positions")[row].as_py(),
                                table.column("cell")[row].as_py())
        ids_by_hash[digest].append(configuration_ids[row])

    n_scanned += table.num_rows
    if (row_group + 1) % 5 == 0:
        print(f"  scanned {n_scanned:,}/{parquet_file.metadata.num_rows:,}", flush=True)

collisions = {digest: ids for digest, ids in ids_by_hash.items() if len(ids) > 1}
n_extra = sum(len(ids) - 1 for ids in collisions.values())

print(f"\nmad_next rows            : {n_scanned:,}")
print(f"distinct structure hashes: {len(ids_by_hash):,}")
print(f"hashes with >1 row       : {len(collisions):,}")
print(f"surplus rows             : {n_extra:,} ({100 * n_extra / n_scanned:.4f}%)")

with open(f"{output_dir}/failed_frame_ids_BOR_detail.tsv") as handle:
    next(handle)
    matched_ids = {line.split("\t")[2] for line in handle}

affected = {digest: ids for digest, ids in collisions.items()
            if any(cid in matched_ids for cid in ids)}

print(f"\nof the 723 reported configuration_ids, how many sit in a colliding group: "
      f"{sum(1 for ids in affected.values() for cid in ids if cid in matched_ids):,}")

if affected:
    print("\nexamples:")
    for digest, ids in list(affected.items())[:5]:
        print(f"  {len(ids)} rows share one geometry: {ids[:4]}")
else:
    print("  none - every reported configuration_id maps to exactly one geometry")

if collisions:
    sizes = collections.Counter(len(ids) for ids in collisions.values())
    print(f"\ncollision group sizes: {dict(sorted(sizes.items()))}")
    with open(f"{output_dir}/mad_next_duplicate_geometries.txt", "w") as handle:
        for digest, ids in sorted(collisions.items(), key=lambda item: -len(item[1])):
            handle.write("\t".join(ids) + "\n")
    print(f"wrote mad_next_duplicate_geometries.txt ({len(collisions):,} groups)")
