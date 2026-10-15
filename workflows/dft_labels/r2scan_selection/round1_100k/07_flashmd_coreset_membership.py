import hashlib

import numpy as np
import pyarrow.parquet as pq
from ase.io import iread

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
converged_path = f"{project_dir}/flashmd_madcore_3d.xyz"
failed_path = f"{project_dir}/flashmd_failed_frames.xyz"
mad_next_path = f"{project_dir}/mad/mad_next/dataset.parquet"
coreset_10M_path = f"{project_dir}/accrete/runs/madcore_10M/out/coreset_10M_full.parquet"
coreset_224_path = ("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92"
                    "/coreset224/out/coreset_224_full.parquet")
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


def hash_xyz(path):
    return {structure_hash(atoms.get_atomic_numbers(), atoms.get_positions(),
                           atoms.get_cell()[:]): int(atoms.info["frame_id"])
            for atoms in iread(path, format="extxyz")}


converged = hash_xyz(converged_path)
failed = hash_xyz(failed_path)
print(f"converged {len(converged):,}   failed {len(failed):,}   "
      f"union {len(set(converged) | set(failed)):,}")

parquet_file = pq.ParquetFile(mad_next_path)
cid_of = {}

for row_group in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(
        row_group, columns=["configuration_id", "atomic_numbers", "positions", "cell"])
    configuration_ids = table.column("configuration_id").to_pylist()

    for row in range(table.num_rows):
        digest = structure_hash(table.column("atomic_numbers")[row].as_py(),
                                table.column("positions")[row].as_py(),
                                table.column("cell")[row].as_py())
        if digest in converged or digest in failed:
            cid_of[digest] = configuration_ids[row]

print(f"matched into mad_next: {len(cid_of):,}")

with open(f"{output_dir}/flashmd_configuration_ids.tsv", "w") as handle:
    handle.write("configuration_id\tframe_id\tin_converged_file\tin_failed_file\n")
    for digest, cid in cid_of.items():
        frame_id = converged.get(digest, failed.get(digest))
        handle.write(f"{cid}\t{frame_id}\t{int(digest in converged)}"
                     f"\t{int(digest in failed)}\n")

for label, path in (("coreset_10M (10,000,000)", coreset_10M_path),
                    ("coreset224 (16,777,216)", coreset_224_path)):
    table = pq.read_table(path, columns=["fps_order", "configuration_id"])
    fps_orders = table.column("fps_order").to_numpy()
    ids = np.asarray(table.column("configuration_id").to_pylist(), dtype=object)
    rank_of = {cid: int(fps_orders[row]) for row, cid in enumerate(ids)}

    conv = [rank_of[cid_of[d]] for d in converged if cid_of.get(d) in rank_of]
    fail = [rank_of[cid_of[d]] for d in failed if cid_of.get(d) in rank_of]
    both = [rank_of[cid_of[d]] for d in set(converged) | set(failed)
            if cid_of.get(d) in rank_of]

    print(f"\n{label}")
    print(f"  converged selected : {len(conv):>7,} / {len(converged):,} "
          f"({100 * len(conv) / len(converged):.1f}%)")
    print(f"  failed    selected : {len(fail):>7,} / {len(failed):,} "
          f"({100 * len(fail) / len(failed):.1f}%)")
    print(f"  all flashmd        : {len(both):>7,} / {len(set(converged) | set(failed)):,} "
          f"({100 * len(both) / len(set(converged) | set(failed)):.1f}%)")

print(f"\nwrote flashmd_configuration_ids.tsv ({len(cid_of):,} rows)")
