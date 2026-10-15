import hashlib

import numpy as np
import pyarrow.parquet as pq
from ase.io import iread

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
failed_frames_path = f"{project_dir}/flashmd_failed_frames.xyz"
mad_next_path = f"{project_dir}/mad/mad_next/dataset.parquet"
bor_walk_limit = 122_378
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


frame_id_by_hash = {}
frame_ids = []

for atoms in iread(failed_frames_path, format="extxyz"):
    frame_id = int(atoms.info["frame_id"])
    frame_ids.append(frame_id)
    digest = structure_hash(atoms.get_atomic_numbers(), atoms.get_positions(),
                            atoms.get_cell()[:])
    frame_id_by_hash[digest] = frame_id

print(f"failed frames: {len(frame_ids):,}  distinct frame_id: {len(set(frame_ids)):,}  "
      f"distinct hash: {len(frame_id_by_hash):,}")
print(f"frame_id range: {min(frame_ids):,} .. {max(frame_ids):,}")

with open(f"{output_dir}/failed_in_coreset_cids.txt") as handle:
    rows = [line.rstrip("\n").split("\t") for line in handle]
coreset_position = {cid: (int(rank), int(size)) for rank, size, cid in rows}

print(f"\nfailed frames found in coreset_10M: {len(coreset_position):,}")

parquet_file = pq.ParquetFile(mad_next_path)
rows_to_read = {}

for row_group in range(parquet_file.num_row_groups):
    configuration_ids = parquet_file.read_row_group(
        row_group, columns=["configuration_id"]).column(0).to_pylist()
    for row, cid in enumerate(configuration_ids):
        if cid in coreset_position:
            rows_to_read.setdefault(row_group, []).append((row, cid))

mapped = []

for row_group, entries in rows_to_read.items():
    table = parquet_file.read_row_group(
        row_group, columns=["atomic_numbers", "positions", "cell"])
    for row, cid in entries:
        digest = structure_hash(table.column("atomic_numbers")[row].as_py(),
                                table.column("positions")[row].as_py(),
                                table.column("cell")[row].as_py())
        frame_id = frame_id_by_hash.get(digest)
        if frame_id is not None:
            rank, size = coreset_position[cid]
            mapped.append((rank, size, cid, frame_id))

mapped.sort()
print(f"mapped to frame_id: {len(mapped):,} of {len(coreset_position):,}")

in_bor_walk = [entry for entry in mapped if entry[0] <= bor_walk_limit]
print(f"\nin the BoR walk (fps_order <= {bor_walk_limit:,}): {len(in_bor_walk):,}")

with open(f"{output_dir}/failed_frame_ids_BOR_detail.tsv", "w") as handle:
    handle.write("fps_order\tn_atoms\tconfiguration_id\tframe_id\n")
    for rank, size, cid, frame_id in in_bor_walk:
        handle.write(f"{rank}\t{size}\t{cid}\t{frame_id}\n")

print(f"\nwrote failed_frame_ids_BOR_detail.tsv ({len(in_bor_walk):,} rows)")
print(f"first 10 frame_ids: {[entry[3] for entry in in_bor_walk[:10]]}")
