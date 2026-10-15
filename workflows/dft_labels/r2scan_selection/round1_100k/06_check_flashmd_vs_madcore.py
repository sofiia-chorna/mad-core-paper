import hashlib

import numpy as np
import pyarrow.parquet as pq
from ase.io import iread

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
converged_path = f"{project_dir}/flashmd_madcore_3d.xyz"
failed_path = f"{project_dir}/flashmd_failed_frames.xyz"
mad_next_path = f"{project_dir}/mad/mad_next/dataset.parquet"
coreset_path = f"{project_dir}/accrete/runs/madcore_10M/out/coreset_10M_full.parquet"
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
    by_hash = {}
    n_frames = 0
    for atoms in iread(path, format="extxyz"):
        n_frames += 1
        by_hash[structure_hash(atoms.get_atomic_numbers(), atoms.get_positions(),
                               atoms.get_cell()[:])] = int(atoms.info["frame_id"])
    return by_hash, n_frames


converged, n_converged = hash_xyz(converged_path)
print(f"flashmd_madcore_3d.xyz : {n_converged:,} frames, {len(converged):,} distinct hashes")

failed, n_failed = hash_xyz(failed_path)
print(f"flashmd_failed_frames  : {n_failed:,} frames, {len(failed):,} distinct hashes")

overlap = set(converged) & set(failed)
print(f"\nhashes in BOTH files   : {len(overlap):,}")
print(f"union (all flashmd)    : {len(set(converged) | set(failed)):,}")

parquet_file = pq.ParquetFile(mad_next_path)
in_mad_next = {}
n_scanned = 0

for row_group in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(
        row_group, columns=["configuration_id", "atomic_numbers", "positions", "cell"])
    configuration_ids = table.column("configuration_id").to_pylist()

    for row in range(table.num_rows):
        digest = structure_hash(table.column("atomic_numbers")[row].as_py(),
                                table.column("positions")[row].as_py(),
                                table.column("cell")[row].as_py())
        if digest in converged or digest in failed:
            in_mad_next[digest] = configuration_ids[row]

    n_scanned += table.num_rows
    if (row_group + 1) % 5 == 0:
        print(f"  scanned {n_scanned:,}/{parquet_file.metadata.num_rows:,}, "
              f"matched {len(in_mad_next):,}", flush=True)

converged_in = {d for d in converged if d in in_mad_next}
failed_in = {d for d in failed if d in in_mad_next}

print(f"\nmad_next rows scanned  : {n_scanned:,}")
print(f"\nDIRECTION 1 - is everything in the file also in madcore?")
print(f"  converged in mad_next : {len(converged_in):,} / {len(converged):,}")
print(f"  converged MISSING     : {len(converged) - len(converged_in):,}")
print(f"  failed    in mad_next : {len(failed_in):,} / {len(failed):,}")
print(f"  failed    MISSING     : {len(failed) - len(failed_in):,}")

print(f"\nDIRECTION 2 - is every flashmd row in madcore also in the files?")
print(f"  mad_next rows matching either file: {len(in_mad_next):,}")
print(f"  (mad_next has {n_scanned:,} rows total; the rest are non-flashmd components)")

coreset = pq.read_table(coreset_path, columns=["fps_order", "configuration_id"])
fps_orders = coreset.column("fps_order").to_numpy()
coreset_ids = np.asarray(coreset.column("configuration_id").to_pylist(), dtype=object)
rank_of = {cid: int(fps_orders[row]) for row, cid in enumerate(coreset_ids)}

conv_ranks = [rank_of[in_mad_next[d]] for d in converged_in if in_mad_next[d] in rank_of]
fail_ranks = [rank_of[in_mad_next[d]] for d in failed_in if in_mad_next[d] in rank_of]

print(f"\nselected into the coreset (coreset_10M):")
print(f"  converged : {len(conv_ranks):,} of {len(converged_in):,}")
print(f"  failed    : {len(fail_ranks):,} of {len(failed_in):,}")

missing_ids = [(converged[d], d) for d in converged if d not in in_mad_next]
if missing_ids:
    with open(f"{output_dir}/flashmd_converged_not_in_mad_next.txt", "w") as handle:
        handle.write("frame_id\n")
        for frame_id, _ in sorted(missing_ids):
            handle.write(f"{frame_id}\n")
    print(f"\nwrote flashmd_converged_not_in_mad_next.txt ({len(missing_ids):,} frame_ids)")
    print(f"  first 10: {[f for f, _ in sorted(missing_ids)][:10]}")
