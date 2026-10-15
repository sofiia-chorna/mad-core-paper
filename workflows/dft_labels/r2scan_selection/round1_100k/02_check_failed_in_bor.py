import hashlib

import numpy as np
import pyarrow.parquet as pq
from ase.io import iread

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
failed_frames_path = f"{project_dir}/flashmd_failed_frames.xyz"
mad_next_path = f"{project_dir}/mad/mad_next/dataset.parquet"
coreset_path = f"{project_dir}/accrete/runs/madcore_10M/out/coreset_10M_full.parquet"
bor_walk_limit = 122_378
max_atoms = 300
n_failed_frames = 5_230


def structure_hash(atomic_numbers, positions, cell, decimals):
    scale = 10 ** decimals
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    rounded_positions = np.round(
        np.asarray(positions, dtype=np.float64).reshape(-1, 3) * scale).astype(np.int64)
    rounded_cell = np.round(
        np.asarray(cell, dtype=np.float64).reshape(-1) * scale).astype(np.int64)

    return hashlib.blake2b(
        numbers.tobytes() + rounded_positions.tobytes() + rounded_cell.tobytes(),
        digest_size=16).digest()


def hash_failed_frames(decimals):
    frames_by_hash = {}
    for index, atoms in enumerate(iread(failed_frames_path, format="extxyz")):
        digest = structure_hash(atoms.get_atomic_numbers(), atoms.get_positions(),
                                atoms.get_cell()[:], decimals)
        frames_by_hash[digest] = index

    return frames_by_hash


def match_mad_next(frames_by_hash, decimals):
    parquet_file = pq.ParquetFile(mad_next_path)
    matched = {}
    n_scanned = 0

    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group, columns=["configuration_id", "atomic_numbers", "positions", "cell"])
        configuration_ids = table.column("configuration_id").to_pylist()

        for row in range(table.num_rows):
            digest = structure_hash(table.column("atomic_numbers")[row].as_py(),
                                    table.column("positions")[row].as_py(),
                                    table.column("cell")[row].as_py(), decimals)
            if digest in frames_by_hash:
                matched[configuration_ids[row]] = frames_by_hash[digest]

        n_scanned += table.num_rows
        if (row_group + 1) % 5 == 0:
            print(f"  scanned {n_scanned:,}/{parquet_file.metadata.num_rows:,} mad_next, "
                  f"matched {len(matched):,}", flush=True)

    return matched


for decimals in (3, 1):
    print(f"\nmatching with positions rounded to {decimals} decimals", flush=True)

    frames_by_hash = hash_failed_frames(decimals)
    print(f"failed frames hashed: {len(frames_by_hash):,} distinct of {n_failed_frames:,}",
          flush=True)

    matched = match_mad_next(frames_by_hash, decimals)
    print(f"\nmatched {len(matched):,} of {n_failed_frames:,} failed frames to mad_next "
          f"configuration_ids ({100 * len(matched) / n_failed_frames:.1f}%)", flush=True)

    if len(matched) < 100 and decimals == 3:
        print("  low match rate, retrying with looser rounding", flush=True)
        continue

    coreset = pq.read_table(coreset_path,
                            columns=["fps_order", "configuration_id", "n_sites"])
    fps_orders = coreset.column("fps_order").to_numpy()
    configuration_ids = np.asarray(coreset.column("configuration_id").to_pylist(),
                                   dtype=object)
    n_sites = coreset.column("n_sites").to_numpy()

    in_coreset = {cid: (int(fps_orders[row]), int(n_sites[row]))
                  for row, cid in enumerate(configuration_ids) if cid in matched}
    print(f"\nof those, in coreset_10M: {len(in_coreset):,}")

    if not in_coreset:
        break

    ranks = np.array([rank for rank, _ in in_coreset.values()])
    sizes = np.array([size for _, size in in_coreset.values()])
    in_bor_walk = ranks <= bor_walk_limit

    print(f"  fps_order range: {ranks.min():,} .. {ranks.max():,}")
    print(f"\n  in the BoR walk (fps_order <= {bor_walk_limit:,}): {int(in_bor_walk.sum()):,}")
    print(f"      of these, < {max_atoms} atoms (would have qualified): "
          f"{int((in_bor_walk & (sizes < max_atoms)).sum()):,}")
    print(f"      >= {max_atoms} atoms (excluded by the size filter anyway): "
          f"{int((in_bor_walk & (sizes >= max_atoms)).sum()):,}")

    by_rank = sorted(in_coreset.items(), key=lambda item: item[1][0])

    with open(f"{output_dir}/failed_in_coreset_cids.txt", "w") as handle:
        for cid, (rank, size) in by_rank:
            handle.write(f"{rank}\t{size}\t{cid}\n")

    print(f"\n  wrote failed_in_coreset_cids.txt ({len(by_rank):,} rows)")
    break
