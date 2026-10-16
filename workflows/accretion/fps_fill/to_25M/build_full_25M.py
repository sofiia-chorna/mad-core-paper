import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

base_dir = "${DEV_ROOT}/phace-atomic/accrete/runs"
run_dir = f"{base_dir}/madcore_25M"
prod_dir = f"{base_dir}/madcore_production"
coreset_path = f"{base_dir}/madcore_10M/out/coreset_25M.parquet"
output_path = f"{run_dir}/out/coreset_25M_full.parquet"
zbl_columns = ["zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy"]
geometry_columns = ["atomic_numbers", "positions", "cell", "pbc", "phace_features_bb_mean"]
phace_columns = ["phace_energy", "phace_non_conservative_forces"]
num_processes = int(os.environ.get("GATHER_NPROC", "96"))
chunk_size = int(os.environ.get("GATHER_CHUNK", "12000"))

output_schema = pa.schema([
    ("fps_order", pa.int64()), ("epsilon_sq", pa.float64()), ("k_set", pa.int32()),
    ("source_block", pa.string()),
    ("configuration_id", pa.string()), ("dataset_id", pa.string()), ("family", pa.string()),
    ("n_sites", pa.int32()), ("periodicity", pa.int8()), ("elements", pa.list_(pa.int16())),
    ("zbl_max_force", pa.float64()), ("zbl_energy_per_atom", pa.float64()), ("zbl_max_atomic_energy", pa.float64()),
    ("atomic_numbers", pa.list_(pa.int16())), ("positions", pa.list_(pa.float32())),
    ("cell", pa.list_(pa.float32())), ("pbc", pa.list_(pa.bool_())),
    ("phace_energy", pa.float64()), ("phace_non_conservative_forces", pa.list_(pa.float32())),
    ("phace_features_bb_meanstd", pa.list_(pa.float32())),
])


def is_mad_family(family):
    return family.startswith("mad-") or family == "mad-coreset"


def compute_k_set(fps_order):
    return 1 if fps_order == 0 else int(np.floor(np.log2(fps_order)) + 1)


def read_rows(parquet_file, row_indices, columns):
    row_indices = sorted(row_indices)
    row_group_tables, kept_rows, row_group_start = [], [], 0
    for row_group_index in range(parquet_file.num_row_groups):
        row_group_num_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        local_row_indices = [row_index - row_group_start for row_index in row_indices
                             if row_group_start <= row_index < row_group_start + row_group_num_rows]
        if local_row_indices:
            row_group_tables.append(parquet_file.read_row_group(row_group_index, columns=columns).take(local_row_indices))
            kept_rows += [row_index for row_index in row_indices
                          if row_group_start <= row_index < row_group_start + row_group_num_rows]
        row_group_start += row_group_num_rows
    return (pa.concat_tables(row_group_tables) if row_group_tables else None), kept_rows


def read_task(shard_and_row_metadata):
    shard, row_metadata_list = shard_and_row_metadata
    parquet_file = pq.ParquetFile(shard)
    column_names = parquet_file.schema_arrow.names
    has_zbl_columns = all(column in column_names for column in zbl_columns)
    has_phace_columns = all(column in column_names for column in phace_columns)
    columns_to_read = geometry_columns + (zbl_columns if has_zbl_columns else []) + (phace_columns if has_phace_columns else [])
    table, kept_rows = read_rows(parquet_file, [row_index for row_index, _, _ in row_metadata_list], columns_to_read)
    row_to_meta = {row_index: (configuration_id, meta) for row_index, configuration_id, meta in row_metadata_list}

    output_rows = []
    for row_position, row_index in enumerate(kept_rows):
        configuration_id, (fps_order, epsilon_sq, k_set, source_block, dataset_id, family) = row_to_meta[row_index]
        atomic_numbers_array = np.asarray(table.column("atomic_numbers")[row_position].as_py(), dtype=np.int16)
        pbc_flags = table.column("pbc")[row_position].as_py()
        zbl_values = ([float(table.column(column)[row_position].as_py()) for column in zbl_columns]
                      if has_zbl_columns else [float("nan")] * 3)
        phace_energy_value = float(table.column("phace_energy")[row_position].as_py()) if has_phace_columns else float("nan")
        phace_forces_flat = (np.asarray(table.column("phace_non_conservative_forces")[row_position].as_py(),
                                        dtype=np.float32).reshape(-1).tolist() if has_phace_columns else [])
        output_rows.append({
            "fps_order": fps_order, "epsilon_sq": epsilon_sq, "k_set": k_set, "source_block": source_block,
            "configuration_id": configuration_id, "dataset_id": dataset_id, "family": family,
            "n_sites": int(len(atomic_numbers_array)), "periodicity": int(sum(bool(flag) for flag in pbc_flags)),
            "elements": np.unique(atomic_numbers_array).tolist(),
            "zbl_max_force": zbl_values[0], "zbl_energy_per_atom": zbl_values[1], "zbl_max_atomic_energy": zbl_values[2],
            "atomic_numbers": atomic_numbers_array.tolist(),
            "positions": np.asarray(table.column("positions")[row_position].as_py(), dtype=np.float32).reshape(-1).tolist(),
            "cell": np.asarray(table.column("cell")[row_position].as_py(), dtype=np.float32).reshape(-1).tolist(),
            "pbc": [bool(flag) for flag in pbc_flags],
            "phace_energy": phace_energy_value, "phace_non_conservative_forces": phace_forces_flat,
            "phace_features_bb_meanstd": np.asarray(table.column("phace_features_bb_mean")[row_position].as_py(), dtype=np.float32).tolist(),
        })
    return output_rows


def main():
    coreset_table = pq.read_table(coreset_path, columns=["fps_order", "epsilon_sq", "configuration_id", "dataset_id"])
    fps_order_array = coreset_table.column("fps_order").to_numpy()
    epsilon_sq_array = coreset_table.column("epsilon_sq").to_numpy()
    dataset_id_list = coreset_table.column("dataset_id").to_pylist()
    pick_info = {
        configuration_id: (int(fps_order_array[index]), float(epsilon_sq_array[index]), dataset_id_list[index])
        for index, configuration_id in enumerate(coreset_table.column("configuration_id").to_pylist())
    }
    del coreset_table, fps_order_array, epsilon_sq_array, dataset_id_list
    num_picks = len(pick_info)

    accrete_table = pq.read_table(f"{prod_dir}/out/accrete.parquet", columns=["fps_order", "configuration_id"])
    separation_by_k_set = {
        k_set_value: float(np.load(f"{prod_dir}/out/ckpt/ckpt_k{k_set_value:02d}_sep.npy")[0])
        for k_set_value in range(1, 21)
    }
    anchor_info = {}
    for fps_order_value, configuration_id in zip(accrete_table.column("fps_order").to_numpy(),
                                                 accrete_table.column("configuration_id").to_pylist()):
        k_set_value = compute_k_set(int(fps_order_value))
        anchor_info[configuration_id] = (int(fps_order_value), separation_by_k_set[k_set_value], k_set_value)
    del accrete_table
    print(f"accrete anchor cids: {len(anchor_info):,}", flush=True)

    cid_index_table = pq.read_table(f"{run_dir}/out/cid_index_25M.parquet")
    seen_ids, shard_row_pairs, external_families = set(), defaultdict(list), defaultdict(set)
    for configuration_id, family, shard_path, row_index in zip(
            cid_index_table.column("configuration_id").to_pylist(),
            cid_index_table.column("family").to_pylist(),
            cid_index_table.column("shard").to_pylist(),
            cid_index_table.column("row").to_numpy()):
        if configuration_id not in seen_ids:
            seen_ids.add(configuration_id)
            shard_row_pairs[shard_path].append((int(row_index), configuration_id))
        if not is_mad_family(family):
            external_families[configuration_id].add(family)
    del cid_index_table

    family_of_cid = {}
    for configuration_id in seen_ids:
        external = external_families.get(configuration_id)
        family_of_cid[configuration_id] = sorted(external)[0] if external else "mad-union"
    del external_families

    # anchor picks are walked in ACCRETE-hierarchy order (accrete.parquet fps_order + its
    pick_metadata = {}
    unresolved = 0
    for configuration_id, (fps_order, epsilon_sq, dataset_id) in pick_info.items():
        if configuration_id not in family_of_cid:
            unresolved += 1
            continue
        if configuration_id in anchor_info:
            resolved_fps_order, resolved_epsilon_sq, resolved_k_set = anchor_info[configuration_id]
            source_block = "accrete_anchor"
        else:
            resolved_fps_order = fps_order
            resolved_epsilon_sq = epsilon_sq
            resolved_k_set = compute_k_set(fps_order)
            source_block = "fps_fill"
        pick_metadata[configuration_id] = (
            resolved_fps_order, resolved_epsilon_sq, resolved_k_set,
            source_block, dataset_id, family_of_cid[configuration_id],
        )
    if unresolved:
        print(f"WARNING: {unresolved:,} pick cids not in cid_index (dropped)", flush=True)
    del pick_info, anchor_info, family_of_cid

    num_tasks = sum((len(row_pairs) + chunk_size - 1) // chunk_size for row_pairs in shard_row_pairs.values())
    print(f"{num_picks:,} picks -> {num_tasks:,} bounded tasks on {num_processes} workers -> {output_path}", flush=True)

    def generate_tasks():
        for shard, row_pairs in shard_row_pairs.items():
            for chunk_start in range(0, len(row_pairs), chunk_size):
                chunk = row_pairs[chunk_start:chunk_start + chunk_size]
                yield (shard, [(row_index, configuration_id, pick_metadata[configuration_id])
                               for row_index, configuration_id in chunk if configuration_id in pick_metadata])

    # bounded sliding window keep at most window tasks in flight so the single
    from collections import deque
    window = num_processes * 2
    parquet_writer = pq.ParquetWriter(output_path, output_schema, compression="zstd")
    rows_written = 0
    tasks = generate_tasks()
    pending = deque()

    def drain_one():
        nonlocal rows_written
        task_rows = pending.popleft().get()
        if task_rows:
            parquet_writer.write_table(pa.Table.from_pylist(task_rows, schema=output_schema))
            rows_written += len(task_rows)
            if rows_written % 2_000_000 < len(task_rows):
                print(f"  written {rows_written:,}/{num_picks:,}", flush=True)

    with Pool(num_processes, maxtasksperchild=200) as pool:
        for task in tasks:
            pending.append(pool.apply_async(read_task, (task,)))
            if len(pending) >= window:
                drain_one()
        while pending:
            drain_one()
    parquet_writer.close()
    print(f"wrote {rows_written:,} rows -> {output_path}  (shard order; sort by fps_order for pick order)  DONE", flush=True)


if __name__ == "__main__":
    main()
