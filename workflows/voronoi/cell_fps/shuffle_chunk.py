import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCALER = "${STORE_ROOT}/mad-core/release/ecumetric_scaler.npz"
N_CELLS = 2 ** 24

SCHEMA = pa.schema([
    ("cell", pa.int32()),
    ("d24", pa.float32()),
    ("source_row", pa.int64()),
    ("configuration_id", pa.string()),
    ("dataset_group", pa.string()),
    ("dataset_id", pa.string()),
    ("features", pa.list_(pa.float32(), 512)),
])


def load_chunk(source, start, stop, mean, inv_scale):
    path = source["path"]
    n = stop - start

    ns = source["ns"]
    a24 = np.array(np.memmap(f"{path}/voronoi_a.bin", dtype=np.int32,
                             mode="r", shape=(ns, 25))[start:stop, 24])
    d24 = np.array(np.memmap(f"{path}/voronoi_d.bin", dtype=np.float32,
                             mode="r", shape=(ns, 25))[start:stop, 24])

    features = np.array(np.memmap(f"{path}/ecumetric_features.bin", dtype=np.float32,
                                  mode="r", shape=(ns, 512))[start:stop])
    features -= mean
    features *= inv_scale

    cid = np.load(f"{path}/configuration_id.npy", mmap_mode="r")[start:stop].astype("U")
    dataset_id = np.load(f"{path}/dataset_id.npy", mmap_mode="r")[start:stop].astype("U")

    if source["tier"] == "collection":
        filter_code = np.load(f"{path}/filter_code.npy", mmap_mode="r")[start:stop]
        is_duplicate = np.load(f"{path}/is_duplicate.npy", mmap_mode="r")[start:stop]
        clean = (np.asarray(filter_code) == 0) & ~np.asarray(is_duplicate)
        group = np.full(n, source["name"])
    else:
        clean = np.ones(n, dtype=bool)
        group = np.load(f"{path}/dataset_group.npy", mmap_mode="r")[start:stop].astype("U")

    source_row = np.arange(start, stop, dtype=np.int64)
    return a24, d24, features, cid, group, dataset_id, source_row, clean


def group_table(order, a24, d24, features, cid, group, dataset_id, source_row):
    flat = pa.array(features[order].reshape(-1))
    return pa.table({
        "cell": pa.array(a24[order]),
        "d24": pa.array(d24[order]),
        "source_row": pa.array(source_row[order]),
        "configuration_id": pa.array(cid[order]),
        "dataset_group": pa.array(group[order]),
        "dataset_id": pa.array(dataset_id[order]),
        "features": pa.FixedSizeListArray.from_arrays(flat, 512),
    }, schema=SCHEMA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    args = parser.parse_args()

    with open(args.plan) as fh:
        plan = json.load(fh)
    out = plan["out"]

    log_dir = f"{out}/shuffle/logs"
    os.makedirs(log_dir, exist_ok=True)
    done_marker = f"{log_dir}/task_{args.task_id:04d}.done"
    if os.path.exists(done_marker):
        print(f"task {args.task_id}: already done")
        return

    scaler = np.load(SCALER)
    mean = scaler["mean"].astype(np.float32)
    inv_scale = np.float32(1.0 / float(scaler["scale"]))

    n_buckets = plan["n_buckets"]
    is_mega = np.zeros(N_CELLS, dtype=bool)
    is_mega[np.array(plan["mega_cells"], dtype=np.int64)] = True

    writers = {}

    def writer_for(key_kind, key_value):
        key = (key_kind, key_value)
        if key not in writers:
            if key_kind == "bucket":
                part_dir = f"{out}/shuffle/buckets/b{key_value:03d}"
            else:
                part_dir = f"{out}/shuffle/cells/c{key_value:08d}"
            os.makedirs(part_dir, exist_ok=True)
            writers[key] = pq.ParquetWriter(f"{part_dir}/part_{args.task_id:04d}.parquet", SCHEMA)
        return writers[key]

    cell_counts = np.zeros(N_CELLS, dtype=np.int64)
    rows_seen = rows_clean = rows_mega = 0
    my_chunks = plan["chunks"][args.task_id::args.num_tasks]

    for chunk_index, (source_index, start, stop) in enumerate(my_chunks):
        t0 = time.time()
        source = plan["sources"][source_index]
        a24, d24, features, cid, group, dataset_id, source_row, clean = load_chunk(
            source, start, stop, mean, inv_scale)

        rows_seen += len(a24)
        keep = np.where(clean)[0]
        if len(keep) == 0:
            continue
        a24, d24, source_row = a24[keep], d24[keep], source_row[keep]
        cid, group, dataset_id = cid[keep], group[keep], dataset_id[keep]
        features = features[keep]
        rows_clean += len(keep)

        cell_counts += np.bincount(a24, minlength=N_CELLS)

        mega_row = is_mega[a24]
        routing_key = np.where(mega_row, 2 ** 32 + a24.astype(np.int64),
                               a24.astype(np.int64) % n_buckets)
        perm = np.argsort(routing_key, kind="stable")
        sorted_keys = routing_key[perm]
        boundaries = np.concatenate([[0], np.where(np.diff(sorted_keys) != 0)[0] + 1,
                                     [len(sorted_keys)]])

        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            key = int(sorted_keys[lo])
            order = perm[lo:hi]
            table = group_table(order, a24, d24, features, cid, group, dataset_id, source_row)
            if key >= 2 ** 32:
                writer_for("cell", key - 2 ** 32).write_table(table)
                rows_mega += hi - lo
            else:
                writer_for("bucket", key).write_table(table)

        print(f"task {args.task_id}: chunk {chunk_index + 1}/{len(my_chunks)} "
              f"{source['name']}[{start}:{stop}] clean {len(keep):,} "
              f"({time.time() - t0:.1f}s)", flush=True)

    for writer in writers.values():
        writer.close()

    np.save(f"{log_dir}/cell_counts_{args.task_id:04d}.npy", cell_counts.astype(np.uint32))
    with open(f"{log_dir}/task_{args.task_id:04d}.json", "w") as fh:
        json.dump({"task_id": args.task_id, "chunks": len(my_chunks),
                   "rows_seen": int(rows_seen), "rows_clean": int(rows_clean),
                   "rows_mega": int(rows_mega)}, fh)
    with open(done_marker, "w") as fh:
        fh.write("done\n")

    print(f"task {args.task_id}: DONE seen {rows_seen:,} clean {rows_clean:,} "
          f"mega {rows_mega:,}", flush=True)


if __name__ == "__main__":
    main()
