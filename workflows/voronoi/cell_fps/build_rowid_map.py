import argparse
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_CELLS = 2 ** 24

CATEGORY_CLEAN = 0
CATEGORY_DUPLICATE = 1
CATEGORY_FILTERED = 2

SCHEMA = pa.schema([
    ("cell", pa.int32()),
    ("row_id", pa.int64()),
    ("category", pa.int8()),
    ("configuration_id", pa.string()),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    args = parser.parse_args()

    with open(args.plan) as fh:
        plan = json.load(fh)

    sources = plan["sources"]
    source_offsets = np.concatenate([[0], np.cumsum([s["ns"] for s in sources])])

    n_buckets = plan["n_buckets"]
    is_mega = np.zeros(N_CELLS, dtype=bool)
    is_mega[np.array(plan["mega_cells"], dtype=np.int64)] = True

    log_dir = f"{args.out}/logs"
    os.makedirs(log_dir, exist_ok=True)
    done_marker = f"{log_dir}/task_{args.task_id:04d}.done"
    if os.path.exists(done_marker):
        print(f"task {args.task_id}: already done")
        return

    writers = {}

    def writer_for(kind, key):
        if (kind, key) not in writers:
            part_dir = (f"{args.out}/buckets/b{key:03d}" if kind == "bucket"
                        else f"{args.out}/cells/c{key:08d}")
            os.makedirs(part_dir, exist_ok=True)
            writers[(kind, key)] = pq.ParquetWriter(
                f"{part_dir}/part_{args.task_id:04d}.parquet", SCHEMA)
        return writers[(kind, key)]

    cell_rows = np.zeros(N_CELLS, dtype=np.int64)
    cell_bytes = np.zeros(N_CELLS, dtype=np.int64)
    counts = {CATEGORY_CLEAN: 0, CATEGORY_DUPLICATE: 0, CATEGORY_FILTERED: 0}
    my_chunks = plan["chunks"][args.task_id::args.num_tasks]

    for chunk_index, (source_index, start, stop) in enumerate(my_chunks):
        source = sources[source_index]
        path, ns = source["path"], source["ns"]

        cells = np.array(np.memmap(f"{path}/voronoi_a.bin", dtype=np.int32,
                                   mode="r", shape=(ns, 25))[start:stop, 24])
        cid = np.load(f"{path}/configuration_id.npy", mmap_mode="r")[start:stop]

        if source["tier"] == "collection":
            filter_code = np.asarray(np.load(f"{path}/filter_code.npy",
                                             mmap_mode="r")[start:stop])
            duplicate = np.asarray(np.load(f"{path}/is_duplicate.npy",
                                           mmap_mode="r")[start:stop])
            category = np.where(duplicate, CATEGORY_DUPLICATE,
                                np.where(filter_code == 0, CATEGORY_CLEAN,
                                         CATEGORY_FILTERED)).astype(np.int8)
        else:
            category = np.full(stop - start, CATEGORY_CLEAN, dtype=np.int8)

        row_id = source_offsets[source_index] + np.arange(start, stop, dtype=np.int64)
        cid_str = cid.astype("U")

        # every release row is indexed, so both tallies cover all categories
        cell_rows += np.bincount(cells, minlength=N_CELLS)
        lengths = np.char.str_len(cid_str).astype(np.int64)
        cell_bytes += np.bincount(cells, weights=lengths,
                                  minlength=N_CELLS).astype(np.int64)
        for key in counts:
            counts[key] += int((category == key).sum())

        routing = np.where(is_mega[cells], 2 ** 32 + cells.astype(np.int64),
                           cells.astype(np.int64) % n_buckets)
        order = np.argsort(routing, kind="stable")
        keys = routing[order]
        bounds = np.concatenate([[0], np.where(np.diff(keys) != 0)[0] + 1, [len(keys)]])

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            key = int(keys[lo])
            rows = order[lo:hi]
            table = pa.table({
                "cell": pa.array(cells[rows]),
                "row_id": pa.array(row_id[rows]),
                "category": pa.array(category[rows]),
                "configuration_id": pa.array(cid_str[rows]),
            }, schema=SCHEMA)
            if key >= 2 ** 32:
                writer_for("cell", key - 2 ** 32).write_table(table)
            else:
                writer_for("bucket", key).write_table(table)

        print(f"task {args.task_id}: chunk {chunk_index + 1}/{len(my_chunks)} "
              f"{source['name']}[{start}:{stop}]", flush=True)

    for writer in writers.values():
        writer.close()

    np.save(f"{log_dir}/cell_rows_{args.task_id:04d}.npy", cell_rows)
    np.save(f"{log_dir}/cell_bytes_{args.task_id:04d}.npy", cell_bytes)
    with open(f"{log_dir}/task_{args.task_id:04d}.json", "w") as fh:
        json.dump({"task_id": args.task_id,
                   "clean": counts[CATEGORY_CLEAN],
                   "duplicate": counts[CATEGORY_DUPLICATE],
                   "filtered": counts[CATEGORY_FILTERED]}, fh)
    with open(done_marker, "w") as fh:
        fh.write("done\n")

    print(f"task {args.task_id}: DONE {counts}", flush=True)


if __name__ == "__main__":
    main()
