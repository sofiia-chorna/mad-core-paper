import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

OUT_ROOT = ("${DEV_ROOT}/phace-atomic/voronoi/out/"
            "cell_fps")
ROWID = f"{OUT_ROOT}/rowid"
FPS = f"{OUT_ROOT}/fps"
N_CELLS = 2 ** 24
UNORDERED = -1


def source_offsets(plan):
    return np.concatenate([[0], np.cumsum([s["ns"] for s in plan["sources"]])])


def load_rowid(part_dir):
    paths = sorted(glob.glob(f"{part_dir}/part_*.parquet"))
    tables = [pq.read_table(p, columns=["cell", "row_id", "configuration_id"])
              for p in paths]
    cell = np.concatenate([t.column("cell").to_numpy() for t in tables])
    row_id = np.concatenate([t.column("row_id").to_numpy() for t in tables])
    cid = np.concatenate([np.asarray(t.column("configuration_id").to_pandas(),
                                     dtype=object) for t in tables])
    return cell, row_id, cid


def load_orders(name, is_cell_dir):
    """(cell, configuration_id) -> (fps_order_in_cell, epsilon_sq) from the FPS output.

    Keyed by cell as well as id: a duplicated row carries ~1e-7 of featurisation
    jitter and can land in a different cell from its twin. Keying on the id alone
    lets such a row inherit its twin's rank, which is a position in another cell
    and meaningless here — FORMAT.md specifies -1 for that case.
    """
    path = f"{FPS}/cell_{name}.parquet" if is_cell_dir else f"{FPS}/{name}.parquet"
    if not os.path.exists(path):
        return {}
    table = pq.read_table(path, columns=["cell", "cell_rank", "eps_sq",
                                         "configuration_id"])
    cell = table.column("cell").to_numpy()
    rank = table.column("cell_rank").to_numpy()
    eps = table.column("eps_sq").to_numpy()
    cid = np.asarray(table.column("configuration_id").to_pandas(), dtype=object)
    return {(int(cell[i]), cid[i]): (int(rank[i]), float(eps[i]))
            for i in range(len(cid))}


def process(part_dir, name, is_cell_dir, offsets, dir_offsets, targets,
            deleted_sources=()):
    cell, row_id, cid = load_rowid(part_dir)
    orders = load_orders(name, is_cell_dir)

    n = len(cell)
    order = np.full(n, UNORDERED, dtype=np.int32)
    eps = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        hit = orders.get((int(cell[i]), cid[i]))
        if hit is not None:
            order[i], eps[i] = hit

    directory = (np.searchsorted(dir_offsets, row_id, "right") - 1).astype(np.int16)
    local = (row_id - dir_offsets[directory]).astype(np.int32)

    # sources deleted from the release keep their plan entry (row_id offsets stay
    # valid) but ship no rows; shift directory indices past them so they match
    # release_directory_names.npy, which omits them
    for dead in sorted(deleted_sources):
        if (directory == dead).any():
            raise RuntimeError(f"row references deleted source {dead}")
    for dead in sorted(deleted_sources, reverse=True):
        directory = np.where(directory > dead, directory - 1, directory).astype(np.int16)

    # cell-major, ordered members first (-1 last), ties broken by release row
    sort_key = np.where(order < 0, np.iinfo(np.int32).max, order).astype(np.int64)
    permutation = np.lexsort((row_id, sort_key, cell))

    cell, order, eps = cell[permutation], order[permutation], eps[permutation]
    directory, local = directory[permutation], local[permutation]
    cid = cid[permutation]

    # every cell of this partition is wholly owned by it, so spans are disjoint
    boundaries = np.concatenate([[0], np.where(np.diff(cell) != 0)[0] + 1, [n]])
    written = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        c = int(cell[lo])
        start = int(offsets[c])
        if int(offsets[c + 1]) - start != hi - lo:
            raise RuntimeError(f"cell {c}: expected {int(offsets[c+1]) - start} rows, "
                               f"partition holds {hi - lo}")
        targets["fps_order_in_cell"][start:start + (hi - lo)] = order[lo:hi]
        targets["epsilon_sq"][start:start + (hi - lo)] = eps[lo:hi]
        targets["release_directory"][start:start + (hi - lo)] = directory[lo:hi]
        targets["release_row_in_directory"][start:start + (hi - lo)] = local[lo:hi]
        written += hi - lo

    lengths = np.array([len(v.encode()) for v in cid], dtype=np.int64)
    return written, cell, cid, lengths, boundaries, permutation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    with open(f"{OUT_ROOT}/plan.json") as fh:
        plan = json.load(fh)
    dir_offsets = source_offsets(plan)
    deleted_sources = tuple(plan.get("deleted_sources", {}).get("indices", ()))

    offsets = np.load(f"{args.out}/center_offsets.npy", mmap_mode="r")

    partitions = ([(p, os.path.basename(p), False)
                   for p in sorted(glob.glob(f"{ROWID}/buckets/b*"))]
                  + [(p, os.path.basename(p), True)
                     for p in sorted(glob.glob(f"{ROWID}/cells/c*"))])
    mine = partitions[args.task_id::args.num_tasks]

    if args.apply:
        targets = {name: np.lib.format.open_memmap(f"{args.out}/{name}.npy", mode="r+")
                   for name in ("fps_order_in_cell", "epsilon_sq",
                                "release_directory", "release_row_in_directory")}
    else:
        targets = {k: np.zeros(1, dtype=np.int64) for k in
                   ("fps_order_in_cell", "epsilon_sq", "release_directory",
                    "release_row_in_directory")}

    os.makedirs(f"{args.out}/logs", exist_ok=True)
    marker = f"{args.out}/logs/build_{args.task_id:04d}.done"
    if args.apply and os.path.exists(marker):
        print(f"task {args.task_id}: already done", flush=True)
        return

    summary = {"rows": 0, "ordered": 0, "id_bytes": 0, "partitions": 0}
    for index, (part_dir, name, is_cell) in enumerate(mine):
        if args.apply:
            written, cell, cid, lengths, _, _ = process(
                part_dir, name, is_cell, offsets, dir_offsets, targets,
                deleted_sources)
            np.savez(f"{args.out}/ids/part_{args.task_id:04d}_{index:04d}.npz",
                     cell=cell, lengths=lengths,
                     payload=np.frombuffer(b"".join(v.encode() for v in cid),
                                           dtype=np.uint8))
        else:
            cell, row_id, cid = load_rowid(part_dir)
            orders = load_orders(name, is_cell)
            written = len(cell)
            lengths = np.array([len(v.encode()) for v in cid], dtype=np.int64)
            summary["ordered"] += sum(1 for v in cid if v in orders)
        summary["rows"] += int(written)
        summary["id_bytes"] += int(lengths.sum())
        summary["partitions"] += 1
        if index % 10 == 0:
            print(f"task {args.task_id}: {index + 1}/{len(mine)} {name} "
                  f"rows={summary['rows']:,}", flush=True)

    with open(f"{args.out}/logs/build_{args.task_id:04d}.json", "w") as fh:
        json.dump(summary, fh)
    if args.apply:
        with open(marker, "w") as fh:
            fh.write("done\n")
    print(f"task {args.task_id}: {summary}", flush=True)


if __name__ == "__main__":
    main()
