import glob
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RELEASE = "${STORE_ROOT}/mad-core/release"
OUT = "${DEV_ROOT}/phace-atomic/voronoi/out/cell_fps"

OUT_SCHEMA = pa.schema([
    ("cell", pa.int32()),
    ("cell_rank", pa.int32()),
    ("eps_sq", pa.float32()),
    ("configuration_id", pa.string()),
    ("dataset_group", pa.string()),
    ("dataset_id", pa.string()),
])


def collect_duplicates():
    groups = sorted(glob.glob(f"{RELEASE}/collection/*"))
    cid_parts, group_parts, dataset_parts = [], [], []
    for path in groups:
        dup = np.load(f"{path}/is_duplicate.npy", mmap_mode="r")
        rows = np.where(np.asarray(dup))[0]
        if len(rows) == 0:
            continue
        cid_parts.append(np.load(f"{path}/configuration_id.npy",
                                 mmap_mode="r")[rows].astype("U"))
        dataset_parts.append(np.load(f"{path}/dataset_id.npy",
                                     mmap_mode="r")[rows].astype("U"))
        group_parts.append(np.full(len(rows), path.rsplit("/", 1)[1]))
    return (np.concatenate(cid_parts), np.concatenate(group_parts),
            np.concatenate(dataset_parts))


def main():
    dup_cid, dup_group, dup_dataset = collect_duplicates()
    print(f"duplicate rows: {len(dup_cid):,}", flush=True)

    twin = {}
    wanted = pa.array(np.unique(dup_cid))
    ranked_paths = sorted(glob.glob(f"{OUT}/fps/b*.parquet")) \
        + sorted(glob.glob(f"{OUT}/fps/cell_*.parquet"))
    for path in ranked_paths:
        table = pq.read_table(path, columns=["cell", "cell_rank", "eps_sq",
                                             "configuration_id"])
        mask = pc.is_in(table.column("configuration_id"), value_set=wanted)
        hits_table = table.filter(mask)
        cid = np.asarray(hits_table.column("configuration_id").to_pandas(), dtype=object)
        cell = hits_table.column("cell").to_numpy()
        rank = hits_table.column("cell_rank").to_numpy()
        eps = hits_table.column("eps_sq").to_numpy()
        for i in range(len(cid)):
            twin[cid[i]] = (int(cell[i]), int(rank[i]), float(eps[i]))
    print(f"twins found for {len(twin):,} distinct duplicated cids", flush=True)

    matched = np.array([c in twin for c in dup_cid])
    hits = np.where(matched)[0]
    cells = np.array([twin[dup_cid[i]][0] for i in hits], dtype=np.int32)
    ranks = np.array([twin[dup_cid[i]][1] for i in hits], dtype=np.int32)
    eps = np.array([twin[dup_cid[i]][2] for i in hits], dtype=np.float32)

    order = np.lexsort((ranks, cells))
    table = pa.table({
        "cell": pa.array(cells[order]),
        "cell_rank": pa.array(ranks[order]),
        "eps_sq": pa.array(eps[order]),
        "configuration_id": pa.array(dup_cid[hits][order]),
        "dataset_group": pa.array(dup_group[hits][order]),
        "dataset_id": pa.array(dup_dataset[hits][order]),
    }, schema=OUT_SCHEMA)
    pq.write_table(table, f"{OUT}/fps/duplicates_inherited.parquet")

    summary = {"duplicate_rows": int(len(dup_cid)),
               "inherited": int(matched.sum()),
               "unmatched_twin_filtered": int((~matched).sum())}
    with open(f"{OUT}/fps/duplicates_inherited.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"DONE {summary}", flush=True)


if __name__ == "__main__":
    main()
