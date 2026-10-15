import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RELEASE = "${STORE_ROOT}/mad-core/release"
FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
SMALL_ROOT = "${STORE_ROOT}/mad-core/accrete_pools/madcore_prod/small"
ID_MAPS = "${STORE_ROOT}/mad-core/id_maps"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
POP = f"{HERE}/out/population"
N_ANCHOR = 856012


def pool_shards(kind, fam):
    root = f"{FPS_ROOT}/{fam}/with_features" if kind == "big" else f"{SMALL_ROOT}/{fam}"
    return sorted(glob.glob(f"{root}/*.parquet"))


def map_first_rows(map_path, needed):
    value_set = pa.array(list(set(needed)), type=pa.binary())
    reader = pq.ParquetFile(map_path)
    lookup = {}
    for group in range(reader.num_row_groups):
        table = reader.read_row_group(group, columns=["old_cid", "first_row"])
        hit = table.filter(pc.is_in(table.column("old_cid"), value_set=value_set))
        lookup.update(zip(hit.column("old_cid").to_pylist(), hit.column("first_row").to_numpy().astype(np.int64)))
    return lookup


def pool_family(task):
    kind, fam = task
    shards = pool_shards(kind, fam)
    old = []
    shard_sizes = []
    for p in shards:
        cids = pq.read_table(p, columns=["configuration_id"]).column("configuration_id").to_pylist()
        shard_sizes.append(len(cids))
        old.extend(c.encode() for c in cids)

    map_path = f"{ID_MAPS}/{fam}.parquet"
    if os.path.exists(map_path):
        lookup = map_first_rows(map_path, old)
        mapping = "id_map"
    else:
        rel = np.load(f"{RELEASE}/collection/{fam}/configuration_id.npy")
        lookup = {}
        for i, c in enumerate(rel.tolist()):
            lookup.setdefault(c, i)
        mapping = "identity_first_occurrence"
    first_row = np.fromiter((lookup.get(c, -1) for c in old), dtype=np.int64, count=len(old))
    misses = int((first_row < 0).sum())
    rows = np.unique(first_row[first_row >= 0])

    np.save(f"{POP}/collection__{fam}.npy", rows)
    np.save(f"{POP}/pool_row_to_release__{fam}.npy", first_row)
    np.save(f"{POP}/pool_shard_sizes__{fam}.npy", np.asarray(shard_sizes, dtype=np.int64))
    dup = np.load(f"{RELEASE}/collection/{fam}/is_duplicate.npy")
    fc = np.load(f"{RELEASE}/collection/{fam}/filter_code.npy")

    return fam, {"kind": kind, "mapping": mapping, "pool_rows": int(len(old)), "pool_shards": len(shards),
                 "id_misses": misses, "release_rows_in_population": int(rows.size),
                 "release_rows_in_population_not_duplicate": int((~dup[rows]).sum()),
                 "release_rows_filter_code_0": int((fc == 0).sum())}


def main():
    os.makedirs(POP, exist_ok=True)
    fams = json.load(open(f"{HERE}/out/pool_families.json"))

    with Pool(24) as pool:
        small = dict(pool.imap_unordered(pool_family, [("small", f) for f in fams["small_families"]]))
    print(f"small families done: {len(small)}", flush=True)

    with Pool(10) as pool:
        big = dict(pool.imap_unordered(pool_family, [("big", f) for f in fams["big_families"]]))
    print(f"big families done: {len(big)}", flush=True)

    np.save(f"{POP}/mad-next-coreset.npy", np.arange(N_ANCHOR, dtype=np.int64))

    small_rows = sum(v["pool_rows"] for v in small.values())
    big_rows = sum(v["pool_rows"] for v in big.values())
    summary = {
        "small_pool_rows": small_rows, "small_pool_target": 83386207,
        "big_pool_rows": big_rows, "big_pool_target": 99938439,
        "id_misses": sum(v["id_misses"] for v in list(small.values()) + list(big.values())),
        "release_rows_small": sum(v["release_rows_in_population"] for v in small.values()),
        "release_rows_big": sum(v["release_rows_in_population"] for v in big.values()),
        "anchor_rows": N_ANCHOR, "pool_rows_total": small_rows + big_rows + N_ANCHOR,
        "accrete_input_target": 184180658,
        "families": {**small, **big},
    }
    json.dump(summary, open(f"{HERE}/out/population_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
