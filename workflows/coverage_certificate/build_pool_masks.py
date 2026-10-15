import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
RELEASE = "${STORE_ROOT}/mad-core/release"
ID_MAPS = "${STORE_ROOT}/mad-core/id_maps"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
OUT = f"{HERE}/out"


def big_family_rows(fam):
    shards = sorted(glob.glob(f"{FPS_ROOT}/{fam}/with_features/*.parquet"))
    cids = pa.concat_tables([pq.read_table(p, columns=["configuration_id"]) for p in shards]).column("configuration_id")
    cids = pc.cast(cids, pa.binary())
    pool = pa.table({"old_cid": cids})
    n_pool = pool.num_rows

    id_map = pq.read_table(f"{ID_MAPS}/{fam}.parquet", columns=["old_cid", "first_row"])
    joined = pool.join(id_map, keys="old_cid", join_type="left outer")
    first_row = joined.column("first_row").to_numpy(zero_copy_only=False)
    missing = int(np.isnan(first_row.astype(np.float64)).sum()) if first_row.dtype.kind == "f" else int(pc.sum(pc.is_null(joined.column("first_row"))).as_py() or 0)
    rows = first_row[~pd_isnull(first_row)].astype(np.int64)
    rows_unique = np.unique(rows)
    n_release = int(np.load(f"{RELEASE}/collection/{fam}/ns.npy")[0])
    filter_code = np.load(f"{RELEASE}/collection/{fam}/filter_code.npy", mmap_mode="r")
    fc = np.asarray(filter_code[rows_unique])
    return rows_unique, {"family": fam, "pool_rows": n_pool, "id_map_missing": missing, "distinct_release_rows": int(rows_unique.size),
                         "duplicate_first_rows": int(rows.size - rows_unique.size), "release_ns": n_release,
                         "filter_code_counts_on_mask": np.bincount(fc.astype(np.int64), minlength=4).tolist()}


def pd_isnull(a):
    if a.dtype.kind == "f":
        return np.isnan(a)
    return np.zeros(a.shape, dtype=bool)


def main():
    started = time.time()
    families = json.load(open(f"{OUT}/pool_families.json"))
    summary = {"big": [], "small": [], "anchor": None}

    for fam in families["big_families"]:
        t0 = time.time()
        rows, info = big_family_rows(fam)
        np.save(f"{OUT}/masks/{fam}.npy", rows)
        info["seconds"] = round(time.time() - t0, 1)
        summary["big"].append(info)
        print(info, flush=True)

    total_small = 0
    total_small_nodup = 0
    for fam in families["small_families"]:
        fc = np.load(f"{RELEASE}/collection/{fam}/filter_code.npy")
        dup = np.load(f"{RELEASE}/collection/{fam}/is_duplicate.npy")
        rows = np.where(fc == 0)[0]
        np.save(f"{OUT}/masks/{fam}.npy", rows)
        total_small += rows.size
        total_small_nodup += int(((fc == 0) & ~dup).sum())
        summary["small"].append({"family": fam, "rows_filter_ok": int(rows.size), "rows_filter_ok_not_duplicate": int(((fc == 0) & ~dup).sum()), "release_ns": int(fc.size)})

    n_tier = np.load(f"{RELEASE}/mad-next-coreset/configuration_id.npy", mmap_mode="r").shape[0]
    np.save(f"{OUT}/masks/mad-next-coreset.npy", np.arange(n_tier, dtype=np.int64))
    summary["anchor"] = {"rows": int(n_tier)}

    summary["totals"] = {
        "big_distinct_release_rows": int(sum(i["distinct_release_rows"] for i in summary["big"])),
        "big_pool_rows": int(sum(i["pool_rows"] for i in summary["big"])),
        "small_filter_ok": int(total_small), "small_filter_ok_not_duplicate": int(total_small_nodup),
        "expected_big": 99938439, "expected_small": 83386207, "expected_pool": 184180658,
        "wall_seconds": round(time.time() - started, 1),
    }
    json.dump(summary, open(f"{OUT}/pool_masks_summary.json", "w"), indent=1)
    print(summary["totals"], flush=True)


if __name__ == "__main__":
    main()
