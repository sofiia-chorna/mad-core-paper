import os
import pickle
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
manifest_path = f"{voronoi_dir}/out/manifest_all.txt"
assign_dir = f"{voronoi_dir}/out/assign"
scaler_path = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
ahmetcik_path = "${DEV_ROOT}/phace-atomic/accrete/runs/no_ahm/ahmetcik_entries.txt"
out_path = f"{voronoi_dir}/out/random_sample_2p20.parquet"

N_TARGET = 2 ** 20
DRAW_MARGIN = 1.02
SEED = 0
MAX_LEVEL = 24
NPROC = int(os.environ.get("SAMPLE_NPROC", "128"))

shards = [line.strip() for line in open(manifest_path) if line.strip()]


def assign_path(shard_path):
    if "/phace_atomic_ep92/" in shard_path:
        relative = shard_path.split("/phace_atomic_ep92/")[1]
        family = relative.split("/")[0]
        stem = relative.replace("/", "__")
        if stem.endswith(".parquet"):
            stem = stem[:-len(".parquet")]
    else:
        family = "mad_union"
        base = os.path.basename(shard_path)
        if base.endswith(".parquet"):
            base = base[:-len(".parquet")]
        stem = "mad_union__" + base
    return f"{assign_dir}/{family}/{stem}.parquet", family


def count_rows(shard_path):
    return pq.ParquetFile(shard_path).metadata.num_rows


def row_group_of(metadata, wanted_rows):
    bounds = np.cumsum([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)])
    groups = np.searchsorted(bounds, wanted_rows, side="right")
    starts = bounds - np.array([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)])
    return groups, wanted_rows - starts[groups]


def fetch(task):
    shard_index, wanted_rows = task
    shard_path = shards[shard_index]
    wanted_rows = np.sort(np.asarray(wanted_rows))

    source_file = pq.ParquetFile(shard_path)
    names = set(source_file.schema_arrow.names)
    source_columns = ["configuration_id", "phace_features_bb_mean"]
    if "dataset_id" in names:
        source_columns.append("dataset_id")
    groups, offsets = row_group_of(source_file.metadata, wanted_rows)

    feats, cids, dsids = [], [], []
    for group in np.unique(groups):
        mask = groups == group
        table = source_file.read_row_group(int(group), columns=source_columns)
        take = pa.array(offsets[mask])
        sub = table.take(take)
        block = np.asarray(sub.column("phace_features_bb_mean").combine_chunks()
                           .values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, 512)
        feats.append(block)
        cids.extend(sub.column("configuration_id").to_pylist())
        dsids.extend(sub.column("dataset_id").to_pylist() if "dataset_id" in names
                     else [None] * sub.num_rows)
    feats = np.concatenate(feats, axis=0)

    out_file_path, family = assign_path(shard_path)
    assign_file = pq.ParquetFile(out_file_path)
    a_cols = [f"a{k:02d}" for k in range(MAX_LEVEL + 1)]
    d_cols = [f"d{k:02d}" for k in range(MAX_LEVEL + 1)]
    c_cols = [f"c{k:02d}" for k in range(MAX_LEVEL + 1)]
    groups, offsets = row_group_of(assign_file.metadata, wanted_rows)

    parts = []
    for group in np.unique(groups):
        mask = groups == group
        table = assign_file.read_row_group(int(group), columns=["row"] + a_cols + d_cols + c_cols)
        parts.append(table.take(pa.array(offsets[mask])))
    assign_table = pa.concat_tables(parts)

    check = assign_table.column("row").to_numpy()
    if not np.array_equal(check, wanted_rows):
        raise RuntimeError(f"row misalignment in {out_file_path}: {check[:5]} vs {wanted_rows[:5]}")

    return {
        "shard_index": shard_index,
        "family": family,
        "rows": wanted_rows,
        "cids": cids,
        "dsids": dsids,
        "feats": feats,
        "assign": assign_table,
    }


def main():
    with Pool(NPROC) as pool:
        counts = np.array(pool.map(count_rows, shards, chunksize=16), dtype=np.int64)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    print(f"shards={len(shards):,}  total rows={total:,}", flush=True)

    n_draw = int(N_TARGET * DRAW_MARGIN)
    rng = np.random.default_rng(SEED)
    draws = np.unique(rng.integers(0, total, size=int(n_draw * 1.05)))
    while draws.shape[0] < n_draw:
        draws = np.unique(np.concatenate([draws, rng.integers(0, total, size=n_draw)]))
    draws = np.sort(rng.permutation(draws)[:n_draw])
    print(f"drew {draws.shape[0]:,} distinct global indices from [0,{total:,}) "
          f"(margin over {N_TARGET:,} to absorb NaN-feature rows)", flush=True)

    shard_index = np.searchsorted(cumulative, draws, side="right")
    starts = cumulative - counts
    local_rows = draws - starts[shard_index]

    tasks = {}
    for si, lr in zip(shard_index, local_rows):
        tasks.setdefault(int(si), []).append(int(lr))
    print(f"touching {len(tasks):,} shards with {NPROC} workers", flush=True)

    with Pool(NPROC) as pool:
        results = pool.map(fetch, list(tasks.items()), chunksize=1)

    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    mean = np.asarray(scaler["mean_"], dtype=np.float32)
    scale = float(scaler["scale_"])

    feats = np.concatenate([r["feats"] for r in results], axis=0)
    cids = [c for r in results for c in r["cids"]]
    dsids = [d for r in results for d in r["dsids"]]
    families = [r["family"] for r in results for _ in range(len(r["cids"]))]
    src = [shards[r["shard_index"]] for r in results for _ in range(len(r["cids"]))]
    rows = np.concatenate([r["rows"] for r in results])
    assign_table = pa.concat_tables([r["assign"] for r in results])

    with open(ahmetcik_path) as fh:
        ahmetcik = set(line.strip() for line in fh if line.strip())

    bad_feat = np.isnan(feats).any(1)
    is_ahmetcik = np.fromiter((("/nomad/" in shard) and (cid.split(":", 1)[-1] in ahmetcik)
                               for shard, cid in zip(src, cids)), bool, len(cids))
    valid = np.where(~bad_feat & ~is_ahmetcik)[0]
    print(f"fetched {feats.shape[0]:,}; dropped {int(bad_feat.sum()):,} NaN-feature, "
          f"{int(is_ahmetcik.sum()):,} ahmetcik nomad; {valid.shape[0]:,} valid", flush=True)
    if valid.shape[0] < N_TARGET:
        raise RuntimeError(f"only {valid.shape[0]:,} valid rows, need {N_TARGET:,}; raise DRAW_MARGIN")
    keep = np.sort(rng.choice(valid, N_TARGET, replace=False))

    feats = feats[keep]
    scaled = (feats - mean) / scale
    cids = [cids[i] for i in keep]
    dsids = [dsids[i] for i in keep]
    families = [families[i] for i in keep]
    src = [src[i] for i in keep]
    rows = rows[keep]
    assign_table = assign_table.take(pa.array(keep))

    columns = {
        "configuration_id": pa.array(cids, pa.string()),
        "dataset_id": pa.array(dsids, pa.string()),
        "family": pa.array(families, pa.string()),
        "source_shard": pa.array(src, pa.string()),
        "source_row": pa.array(rows.astype(np.int64)),
        "feat_scaled": pa.array(list(scaled), pa.list_(pa.float32(), 512)),
    }
    for name in assign_table.schema.names:
        if name != "row":
            columns[name] = assign_table.column(name).combine_chunks()

    table = pa.table(columns)
    pq.write_table(table, out_path, compression="zstd")
    print(f"saved {table.num_rows:,} uniform-random structures -> {out_path}", flush=True)
    print(f"families represented: {len(set(families))}", flush=True)


if __name__ == "__main__":
    main()
