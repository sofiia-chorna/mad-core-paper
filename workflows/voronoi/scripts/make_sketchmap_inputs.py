import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
coreset_path = ("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92"
                "/coreset224/out/coreset_224_full.parquet")
out_dir = f"{voronoi_dir}/out/analysis"

LEVEL = 20
N_LEVEL = 2 ** LEVEL
N_SAMPLE = 65_536
TOTAL_ROWS = 1_695_275_276
SEED = 0


def main():
    pops = np.load(f"{voronoi_dir}/out/analysis/populations.npz")
    offsets = pops["level_offsets"]
    counts20 = pops["populations"][offsets[LEVEL]:offsets[LEVEL + 1]].astype(np.int64)
    assert counts20.shape == (N_LEVEL,), counts20.shape
    assert counts20.sum() == TOTAL_ROWS, counts20.sum()
    np.save(f"{out_dir}/populations_level20.npy", counts20)
    print(f"[1] populations_level20.npy  shape={counts20.shape} sum={counts20.sum():,} "
          f"min={counts20.min():,} max={counts20.max():,}", flush=True)

    table = pq.read_table(coreset_path, columns=["fps_order", "family", "dataset_id"])
    fps_order = table.column("fps_order").to_numpy()
    order = np.argsort(fps_order, kind="stable")
    assert np.array_equal(fps_order[order], np.arange(2 ** 24)), "fps_order is not a 0..2^24-1 permutation"
    family_all = np.asarray(table.column("family").to_pylist(), dtype=object)[order]
    dataset_all = np.asarray(table.column("dataset_id").to_pylist(), dtype=object)[order]

    family20 = family_all[:N_LEVEL]
    dataset20 = dataset_all[:N_LEVEL]
    np.save(f"{out_dir}/family_level20.npy", family20)
    np.save(f"{out_dir}/dataset_id_level20.npy", dataset20)
    fam, fam_counts = np.unique(family20, return_counts=True)
    print(f"[2] family_level20.npy  n={family20.shape[0]:,}  distinct families={fam.shape[0]}", flush=True)
    for f, c in sorted(zip(fam.tolist(), fam_counts.tolist()), key=lambda x: -x[1])[:10]:
        print(f"       {c:>8,}  {100 * c / N_LEVEL:5.2f}%  {f}", flush=True)

    feats = np.memmap(f"{voronoi_dir}/out/tree/center_feats_scaled.f32",
                      dtype=np.float32, mode="r", shape=(2 ** 24, 512))
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(N_LEVEL, N_SAMPLE, replace=False))
    sample_feats = np.asarray(feats[idx])

    sample = pa.table({
        "coreset_index": pa.array(idx.astype(np.int64)),
        "feat_scaled": pa.array(list(sample_feats), pa.list_(pa.float32(), 512)),
        "voronoi_count_level20": pa.array(counts20[idx]),
        "family": pa.array(family20[idx].tolist(), pa.string()),
        "dataset_id": pa.array(dataset20[idx].tolist(), pa.string()),
    })
    pq.write_table(sample, f"{out_dir}/coreset_sample_65536_of_2p20.parquet", compression="zstd")
    print(f"[3] coreset_sample_65536_of_2p20.parquet  rows={sample.num_rows:,} "
          f"weight_sum={counts20[idx].sum():,}", flush=True)

    top = np.sort(counts20)[::-1]
    pair = counts20.astype(np.float64) ** 2
    print(f"\nlevel-20 concentration (the raw-vs-log decision):")
    print(f"   largest cell            : {top[0]:,} points ({100 * top[0] / TOTAL_ROWS:.3f}%)")
    print(f"   top-10 cells            : {100 * top[:10].sum() / TOTAL_ROWS:.3f}% of points")
    print(f"   top-1%% cells           : {100 * top[:N_LEVEL // 100].sum() / TOTAL_ROWS:.3f}% of points")
    print(f"   empty cells             : {int((counts20 == 0).sum()):,}")
    print(f"   largest cell pair-weight: {100 * pair.max() / pair.sum():.3f}% of total n^2")
    print(f"   top-10 pair-weight      : {100 * np.sort(pair)[::-1][:10].sum() / pair.sum():.3f}%", flush=True)


if __name__ == "__main__":
    main()
