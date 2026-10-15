import glob
import json
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
STORE = "${STORE_ROOT}/mad-core/datasets/phace_atomic_ep92"
RELEASE = "${STORE_ROOT}/mad-core/release"
ID_MAPS = "${STORE_ROOT}/mad-core/id_maps"
ANCHOR = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
ACCRETE = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/madcore_856k/out/accrete.parquet"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
OUT = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out"


def features_of(table, column="phace_features_bb_mean"):
    col = table.column(column).combine_chunks()
    return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def release_features(folder):
    n = int(np.load(f"{folder}/ns.npy")[0]) if "collection" in folder else np.load(f"{folder}/configuration_id.npy", mmap_mode="r").shape[0]
    return np.memmap(f"{folder}/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(n, 512))


def compare(a, b):
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"n": int(a.shape[0]), "bitwise_equal_rows": int((diff.max(axis=1) == 0).sum()),
            "max_abs_diff": float(diff.max()), "median_row_max_abs_diff": float(np.median(diff.max(axis=1)))}


def map_first_rows(map_path, needed):
    value_set = pa.array(list(set(needed)), type=pa.binary())
    reader = pq.ParquetFile(map_path)
    lookup = {}
    for group in range(reader.num_row_groups):
        table = reader.read_row_group(group, columns=["old_cid", "first_row"])
        hit = table.filter(pc.is_in(table.column("old_cid"), value_set=value_set))
        lookup.update(zip(hit.column("old_cid").to_pylist(), hit.column("first_row").to_numpy().astype(np.int64)))
    return lookup


def big_family(fam, n_rows=20000):
    shard = sorted(glob.glob(f"{FPS_ROOT}/{fam}/with_features/*.parquet"))[0]
    t = pq.read_table(shard, columns=["configuration_id", "phace_features_bb_mean"]).slice(0, n_rows)
    old = t.column("configuration_id").to_pylist()
    feat = features_of(t)
    old = [c.encode() for c in old]
    lookup = map_first_rows(f"{ID_MAPS}/{fam}.parquet", old)
    rows = np.array([lookup.get(c, -1) for c in old])
    ok = rows >= 0
    rel = release_features(f"{RELEASE}/collection/{fam}")
    got = np.stack([np.asarray(rel[r]) for r in rows[ok]])
    res = compare(feat[ok], got)
    res["id_map_hit_rate"] = float(ok.mean())
    return res


def main():
    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    mean = np.asarray(scaler["mean_"], dtype=np.float64)
    scale = float(scaler["scale_"])
    report = {}

    for fam in ("Alexandria", "gdb13", "OMat24"):
        report[f"big_{fam}_with_features_vs_release"] = big_family(fam)
        print(fam, report[f"big_{fam}_with_features_vs_release"], flush=True)

    ta = pq.read_table(ANCHOR, columns=["configuration_id", "phace_features_bb_mean"])
    anchor_cids = np.asarray(ta.column("configuration_id").to_pylist())
    anchor_feat = features_of(ta)
    tier_cids = np.load(f"{RELEASE}/mad-next-coreset/configuration_id.npy")
    tier_index = {c: i for i, c in enumerate(tier_cids)}
    tier_rows = np.array([tier_index[c.encode()] for c in anchor_cids])
    tier_feat = release_features(f"{RELEASE}/mad-next-coreset")
    tier_got = np.empty_like(anchor_feat)
    for s in range(0, len(tier_rows), 50000):
        idx = tier_rows[s:s + 50000]
        tier_got[s:s + 50000] = np.asarray(tier_feat[np.sort(idx)])[np.argsort(np.argsort(idx))]
    report["anchor_parquet_vs_mad_next_tier_all"] = compare(anchor_feat, tier_got)
    shift = np.linalg.norm((anchor_feat.astype(np.float64) - tier_got) / scale, axis=1)
    report["anchor_scaled_distance_shift"] = {"max": float(shift.max()), "p99": float(np.percentile(shift, 99)),
                                              "median": float(np.median(shift)), "rows_nonzero": int((shift > 0).sum())}
    print("anchor vs tier", report["anchor_parquet_vs_mad_next_tier_all"], report["anchor_scaled_distance_shift"], flush=True)

    acc = pq.read_table(ACCRETE, columns=["fps_order", "global_id"])
    fo = acc.column("fps_order").to_numpy(); gid = acc.column("global_id").to_numpy()
    sel = gid < len(anchor_cids)
    core_feat = release_features(f"{RELEASE}/mad-core")
    core_got = np.stack([np.asarray(core_feat[r]) for r in fo[sel]])
    report["anchor_parquet_vs_mad_core_tier_members"] = compare(anchor_feat[gid[sel]], core_got)
    report["mad_core_vs_mad_next_tier_members_bitwise"] = compare(core_got, tier_got[gid[sel]])
    shift2 = np.linalg.norm((anchor_feat[gid[sel]].astype(np.float64) - core_got) / scale, axis=1)
    report["anchor_members_scaled_distance_shift"] = {"max": float(shift2.max()), "p99": float(np.percentile(shift2, 99)), "median": float(np.median(shift2))}
    print("anchor members vs mad-core", report["anchor_parquet_vs_mad_core_tier_members"], report["anchor_members_scaled_distance_shift"], flush=True)

    shard = sorted(glob.glob(f"{STORE}/nomad/**/*.parquet", recursive=True))[0]
    t = pq.read_table(shard, columns=["configuration_id", "phace_features_bb_mean"]).slice(0, 20000)
    cids = t.column("configuration_id").to_pylist(); feat = features_of(t)
    rel_cids = np.load(f"{RELEASE}/collection/nomad/configuration_id.npy")
    idx = {c: i for i, c in enumerate(rel_cids)}
    rows = np.array([idx.get(c.encode(), -1) for c in cids]); ok = rows >= 0
    rel = release_features(f"{RELEASE}/collection/nomad")
    got = np.stack([np.asarray(rel[r]) for r in rows[ok]])
    report["small_nomad_store_vs_release"] = compare(feat[ok], got); report["small_nomad_store_vs_release"]["cid_hit_rate"] = float(ok.mean())
    print("nomad", report["small_nomad_store_vs_release"], flush=True)

    json.dump(report, open(f"{OUT}/features_identity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
