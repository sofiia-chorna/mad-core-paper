import glob
import json

import numpy as np
import pyarrow.parquet as pq

FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
RELEASE = "${STORE_ROOT}/mad-core/release"
ANCHOR = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
ACCRETE = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/madcore_856k/out/accrete.parquet"
OUT = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out"
BIG = ["OC20-IS2RES", "Open_Molecular_Crystals_2025_OMC25", "ANI-1", "ANI-2x", "Alexandria",
       "OC20_S2EF_train_all", "ODAC", "OMat24", "OMol25", "gdb13"]


def features_of(table):
    col = table.column("phace_features_bb_mean").combine_chunks()
    return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def main():
    report = {}

    big_rows = {}
    for fam in BIG:
        shards = sorted(glob.glob(f"{FPS_ROOT}/{fam}/with_features/*.parquet"))
        big_rows[fam] = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    report["big_family_rows"] = big_rows
    report["big_total"] = int(sum(big_rows.values()))
    print("big total", report["big_total"], flush=True)

    shard = sorted(glob.glob(f"{FPS_ROOT}/Alexandria/with_features/*.parquet"))[0]
    t = pq.read_table(shard, columns=["configuration_id", "phace_features_bb_mean"])
    shard_cids = np.asarray(t.column("configuration_id").to_pylist())
    shard_feat = features_of(t)
    report["with_features_schema"] = pq.ParquetFile(shard).schema_arrow.names
    report["with_features_cid_sample"] = shard_cids[:3].tolist()

    rel_cids = np.load(f"{RELEASE}/collection/Alexandria/configuration_id.npy")
    report["release_cid_sample"] = [c.decode() for c in rel_cids[:3]]
    rel_index = {c: i for i, c in enumerate(rel_cids)}
    hits = [(j, rel_index.get(c.encode())) for j, c in enumerate(shard_cids[:20000])]
    matched = [(j, i) for j, i in hits if i is not None]
    report["alexandria_full_cid_match_rate_20k"] = len(matched) / 20000
    if len(matched) < 20000:
        hash_index = {c.split(b":")[-1]: i for i, c in enumerate(rel_cids)}
        hits_h = [(j, hash_index.get(c.encode().split(b":")[-1])) for j, c in enumerate(shard_cids[:20000])]
        matched = [(j, i) for j, i in hits_h if i is not None]
        report["alexandria_hash_match_rate_20k"] = len(matched) / 20000
    n_rel = rel_cids.shape[0]
    rel_feat = np.memmap(f"{RELEASE}/collection/Alexandria/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(n_rel, 512))
    diffs = [float(np.abs(shard_feat[j] - np.asarray(rel_feat[i])).max()) for j, i in matched[:2000]]
    report["alexandria_feature_maxabsdiff_over_2000"] = float(max(diffs)) if diffs else None
    report["alexandria_feature_exact_equal_count"] = int(sum(d == 0.0 for d in diffs))
    print("alexandria", {k: v for k, v in report.items() if k.startswith("alexandria")}, flush=True)

    ta = pq.read_table(ANCHOR, columns=["configuration_id"])
    anchor_cids = np.asarray(ta.column("configuration_id").to_pylist())
    report["anchor_rows"] = int(len(anchor_cids))
    report["anchor_cid_sample"] = anchor_cids[:3].tolist()
    report["anchor_schema"] = pq.ParquetFile(ANCHOR).schema_arrow.names
    tier_cids = np.load(f"{RELEASE}/mad-next-coreset/configuration_id.npy")
    report["mad_next_tier_rows"] = int(tier_cids.shape[0])
    report["mad_next_tier_cid_sample"] = [c.decode() for c in tier_cids[:3]]
    tier_set = set(tier_cids.tolist())
    report["anchor_cids_in_tier_full"] = int(sum(c.encode() in tier_set for c in anchor_cids))
    tier_hash = {c.split(b":")[-1] for c in tier_cids}
    report["anchor_cids_in_tier_hash"] = int(sum(c.encode().split(b":")[-1] in tier_hash for c in anchor_cids))
    print("anchor", {k: v for k, v in report.items() if "anchor" in k or "tier" in k}, flush=True)

    acc = pq.read_table(ACCRETE, columns=["fps_order", "global_id", "configuration_id"])
    fo = acc.column("fps_order").to_numpy(); gid = acc.column("global_id").to_numpy()
    acc_cids = np.asarray(acc.column("configuration_id").to_pylist())
    is_anchor = gid < len(anchor_cids)
    report["accrete_rows"] = int(len(fo)); report["accrete_anchor_members"] = int(is_anchor.sum())
    report["accrete_fps_order_is_row"] = bool(np.array_equal(np.sort(fo), np.arange(len(fo))))
    core_cids = np.load(f"{RELEASE}/mad-core/configuration_id.npy", mmap_mode="r")
    sel = np.where(is_anchor)[0][:3000]
    same_cid = int(sum(core_cids[fo[i]] == acc_cids[i].encode() for i in sel))
    report["accrete_anchor_cid_equals_release_row_cid_3000"] = same_cid
    anchor_feat_tbl = pq.read_table(ANCHOR, columns=["phace_features_bb_mean"])
    anchor_feat = features_of(anchor_feat_tbl)
    core_feat = np.memmap(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(16777216, 512))
    d = [float(np.abs(anchor_feat[gid[i]] - np.asarray(core_feat[fo[i]])).max()) for i in sel]
    report["anchor_feature_maxabsdiff_over_3000"] = float(max(d)); report["anchor_feature_exact_equal_count"] = int(sum(x == 0.0 for x in d))
    print("accrete", {k: v for k, v in report.items() if k.startswith("accrete") or k.startswith("anchor_feature")}, flush=True)

    json.dump(report, open(f"{OUT}/pool_inputs_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
