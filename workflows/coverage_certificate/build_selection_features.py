import json

import numpy as np
import pyarrow.parquet as pq

RELEASE = "${STORE_ROOT}/mad-core/release"
ANCHOR = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
ACCRETE = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/madcore_856k/out/accrete.parquet"
OUT = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert/out"
N_CORE = 16777216


def main():
    ta = pq.read_table(ANCHOR, columns=["configuration_id", "phace_features_bb_mean"])
    anchor_cids = ta.column("configuration_id").to_pylist()
    col = ta.column("phace_features_bb_mean").combine_chunks()
    anchor_feat = col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)
    anchor_index = {c.encode(): i for i, c in enumerate(anchor_cids)}

    core_cids = np.load(f"{RELEASE}/mad-core/configuration_id.npy")
    anchor_row_of_core = np.full(N_CORE, -1, dtype=np.int64)
    for r in range(N_CORE):
        i = anchor_index.get(core_cids[r])
        if i is not None:
            anchor_row_of_core[r] = i

    acc = pq.read_table(ACCRETE, columns=["fps_order", "global_id"])
    fo = acc.column("fps_order").to_numpy(); gid = acc.column("global_id").to_numpy()
    n_anchor = len(anchor_cids)
    sel = gid < n_anchor
    agree = int((anchor_row_of_core[fo[sel]] == gid[sel]).sum())
    print(f"cid-derived anchor rows agree with accrete global_id on {agree}/{int(sel.sum())} anchor members", flush=True)

    core_feat = np.memmap(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(N_CORE, 512))
    rows = np.where(anchor_row_of_core >= 0)[0]
    print(f"anchor rows in mad-core: {rows.size:,}", flush=True)
    swapped = np.lib.format.open_memmap(f"{OUT}/selection_features_madcore.npy", mode="w+", dtype=np.float32, shape=(N_CORE, 512))
    for s in range(0, N_CORE, 1 << 20):
        swapped[s:s + (1 << 20)] = core_feat[s:s + (1 << 20)]
    swapped[rows] = anchor_feat[anchor_row_of_core[rows]]
    swapped.flush()
    np.save(f"{OUT}/anchor_row_of_core.npy", anchor_row_of_core)

    diff = np.abs(anchor_feat[anchor_row_of_core[rows]].astype(np.float64) - np.asarray(core_feat[rows]).astype(np.float64))
    json.dump({"anchor_rows_in_core": int(rows.size), "accrete_agree": agree, "max_abs_diff": float(diff.max()),
               "rows_bitwise_equal": int((diff.max(axis=1) == 0).sum())},
              open(f"{OUT}/selection_features_build.json", "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
