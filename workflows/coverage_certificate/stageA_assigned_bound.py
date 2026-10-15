import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

RELEASE = "${STORE_ROOT}/mad-core/release"
FPS_ROOT = "${STORE_ROOT}/mad-core/fps/phace_atomic_ep92"
SMALL_ROOT = "${STORE_ROOT}/mad-core/accrete_pools/madcore_prod/small"
ANCHOR = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
POP = f"{HERE}/out/population"
OUT = f"{HERE}/out/stageA"
N_CORE = 16777216
LEVELS = 25
TOP = 4096
CHUNK = 262144
SUB = 32768

centers_release = None
centers_selection = None
scale = None


def load_scale():
    with open(SCALER, "rb") as f:
        return float(pickle.load(f)["scale_"])


def folder_rows(folder):
    return int(np.load(f"{folder}/ns.npy")[0]) if "collection" in folder else np.load(f"{folder}/configuration_id.npy", mmap_mode="r").shape[0]


def features_of(table):
    col = table.column("phace_features_bb_mean").combine_chunks()
    return col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


class TopKeeper:
    def __init__(self):
        self.vals = [np.empty(0) for _ in range(LEVELS)]
        self.rows = [np.empty(0, dtype=np.int64) for _ in range(LEVELS)]

    def push(self, level, vals, rows):
        v = np.concatenate([self.vals[level], vals])
        r = np.concatenate([self.rows[level], rows])
        if v.size > TOP:
            keep = np.argpartition(-v, TOP - 1)[:TOP]
            v, r = v[keep], r[keep]
        order = np.argsort(-v)
        self.vals[level], self.rows[level] = v[order], r[order]


def distances(x, centers, assigned):
    out = np.empty((x.shape[0], LEVELS), dtype=np.float64)
    for s in range(0, x.shape[0], SUB):
        xs = x[s:s + SUB].astype(np.float64)
        for level in range(LEVELS):
            c = centers[assigned[s:s + SUB, level]].astype(np.float64)
            out[s:s + SUB, level] = np.sqrt(np.einsum("ij,ij->i", xs - c, xs - c)) / scale
    return out


def process_rows(x_rel, x_sel, rows, assigned, keep_rel, keep_sel, identity):
    d_rel = distances(x_rel, centers_release, assigned)
    d_sel = distances(x_sel, centers_selection, assigned)
    for level in range(LEVELS):
        keep_rel.push(level, d_rel[:, level], rows)
        keep_sel.push(level, d_sel[:, level], rows)
    row_diff = np.abs(x_rel.astype(np.float64) - x_sel.astype(np.float64)).max(axis=1)
    identity["rows"] += int(rows.size)
    identity["rows_bitwise_equal"] += int((row_diff == 0).sum())
    identity["max_abs_diff"] = max(identity["max_abs_diff"], float(row_diff.max()) if row_diff.size else 0.0)
    shift = np.linalg.norm((x_rel.astype(np.float64) - x_sel.astype(np.float64)) / scale, axis=1)
    identity["max_scaled_shift"] = max(identity["max_scaled_shift"], float(shift.max()) if shift.size else 0.0)


def float32_check(folder, rows, n):
    vd = np.memmap(f"{folder}/voronoi_d.bin", dtype=np.float32, mode="r", shape=(n, LEVELS))
    all_max = np.full(LEVELS, -1.0); all_arg = np.full(LEVELS, -1, dtype=np.int64)
    pop_max = np.full(LEVELS, -1.0); pop_arg = np.full(LEVELS, -1, dtype=np.int64)
    in_pop = np.zeros(n, dtype=bool); in_pop[rows] = True
    for s in range(0, n, 1 << 22):
        block = np.asarray(vd[s:s + (1 << 22)])
        am = block.argmax(axis=0)
        better = block[am, np.arange(LEVELS)] > all_max
        all_max[better] = block[am, np.arange(LEVELS)][better]; all_arg[better] = (s + am)[better]
        mask = in_pop[s:s + block.shape[0]]
        if mask.any():
            sub = block[mask]; idx = np.where(mask)[0]
            am = sub.argmax(axis=0)
            better = sub[am, np.arange(LEVELS)] > pop_max
            pop_max[better] = sub[am, np.arange(LEVELS)][better]; pop_arg[better] = (s + idx[am])[better]
    return all_max, all_arg, pop_max, pop_arg


def run_folder(task):
    kind, name = task
    started = time.time()
    folder = f"{RELEASE}/collection/{name}" if kind != "anchor" else f"{RELEASE}/mad-next-coreset"
    n = folder_rows(folder)
    va = np.memmap(f"{folder}/voronoi_a.bin", dtype=np.int32, mode="r", shape=(n, LEVELS))
    feat = np.memmap(f"{folder}/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(n, 512))
    keep_rel, keep_sel = TopKeeper(), TopKeeper()
    identity = {"rows": 0, "rows_bitwise_equal": 0, "max_abs_diff": 0.0, "max_scaled_shift": 0.0}

    if kind in ("big", "small"):
        first_row = np.load(f"{POP}/pool_row_to_release__{name}.npy")
        root = f"{FPS_ROOT}/{name}/with_features" if kind == "big" else f"{SMALL_ROOT}/{name}"
        shards = sorted(glob.glob(f"{root}/*.parquet"))
        x_sel_all = np.concatenate([features_of(pq.read_table(p, columns=["phace_features_bb_mean"])) for p in shards])
        valid = first_row >= 0
        order = np.argsort(first_row[valid], kind="stable")
        rows_sorted = first_row[valid][order]
        x_sel_sorted = x_sel_all[valid][order]
        del x_sel_all
        rows_pop = np.load(f"{POP}/collection__{name}.npy")
        stream = rows_sorted.size > 0.05 * n
        for s in range(0, rows_sorted.size, CHUNK):
            rows = rows_sorted[s:s + CHUNK]
            if stream:
                lo, hi = int(rows[0]), int(rows[-1]) + 1
                x_rel = np.asarray(feat[lo:hi])[rows - lo]
                assigned = np.asarray(va[lo:hi])[rows - lo].astype(np.int64)
            else:
                x_rel = np.asarray(feat[rows])
                assigned = np.asarray(va[rows]).astype(np.int64)
            process_rows(x_rel, x_sel_sorted[s:s + CHUNK], rows, assigned, keep_rel, keep_sel, identity)
    else:
        rows_pop = np.load(f"{POP}/mad-next-coreset.npy")
        for s in range(0, rows_pop.size, CHUNK):
            rows = rows_pop[s:s + CHUNK]
            x_rel = np.asarray(feat[rows])
            x_sel = x_rel
            assigned = np.asarray(va[rows]).astype(np.int64)
            process_rows(x_rel, x_sel, rows, assigned, keep_rel, keep_sel, identity)

    all_max, all_arg, pop_max, pop_arg = float32_check(folder, rows_pop, n)
    tag = f"{kind}__{name}"
    np.savez(f"{OUT}/{tag}.npz",
             top_vals_rel=np.stack(keep_rel.vals), top_rows_rel=np.stack(keep_rel.rows),
             top_vals_sel=np.stack(keep_sel.vals), top_rows_sel=np.stack(keep_sel.rows),
             f32_all_max=all_max, f32_all_argmax=all_arg, f32_pop_max=pop_max, f32_pop_argmax=pop_arg,
             n_rows=n, n_population=rows_pop.size,
             identity=np.asarray([identity["rows"], identity["rows_bitwise_equal"], identity["max_abs_diff"], identity["max_scaled_shift"]]))
    print(f"{tag}: n={n:,} pop={rows_pop.size:,} lvl24 UB_rel={keep_rel.vals[24][0]:.6f} UB_sel={keep_sel.vals[24][0]:.6f} "
          f"f32_pop={pop_max[24]:.6f} f32_all={all_max[24]:.4f} feat_equal={identity['rows_bitwise_equal']}/{identity['rows']} "
          f"maxdiff={identity['max_abs_diff']:.2e} shift={identity['max_scaled_shift']:.2e} ({time.time() - started:.0f}s)", flush=True)
    return tag


def main():
    global centers_release, centers_selection, scale
    os.makedirs(OUT, exist_ok=True)
    scale = load_scale()
    if os.environ.get("COV_MMAP") == "1":
        centers_release = np.memmap(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32, mode="r", shape=(N_CORE, 512))
        centers_selection = np.load(f"{HERE}/out/selection_features_madcore.npy", mmap_mode="r")
    else:
        centers_release = np.fromfile(f"{RELEASE}/mad-core/ecumetric_features.bin", dtype=np.float32).reshape(N_CORE, 512)
        centers_selection = np.load(f"{HERE}/out/selection_features_madcore.npy")
    assert centers_selection.shape == (N_CORE, 512)
    print("centers loaded", flush=True)

    fams = json.load(open(f"{HERE}/out/pool_families.json"))
    tasks = [("anchor", "mad-next-coreset")] + [("big", f) for f in fams["big_families"]] + [("small", f) for f in fams["small_families"]]
    done = {os.path.basename(p)[:-4] for p in glob.glob(f"{OUT}/*.npz")}
    tasks = [t for t in tasks if f"{t[0]}__{t[1]}" not in done]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    if len(sys.argv) > 2:
        only = set(sys.argv[2].split(","))
        tasks = [t for t in tasks if t[1] in only]

    with Pool(workers) as pool:
        for _ in pool.imap_unordered(run_folder, tasks):
            pass
    print("all folders done", flush=True)


if __name__ == "__main__":
    main()
