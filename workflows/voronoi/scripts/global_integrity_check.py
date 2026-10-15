import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
centers_arrow_path = f"{voronoi_dir}/out/tree/center_configuration_ids.arrow"
level_bound = np.array([2 ** k for k in range(25)], dtype=np.int64)
NOISE_TOLERANCE = 2e-2

if not os.path.exists(centers_arrow_path):
    cids = np.load(f"{voronoi_dir}/out/tree/center_configuration_ids.npy", allow_pickle=True)
    table = pa.table({"configuration_id": pa.array(cids.tolist(), pa.string())})
    with pa.OSFile(centers_arrow_path, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    print("centers arrow file written", flush=True)

center_column = None


def init_worker():
    global center_column
    source = pa.memory_map(centers_arrow_path, "r")
    center_column = pa.ipc.open_file(source).read_all().column("configuration_id").combine_chunks()


def check_file(path):
    columns = (["configuration_id"] + [f"a{k:02d}" for k in range(25)]
               + [f"d{k:02d}" for k in range(25)] + ["c00", "c12", "c24"])
    table = pq.read_table(path, columns=columns)
    n = table.num_rows
    if n == 0:
        return path, n, {}
    assigns = np.stack([table.column(f"a{k:02d}").to_numpy() for k in range(25)], axis=1)
    dists = np.stack([table.column(f"d{k:02d}").to_numpy() for k in range(25)], axis=1)
    problems = {}

    non_monotone = int((np.diff(dists, axis=1) > 1e-4).any(1).sum())
    if non_monotone:
        problems["non_monotone_rows"] = non_monotone
    out_of_bound = int((assigns >= level_bound[None, :]).any(1).sum())
    if out_of_bound:
        problems["assign_out_of_level_bound"] = out_of_bound

    for k in (0, 12, 24):
        expected = center_column.take(pa.array(assigns[:, k]))
        mismatch = int(pc.sum(pc.not_equal(table.column(f"c{k:02d}").combine_chunks(), expected)).as_py() or 0)
        if mismatch:
            problems[f"c_vs_a_mismatch_l{k}"] = mismatch

    cid_column = table.column("configuration_id").combine_chunks()
    is_center = pc.is_in(cid_column, value_set=center_column).to_numpy(zero_copy_only=False)
    if is_center.any():
        self_ok = pc.equal(cid_column, table.column("c24").combine_chunks()).to_numpy(zero_copy_only=False)
        wrong = is_center & ~self_ok
        noisy = is_center & self_ok & (dists[:, 24] > NOISE_TOLERANCE)
        if int(wrong.sum()):
            problems["center_assigned_elsewhere"] = int(wrong.sum())
        if int(noisy.sum()):
            problems["self_distance_noise_gt_2e-2"] = int(noisy.sum())
    return path, n, problems


files = sorted(glob.glob(f"{voronoi_dir}/out/assign/*/*.parquet"))
print(f"checking {len(files):,} files", flush=True)
total_rows = 0
total_problems = {}
n_bad_files = 0
with Pool(64, initializer=init_worker) as pool:
    for index, (path, n, problems) in enumerate(pool.imap_unordered(check_file, files, chunksize=2)):
        total_rows += n
        if problems:
            n_bad_files += 1
            print(f"PROBLEMS {path}: {problems}", flush=True)
            for key, value in problems.items():
                total_problems[key] = total_problems.get(key, 0) + value
        if (index + 1) % 500 == 0:
            print(f"  {index + 1:,}/{len(files):,} files, {total_rows:,} rows, bad: {n_bad_files}", flush=True)

print(f"\nfiles: {len(files):,}  rows: {total_rows:,}", flush=True)
print(f"aggregate problems: {json.dumps(total_problems)}")
real = {key: value for key, value in total_problems.items() if key != "self_distance_noise_gt_2e-2"}
print("GLOBAL INTEGRITY:", "ALL PASS" if not real else "*** REAL ISSUES ***", flush=True)
