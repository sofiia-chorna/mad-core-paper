import glob
import os
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
assign_dir = f"{voronoi_dir}/out/assign"
valid_dir = f"{voronoi_dir}/out/valid"
ahmetcik_path = "${DEV_ROOT}/phace-atomic/accrete/runs/no_ahm/ahmetcik_entries.txt"
NPROC = int(os.environ.get("VALID_NPROC", "128"))

with open(ahmetcik_path) as fh:
    ahmetcik = set(line.strip() for line in fh if line.strip())


def family_of(path):
    return path.split("/assign/")[1].split("/")[0]


def sidecar_path(path):
    relative = path.split("/assign/")[1]
    return f"{valid_dir}/{relative}"


def check_file(path):
    family = family_of(path)
    parquet_file = pq.ParquetFile(path)
    n = parquet_file.metadata.num_rows
    if n == 0:
        os.makedirs(os.path.dirname(sidecar_path(path)), exist_ok=True)
        pq.write_table(pa.table({"is_valid": pa.array([], pa.bool_())}),
                       sidecar_path(path), compression="zstd")
        return path, 0, 0, 0, []

    d24 = pq.read_table(path, columns=["d24"]).column("d24").to_numpy()
    bad_feature = np.isnan(d24)

    is_ahmetcik = np.zeros(n, dtype=bool)
    if family == "nomad":
        cids = pq.read_table(path, columns=["configuration_id"]).column("configuration_id").to_pylist()
        is_ahmetcik = np.fromiter((c.split(":", 1)[-1] in ahmetcik for c in cids), bool, n)

    is_valid = ~bad_feature & ~is_ahmetcik

    out_path = sidecar_path(path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    pq.write_table(pa.table({"is_valid": pa.array(is_valid, pa.bool_())}), tmp_path,
                   compression="zstd")
    os.rename(tmp_path, out_path)

    records = []
    invalid_rows = np.where(~is_valid)[0]
    if invalid_rows.size:
        cids = pq.read_table(path, columns=["configuration_id"]).column("configuration_id").to_pylist()
        for r in invalid_rows:
            reason = "nan_features" if bad_feature[r] else "ahmetcik_nomad"
            records.append((family, os.path.basename(path), int(r), cids[int(r)], reason))

    return path, n, int(bad_feature.sum()), int(is_ahmetcik.sum()), records


def main():
    files = sorted(glob.glob(f"{assign_dir}/*/*.parquet"))
    print(f"scanning {len(files):,} assign files with {NPROC} workers", flush=True)

    total_rows = total_nan = total_ahmetcik = 0
    all_records = []
    with Pool(NPROC) as pool:
        for index, (path, n, n_nan, n_ahm, records) in enumerate(
                pool.imap_unordered(check_file, files, chunksize=4)):
            total_rows += n
            total_nan += n_nan
            total_ahmetcik += n_ahm
            all_records.extend(records)
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1:,}/{len(files):,} files, {total_rows:,} rows, "
                      f"nan={total_nan:,} ahmetcik={total_ahmetcik:,}", flush=True)

    invalid = total_nan + total_ahmetcik
    index_table = pa.table({
        "family": pa.array([r[0] for r in all_records], pa.string()),
        "file": pa.array([r[1] for r in all_records], pa.string()),
        "row": pa.array([r[2] for r in all_records], pa.int64()),
        "configuration_id": pa.array([r[3] for r in all_records], pa.string()),
        "reason": pa.array([r[4] for r in all_records], pa.string()),
    })
    pq.write_table(index_table, f"{valid_dir}/_invalid_index.parquet", compression="zstd")

    sidecar_bytes = sum(os.path.getsize(p) for p in glob.glob(f"{valid_dir}/*/*.parquet"))
    print(f"\nrows: {total_rows:,}")
    print(f"nan_features:   {total_nan:,}")
    print(f"ahmetcik_nomad: {total_ahmetcik:,}")
    print(f"invalid total:  {invalid:,}  ({100 * invalid / max(total_rows, 1):.4f}%)")
    print(f"valid total:    {total_rows - invalid:,}")
    print(f"sidecar size:   {sidecar_bytes / 1e6:.1f} MB")
    print(f"invalid index -> {valid_dir}/_invalid_index.parquet ({index_table.num_rows:,} rows)", flush=True)


if __name__ == "__main__":
    main()
