from __future__ import annotations
import os
import sys
import glob
import pyarrow as pa
import pyarrow.parquet as pq


def rewrite_file(path: str) -> int:
    t = pq.read_table(path)
    names = t.column_names
    if "configuration_id" not in names or "dataset_id" not in names:
        return 0
    cid = t.column("configuration_id").to_pylist()
    dsid = t.column("dataset_id").to_pylist()
    n_changed = 0
    new = cid
    for i, c in enumerate(cid):
        if c is not None and c.startswith("NC_"):
            new[i] = f"{dsid[i]}:{c[3:]}"
            n_changed += 1
    if n_changed == 0:
        return 0
    idx = names.index("configuration_id")
    t = t.set_column(idx, "configuration_id", pa.array(new, type=pa.string()))
    tmp = path + ".tmp"
    pq.write_table(t, tmp, compression="snappy")
    os.replace(tmp, path)
    return n_changed


def main():
    targets = sys.argv[1:]
    files = []
    for tgt in targets:
        if os.path.isdir(tgt):
            files += glob.glob(os.path.join(tgt, "**", "*.parquet"), recursive=True)
        else:
            files.append(tgt)
    files = sorted(set(files))
    tot_files = tot_rows = 0
    for f in files:
        c = rewrite_file(f)
        if c:
            tot_files += 1
            tot_rows += c
            print(f"  rewrote {c} ids in {f}")
    print(f"[rewrite_nc_ids] changed {tot_rows} ids across {tot_files}/{len(files)} files")


if __name__ == "__main__":
    main()
