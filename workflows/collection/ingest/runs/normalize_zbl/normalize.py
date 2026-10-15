
import argparse
import ast
import glob
import os
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "${DEV_ROOT}/mad-core/fps")
import zbl_energy as Z

ZBL = "${STORE_ROOT}/zbl_filtering/output"
SIDECARS = "${DEV_ROOT}/mad-core/accrete/runs/zbl_cols_full/cols"


def norm_atomic_numbers(col):
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        decoded = [ast.literal_eval(x) if x is not None else None for x in col.to_pylist()]
        return pa.array(decoded, type=pa.list_(pa.int64()))
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
        return col.cast(pa.list_(pa.int64()))
    return col


def normalize_shard(src, sidecar, apply):
    t = pq.read_table(src)
    names = t.column_names
    n = t.num_rows
    if "zbl_max_force" in names and "zbl_energy_per_atom" in names:
        return "skip", n, names

    out = {}  # keep columns chunked (no combine_chunks -> no >2GB string overflow)
    for nm in names:
        if nm == "max_force":
            out["zbl_max_force"] = t["max_force"]
        elif nm == "atomic_numbers":
            out["atomic_numbers"] = norm_atomic_numbers(t["atomic_numbers"])
        else:
            out[nm] = t[nm]

    if "zbl_energy_per_atom" in names:
        branch = "in-shard"
    elif sidecar and os.path.exists(sidecar):
        sc = pq.read_table(sidecar, columns=["configuration_id", "zbl_energy_per_atom", "zbl_max_atomic_energy"])
        if sc.num_rows != n or sc["configuration_id"].to_pylist() != t["configuration_id"].to_pylist():
            return "CID_MISMATCH", n, list(out)
        out["zbl_energy_per_atom"] = sc["zbl_energy_per_atom"]
        out["zbl_max_atomic_energy"] = sc["zbl_max_atomic_energy"]
        branch = "sidecar"
    else:
        an, pos = t["atomic_numbers"].to_pylist(), t["positions"].to_pylist()
        cel, pb = t["cell"].to_pylist(), t["pbc"].to_pylist()
        epa = np.empty(n); mae = np.empty(n)
        for k in range(n):
            epa[k], mae[k] = Z.zbl_energy(Z._decode(an[k]), Z._decode(pos[k]), Z._decode(cel[k]), Z._decode(pb[k]))
        out["zbl_energy_per_atom"] = pa.array(epa, pa.float64())
        out["zbl_max_atomic_energy"] = pa.array(mae, pa.float64())
        branch = "compute"

    if apply:
        tmp = f"{src}.norm.{os.getpid()}.tmp"
        pq.write_table(pa.table(out), tmp, compression="zstd")
        os.replace(tmp, src)
    return branch, n, list(out)


def _worker(task):
    s, sidecar, apply = task
    try:
        br, _, out_names = normalize_shard(s, sidecar, apply)
        return (br, s, out_names)
    except Exception as e:
        return ("ERROR", s, repr(e))


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N shards (validation)")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    shards = sorted(p for p in glob.glob(f"{ZBL}/{args.family}/**/*.parquet", recursive=True) if ".tmp" not in p)
    if args.limit:
        shards = shards[:args.limit]
    has_sidecar = os.path.isdir(f"{SIDECARS}/{args.family}")
    tasks = [(s, f"{SIDECARS}/{args.family}/{i:05d}.parquet" if has_sidecar else None, args.apply)
             for i, s in enumerate(shards)]

    with Pool(min(max(len(tasks), 1), args.workers)) as pool:
        results = pool.map(_worker, tasks)

    branches = Counter(r[0] for r in results)
    errors = [(r[1], r[2]) for r in results if r[0] in ("ERROR", "CID_MISMATCH")]
    sample = next((r[2] for r in results if r[0] not in ("skip", "ERROR", "CID_MISMATCH")), None)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] {args.family:42s} shards={len(shards):>4} sidecar={has_sidecar} {dict(branches)} errors={len(errors)}", flush=True)
    if sample and not args.apply:
        print(f"    -> result cols: {sample}")
    for s, e in errors[:3]:
        print(f"    !! {s}: {e}", flush=True)


if __name__ == "__main__":
    main()
