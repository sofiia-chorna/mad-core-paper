import argparse
import glob
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from run_cell_fps_sharded import CORESET, OUT_SCHEMA, WRITE_CHUNK, widen_strings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--parts-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()

    cell_name = f"c{args.cell:08d}"
    prefix = f"{args.checkpoint_root}/{cell_name}"
    with open(f"{prefix}_latest.txt") as fh:
        step = int(fh.read().strip())

    selected_np = np.load(f"{prefix}_k{step:010d}_ids.npy")
    eps_np = np.load(f"{prefix}_k{step:010d}_eps.npy").astype(np.float32)
    eps_np[0] = 0.0

    paths = sorted(glob.glob(f"{args.parts_root}/{cell_name}/part_*.parquet"))
    ids = pa.concat_tables(
        pq.read_table(p, columns=["cell", "d24", "configuration_id",
                                  "dataset_group", "dataset_id"])
        for p in paths)
    n_global = len(ids)

    if len(selected_np) != n_global:
        raise RuntimeError(f"{cell_name}: checkpoint holds {len(selected_np):,} picks "
                           f"but the pool has {n_global:,} rows — FPS is not finished")

    center_cid = str(np.load(f"{CORESET}/configuration_id.npy",
                             mmap_mode="r")[args.cell].astype("U"))
    cid_first = ids.column("configuration_id")[int(selected_np[0])].as_py()

    d24 = ids.column("d24").to_numpy()
    radius_sq = float(np.max(d24.astype(np.float64) ** 2))
    problems = {
        "all_rows_this_cell": bool((ids.column("cell").to_numpy() == args.cell).all()),
        "seed_is_center": bool(cid_first == center_cid),
        "selected_is_permutation": bool(len(np.unique(selected_np)) == n_global),
        "eps_increase_steps": int(np.sum(
            np.diff(eps_np[1:]) > 1e-5 * np.abs(eps_np[1:-1]) + 1e-7)),
        "radius_rel_diff": abs(float(eps_np[1]) - radius_sq) / radius_sq,
    }
    print(f"{cell_name}: n={n_global:,} problems {problems}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/cell_{cell_name}.parquet"
    wide = widen_strings(ids)
    writer = pq.ParquetWriter(out_path, OUT_SCHEMA)
    for start in range(0, n_global, WRITE_CHUNK):
        stop = min(start + WRITE_CHUNK, n_global)
        ordered = wide.take(pa.array(selected_np[start:stop]))
        writer.write_table(pa.table({
            "cell": pa.array(np.full(stop - start, args.cell, dtype=np.int32)),
            "cell_rank": pa.array(np.arange(start, stop, dtype=np.int32)),
            "eps_sq": pa.array(eps_np[start:stop]),
            "configuration_id": ordered.column("configuration_id").cast(pa.string()),
            "dataset_group": ordered.column("dataset_group").cast(pa.string()),
            "dataset_id": ordered.column("dataset_id").cast(pa.string()),
        }, schema=OUT_SCHEMA))
        print(f"  wrote {stop:,}/{n_global:,}", flush=True)
    writer.close()

    summary = {"cell": args.cell, "n_rows": int(n_global), "seed": int(selected_np[0]),
               "written_from": "checkpoint", "problems": problems}
    with open(f"{args.out_dir}/cell_{cell_name}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(f"{args.out_dir}/cell_{cell_name}.done", "w") as fh:
        fh.write("done\n")
    print(f"{cell_name}: DONE wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
