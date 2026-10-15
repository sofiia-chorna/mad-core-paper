import argparse
import collections
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

analysis = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"
tree = "${DEV_ROOT}/phace-atomic/voronoi/out/tree"
N = 2 ** 20

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--d", type=int, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

out = f"{run_dir}/{args.out}"
os.makedirs(out, exist_ok=True)

embedding = np.load(f"{run_dir}/results/embeddings/embedding_l20_d{args.d}_{args.tag}.npy")
family = np.asarray([str(f) for f in
                     np.load(f"{analysis}/family_level20.npy", allow_pickle=True)])
dataset_id = np.asarray([str(x) for x in
                         np.load(f"{analysis}/dataset_id_level20.npy", allow_pickle=True)])
counts = np.load(f"{analysis}/populations_level20.npy").astype(np.int64)

cid = np.load(f"{tree}/center_configuration_ids.npy", allow_pickle=True)[:N]
cid = np.asarray([str(c) for c in cid])

assert embedding.shape == (N, args.d) and family.shape == (N,) and cid.shape == (N,)
assert counts.sum() == 1_695_275_276

cols = {"coreset_index": pa.array(np.arange(N, dtype=np.int64)),
        "configuration_id": pa.array(cid.tolist(), pa.string())}
for i in range(args.d):
    cols[f"x{i}"] = pa.array(embedding[:, i].astype(np.float32))
cols["family"] = pa.array(family.tolist(), pa.string())
cols["dataset_id"] = pa.array(dataset_id.tolist(), pa.string())
cols["voronoi_population"] = pa.array(counts)
cols["log10_population"] = pa.array(np.log10(counts).astype(np.float32))

table = pa.table(cols)
pq.write_table(table, f"{out}/sketchmap_2p20_d{args.d}.parquet", compression="zstd")

for f in (f"results/metrics/tiled_{args.tag}.json", f"results/metrics/score_{args.tag}.json"):
    if os.path.exists(f"{run_dir}/{f}"):
        shutil.copy(f"{run_dir}/{f}", f"{out}/{os.path.basename(f)}")

top = collections.Counter(family.tolist()).most_common()
summary = {"n_points": N, "n_components": args.d, "tag": args.tag,
           "library_structures_represented": int(counts.sum()),
           "n_families": len(top),
           "families_top20": [{"family": f, "n_centers": c,
                               "pct_of_centers": round(100 * c / N, 3)} for f, c in top[:20]]}
with open(f"{out}/families.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"bundle -> {out}")
print(f"  parquet rows={table.num_rows:,} cols={len(table.schema.names)}")
print(f"  size: {os.path.getsize(f'{out}/sketchmap_2p20_d{args.d}.parquet')/1e6:.1f} MB")
print(f"  columns: {table.schema.names}")
print(f"  unique configuration_id: {len(set(cid.tolist())):,} of {N:,}")
for f, c in top[:8]:
    print(f"     {c:>8,}  {100*c/N:5.2f}%  {f}")
