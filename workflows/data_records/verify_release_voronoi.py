import glob

import numpy as np
import pyarrow.parquet as pq

center = 16531443
center_cid = "gdb13:C#CC#CC1NCC2CNCC1N2"
group = "${STORE_ROOT}/mad-core/release/collection/gdb13"
proj = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap/results/cell_115m/proj"
store_feats_scaler = np.load("${DEV_ROOT}/phace-atomic/release/build/ecumetric_scaler.npz")

cids = np.load(f"{group}/configuration_id.npy", mmap_mode="r")
ns = len(cids)
voronoi_a = np.memmap(f"{group}/voronoi_a.bin", dtype=np.int32, mode="r").reshape(ns, 25)
voronoi_d = np.memmap(f"{group}/voronoi_d.bin", dtype=np.float32, mode="r").reshape(ns, 25)

member_mask = voronoi_a[:, 24] == center
n_release = int(member_mask.sum())
print(f"release gdb13 group ns={ns:,}")
print(f"release gdb13 members (voronoi_a[:,24]==center): {n_release:,}")

# assign-side truth: gdb13 members from the extraction outputs
n_assign = sum(pq.ParquetFile(p).metadata.num_rows for p in glob.glob(f"{proj}/gdb13__*.parquet"))
print(f"assign  gdb13 members (extraction outputs):      {n_assign:,}")
print(f"COUNTS MATCH: {n_release == n_assign}")

# set equality of member configuration_ids
release_member_cids = set(np.asarray(cids[member_mask]).astype("U"))
assign_member_cids = set()
for p in glob.glob(f"{proj}/gdb13__*.parquet"):
    assign_member_cids.update(pq.read_table(p, columns=["configuration_id"]).column(0).to_pylist())
print(f"member cid sets equal: {release_member_cids == assign_member_cids}  "
      f"(release only {len(release_member_cids - assign_member_cids)}, "
      f"assign only {len(assign_member_cids - release_member_cids)})")

# the center itself: self-assigned at level 24, zero distance
center_index = int(np.where(np.asarray(cids).astype('U') == center_cid)[0][0])
print(f"center row {center_index}: voronoi_a[24]={voronoi_a[center_index,24]} (==center {voronoi_a[center_index,24]==center}), "
      f"voronoi_d[24]={voronoi_d[center_index,24]:.2e}")
