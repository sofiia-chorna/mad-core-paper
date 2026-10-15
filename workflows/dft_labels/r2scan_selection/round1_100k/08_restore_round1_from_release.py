import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import write

release_dir = "${STORE_ROOT}/mad-core/release/mad-core/"
mad_tier_dir = "${STORE_ROOT}/mad-core/release/mad-next-coreset/"
round1_index_path = ("${DEV_ROOT}/phace-atomic/accrete/runs/"
                     "madcore_10M/recalc_100k/first100k_nonMAD_lt300atoms.parquet")
output_dir = "${DEV_ROOT}/phace-atomic/botr/round1_100k/out"
output_stem = f"{output_dir}/cscs-madcore_first100k_non-mad_max300"

WALK_LIMIT = 1_000_000
MAX_ATOMS = 300
N_TARGET = 100_000

os.makedirs(output_dir, exist_ok=True)

na = np.load(release_dir + "na.npy", mmap_mode="r")
n_atoms_window = np.diff(na[: WALK_LIMIT + 1]).astype(np.int64)

dataset_group_window = np.load(release_dir + "dataset_group.npy", mmap_mode="r")[:WALK_LIMIT]
release_cids_window = np.load(release_dir + "configuration_id.npy", mmap_mode="r")[:WALK_LIMIT]
mad_union_cids = np.load(mad_tier_dir + "configuration_id.npy")

is_mad_by_group = dataset_group_window == b"mad-next"
is_mad = np.isin(release_cids_window, mad_union_cids.astype(release_cids_window.dtype))
assert (is_mad == is_mad_by_group).all(), "mad-tier id join disagrees with dataset_group"

qualifying_rows = np.flatnonzero((~is_mad) & (n_atoms_window < MAX_ATOMS))
selected_rows = qualifying_rows[:N_TARGET]

round1_index = pq.read_table(round1_index_path).sort_by("fps_rank")
stored_ranks = round1_index.column("fps_rank").to_numpy()
stored_n_atoms = round1_index.column("n_atoms").to_numpy().astype(np.int64)
stored_old_cids = np.array(round1_index.column("configuration_id").to_pylist())

assert np.array_equal(selected_rows, stored_ranks), "restored ranks differ from the round-1 index"
assert np.array_equal(n_atoms_window[selected_rows], stored_n_atoms), "n_atoms differ from the round-1 index"
print(f"restored ranks match the round-1 index exactly: {selected_rows.min()} .. {selected_rows.max():,}", flush=True)

dataset_id_selected = np.load(release_dir + "dataset_id.npy", mmap_mode="r")[selected_rows]
pbc_selected = np.load(release_dir + "pbc.npy", mmap_mode="r")[selected_rows]
cell_selected = np.memmap(release_dir + "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3)[selected_rows]

atom_start = int(na[selected_rows.min()])
atom_stop = int(na[selected_rows.max() + 1])
numbers_span = np.memmap(release_dir + "a.bin", dtype=np.int32, mode="r")[atom_start:atom_stop]
positions_span = np.memmap(release_dir + "x.bin", dtype=np.float32, mode="r").reshape(-1, 3)[atom_start:atom_stop]

atoms_list = []
for j, row in enumerate(selected_rows):
    lo, hi = int(na[row]) - atom_start, int(na[row + 1]) - atom_start
    assert hi - lo == n_atoms_window[row]

    atoms = Atoms(
        numbers=numbers_span[lo:hi],
        positions=positions_span[lo:hi].astype(np.float64),
        cell=cell_selected[j].astype(np.float64),
        pbc=pbc_selected[j],
    )
    atoms.info.update(
        fps_rank=int(row),
        configuration_id=release_cids_window[row].decode(),
        dataset_id=dataset_id_selected[j].decode(),
        family=dataset_group_window[row].decode(),
        n_atoms=int(n_atoms_window[row]),
        periodicity=int(pbc_selected[j].sum()),
    )
    atoms_list.append(atoms)

assert len(atoms_list) == N_TARGET
assert len({atoms.info["configuration_id"] for atoms in atoms_list}) == N_TARGET, "duplicate cid in selection"

write(f"{output_stem}.xyz", atoms_list, format="extxyz")


def build_column(key, dtype):
    return pa.array([atoms.info[key] for atoms in atoms_list], dtype)


index_table = pa.table({
    "fps_rank": build_column("fps_rank", pa.int64()),
    "configuration_id": build_column("configuration_id", pa.string()),
    "configuration_id_pre_unification": pa.array(stored_old_cids.tolist(), pa.string()),
    "dataset_id": build_column("dataset_id", pa.string()),
    "family": build_column("family", pa.string()),
    "n_atoms": build_column("n_atoms", pa.int32()),
    "periodicity": build_column("periodicity", pa.int8()),
})
pq.write_table(index_table, f"{output_stem}.parquet")

periodicity_counts = Counter(atoms.info["periodicity"] for atoms in atoms_list)
print(f"\nwrote {len(atoms_list):,} structures:")
print(f"  {output_stem}.xyz")
print(f"  {output_stem}.parquet  (carries both id eras)")
print(f"  n_atoms: min={n_atoms_window[selected_rows].min()} max={n_atoms_window[selected_rows].max()} "
      f"mean={n_atoms_window[selected_rows].mean():.1f}")
print(f"  periodicity (0=mol,1/2=surface,3=bulk): {dict(sorted(periodicity_counts.items()))}")
print(f"  top families: {Counter(atoms.info['family'] for atoms in atoms_list).most_common(8)}")

image_flag_rows = np.load(release_dir + "image_flag_rows.npy")
print(f"  folded far-from-origin structures in selection: {np.intersect1d(image_flag_rows, selected_rows).tolist()}")
print("DONE", flush=True)
