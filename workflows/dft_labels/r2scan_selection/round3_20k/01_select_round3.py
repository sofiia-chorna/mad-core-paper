import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import write

release_dir = "${STORE_ROOT}/mad-core/release/mad-core/"
mad_tier_dir = "${STORE_ROOT}/mad-core/release/mad-next-coreset/"
botr_store_dir = "${STORE_ROOT}/mad-core/botr/"
previous_parquet_paths = [
    botr_store_dir + "round1_100k/cscs-madcore_first100k_non-mad_max300.parquet",
    botr_store_dir + "round2_100k/cscs-madcore_next100k_non-mad_max300.parquet",
]
output_dir = "${DEV_ROOT}/phace-atomic/botr/round3_20k/out"
output_stem = f"{output_dir}/cscs-madcore_next20k_non-mad_max300"

WALK_LIMIT = 1_000_000
MAX_ATOMS = 300
N_PREVIOUS = 200_000
LAST_SENT_RANK = 238_371
N_TARGET = 20_000

os.makedirs(output_dir, exist_ok=True)

na = np.load(release_dir + "na.npy", mmap_mode="r")
n_atoms_window = np.diff(na[: WALK_LIMIT + 1]).astype(np.int64)

dataset_group_window = np.load(release_dir + "dataset_group.npy", mmap_mode="r")[:WALK_LIMIT]
release_cids_window = np.load(release_dir + "configuration_id.npy", mmap_mode="r")[:WALK_LIMIT]
mad_union_cids = np.load(mad_tier_dir + "configuration_id.npy")

is_mad_by_group = dataset_group_window == b"mad-next"
is_mad = np.isin(release_cids_window, mad_union_cids.astype(release_cids_window.dtype))
assert (is_mad == is_mad_by_group).all(), "mad-tier id join disagrees with dataset_group"
print(f"MAD rows in first {WALK_LIMIT:,}: {int(is_mad.sum()):,} (id join == dataset_group)", flush=True)

qualifying_mask = (~is_mad) & (n_atoms_window < MAX_ATOMS)
qualifying_rows = np.flatnonzero(qualifying_mask)
assert len(qualifying_rows) >= N_PREVIOUS + N_TARGET, f"only {len(qualifying_rows):,} qualify in the window"

previous_stored_ranks = np.sort(np.concatenate([
    pq.read_table(path, columns=["fps_rank"]).column(0).to_numpy() for path in previous_parquet_paths]))
assert previous_stored_ranks.max() == LAST_SENT_RANK
assert np.array_equal(qualifying_rows[:N_PREVIOUS], previous_stored_ranks), \
    "release walk does not reproduce the round-1 + round-2 selection"
print(f"rounds 1+2 reproduced exactly: ranks {qualifying_rows[0]} .. {qualifying_rows[N_PREVIOUS - 1]:,}", flush=True)

selected_rows = qualifying_rows[N_PREVIOUS : N_PREVIOUS + N_TARGET]
assert selected_rows.min() > LAST_SENT_RANK
assert (~is_mad[selected_rows]).all() and (n_atoms_window[selected_rows] < MAX_ATOMS).all()

walked = selected_rows.max() - LAST_SENT_RANK
skipped = walked - N_TARGET
skipped_rows = np.flatnonzero(~qualifying_mask[: selected_rows.max() + 1])
skipped_rows = skipped_rows[skipped_rows > LAST_SENT_RANK]
assert len(skipped_rows) == skipped
print(f"selected ranks {selected_rows.min():,} .. {selected_rows.max():,} "
      f"({walked:,} walked, {skipped:,} skipped: "
      f"{int(is_mad[skipped_rows].sum()):,} mad, "
      f"{int((n_atoms_window[skipped_rows] >= MAX_ATOMS).sum()):,} >= {MAX_ATOMS} atoms)", flush=True)

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

xyz_path = f"{output_stem}.xyz"
write(xyz_path, atoms_list, format="extxyz")

with open(xyz_path) as handle:
    frame_count = 0
    line = handle.readline()
    while line:
        frame_atoms = int(line)
        handle.readline()
        for _ in range(frame_atoms):
            handle.readline()
        frame_count += 1
        line = handle.readline()
assert frame_count == N_TARGET, f"written xyz holds {frame_count:,} frames"


def build_column(key, dtype):
    return pa.array([atoms.info[key] for atoms in atoms_list], dtype)


index_table = pa.table({
    "fps_rank": build_column("fps_rank", pa.int64()),
    "configuration_id": build_column("configuration_id", pa.string()),
    "dataset_id": build_column("dataset_id", pa.string()),
    "family": build_column("family", pa.string()),
    "n_atoms": build_column("n_atoms", pa.int32()),
    "periodicity": build_column("periodicity", pa.int8()),
})
pq.write_table(index_table, f"{output_stem}.parquet")

periodicity_counts = Counter(atoms.info["periodicity"] for atoms in atoms_list)
print(f"\nwrote {len(atoms_list):,} structures:")
print(f"  {xyz_path}")
print(f"  {output_stem}.parquet")
print(f"  n_atoms: min={n_atoms_window[selected_rows].min()} max={n_atoms_window[selected_rows].max()} "
      f"mean={n_atoms_window[selected_rows].mean():.1f}")
print(f"  periodicity (0=mol,1/2=surface,3=bulk): {dict(sorted(periodicity_counts.items()))}")
print(f"  top families: {Counter(atoms.info['family'] for atoms in atoms_list).most_common(8)}")

image_flag_rows = np.load(release_dir + "image_flag_rows.npy")
folded_selected = np.intersect1d(image_flag_rows, selected_rows)
print(f"  folded far-from-origin structures in selection: {folded_selected.tolist()}")
print("DONE", flush=True)
