import os
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ase import Atoms
from ase.io import write

run_dir = "${DEV_ROOT}/phace-atomic/accrete/runs/madcore_10M"
full_parquet_path = f"{run_dir}/out/coreset_10M_full.parquet"
anchor_parquet_path = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"
output_dir = f"{run_dir}/recalc_100k"
N_TARGET = 100_000
MAX_ATOMS = 300

os.makedirs(output_dir, exist_ok=True)

mad_configuration_ids = pa.array(
    pq.read_table(anchor_parquet_path, columns=["configuration_id"]).column(0).to_pylist(),
    pa.string(),
)

light_table = pq.read_table(full_parquet_path, columns=["fps_order", "configuration_id", "n_sites", "family"])
fps_orders = light_table.column("fps_order").to_numpy()
n_sites_array = light_table.column("n_sites").to_numpy()
is_mad_array = pc.is_in(light_table.column("configuration_id"), value_set=mad_configuration_ids).to_numpy(zero_copy_only=False)
family_array = np.asarray(light_table.column("family").to_pylist())
print(
    f"MAD by anchor set = {int(is_mad_array.sum()):,}   "
    f"(family=='mad-coreset' = {int((family_array == 'mad-coreset').sum()):,}, cross-check)",
    flush=True,
)

qualifying_mask = (~is_mad_array) & (n_sites_array < MAX_ATOMS)
qualifying_indices = np.nonzero(qualifying_mask)[0]
selected_indices = qualifying_indices[np.argsort(fps_orders[qualifying_indices], kind="stable")][:N_TARGET]
if len(selected_indices) < N_TARGET:
    raise SystemExit(f"only {len(selected_indices):,} structures qualify (< {N_TARGET:,})")
selected_indices_sorted = np.sort(selected_indices)
print(f"non-MAD & <{MAX_ATOMS} atoms: {len(qualifying_indices):,} available -> taking first {N_TARGET:,} by fps_rank", flush=True)
print(f"  selected fps_rank range: {int(fps_orders[selected_indices].min()):,} .. {int(fps_orders[selected_indices].max()):,}", flush=True)

selected_columns = [
    "fps_order", "configuration_id", "dataset_id", "family", "n_sites", "periodicity",
    "atomic_numbers", "positions", "cell", "pbc",
]
parquet_file = pq.ParquetFile(full_parquet_path)
atoms_list = []
row_offset = 0
for row_group_index in range(parquet_file.num_row_groups):
    row_group_num_rows = parquet_file.metadata.row_group(row_group_index).num_rows
    start_pos = np.searchsorted(selected_indices_sorted, row_offset)
    end_pos = np.searchsorted(selected_indices_sorted, row_offset + row_group_num_rows)
    local_row_indices = (selected_indices_sorted[start_pos:end_pos] - row_offset).tolist()
    if local_row_indices:
        row_group_table = parquet_file.read_row_group(row_group_index, columns=selected_columns).take(local_row_indices)
        for i in range(row_group_table.num_rows):
            num_atoms = int(row_group_table.column("n_sites")[i].as_py())
            atomic_numbers = np.asarray(row_group_table.column("atomic_numbers")[i].as_py(), dtype=np.int32)
            positions = np.asarray(row_group_table.column("positions")[i].as_py(), dtype=np.float64).reshape(-1, 3)
            cell = np.asarray(row_group_table.column("cell")[i].as_py(), dtype=np.float64).reshape(3, 3)
            pbc_flags = [bool(x) for x in row_group_table.column("pbc")[i].as_py()]
            configuration_id = row_group_table.column("configuration_id")[i].as_py()
            assert len(atomic_numbers) == num_atoms and positions.shape == (num_atoms, 3), \
                f"geometry/n_sites mismatch for {configuration_id}"
            atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc_flags)
            atoms.info.update(
                fps_rank=int(row_group_table.column("fps_order")[i].as_py()),
                configuration_id=configuration_id,
                dataset_id=row_group_table.column("dataset_id")[i].as_py(),
                family=row_group_table.column("family")[i].as_py(),
                n_atoms=num_atoms,
                periodicity=int(row_group_table.column("periodicity")[i].as_py()),
            )
            atoms_list.append(atoms)
    row_offset += row_group_num_rows

atoms_list.sort(key=lambda atoms: atoms.info["fps_rank"])
assert len(atoms_list) == N_TARGET
assert len({atoms.info["configuration_id"] for atoms in atoms_list}) == N_TARGET, "duplicate cid in selection"

xyz_path = f"{output_dir}/first100k_nonMAD_lt300atoms.xyz"
write(xyz_path, atoms_list, format="extxyz")


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
pq.write_table(index_table, f"{output_dir}/first100k_nonMAD_lt300atoms.parquet")

periodicity_counts = Counter(atoms.info["periodicity"] for atoms in atoms_list)
print(f"\nwrote {len(atoms_list):,} structures:")
print(f"  {xyz_path}")
print(f"  {output_dir}/first100k_nonMAD_lt300atoms.parquet")
print(f"  n_atoms: min={min(atoms.info['n_atoms'] for atoms in atoms_list)} max={max(atoms.info['n_atoms'] for atoms in atoms_list)} "
      f"mean={np.mean([atoms.info['n_atoms'] for atoms in atoms_list]):.1f}")
print(f"  periodicity (0=mol,1/2=surface,3=bulk): {dict(sorted(periodicity_counts.items()))}")
print(f"  top families: {Counter(atoms.info['family'] for atoms in atoms_list).most_common(8)}")
print("DONE", flush=True)
