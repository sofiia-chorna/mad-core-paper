import numpy as np
from ase.io import iread, write

project_dir = "${DEV_ROOT}/phace-atomic"
output_dir = f"{project_dir}/accrete/runs/madcore_production/bor"
failed_frames_path = f"{project_dir}/flashmd_failed_frames.xyz"
detail_path = f"{output_dir}/failed_frame_ids_BOR_detail.tsv"
output_path = f"{output_dir}/failed_frames_in_bor_range.xyz"

with open(detail_path) as handle:
    next(handle)
    rows = [line.rstrip("\n").split("\t") for line in handle]

by_frame_id = {int(frame_id): (int(fps_order), int(n_atoms), configuration_id)
               for fps_order, n_atoms, configuration_id, frame_id in rows}
print(f"target frames from {detail_path.split('/')[-1]}: {len(by_frame_id):,}")

selected = []
n_seen = 0

for atoms in iread(failed_frames_path, format="extxyz"):
    n_seen += 1
    frame_id = int(atoms.info["frame_id"])
    if frame_id not in by_frame_id:
        continue

    fps_order, n_atoms, configuration_id = by_frame_id[frame_id]
    if len(atoms) != n_atoms:
        raise RuntimeError(f"frame_id {frame_id}: xyz has {len(atoms)} atoms, "
                           f"the table says {n_atoms}")

    atoms.info["configuration_id"] = configuration_id
    atoms.info["fps_order"] = fps_order
    selected.append((fps_order, atoms))

print(f"scanned {n_seen:,} frames, selected {len(selected):,}")

missing = set(by_frame_id) - {int(atoms.info["frame_id"]) for _, atoms in selected}
if missing:
    raise RuntimeError(f"{len(missing):,} frame_ids from the table not found in the xyz")

selected.sort(key=lambda item: item[0])
write(output_path, [atoms for _, atoms in selected], format="extxyz")

sizes = np.array([len(atoms) for _, atoms in selected])
orders = np.array([fps_order for fps_order, _ in selected])

print(f"\nwrote {output_path}")
print(f"  frames    : {len(selected):,}")
print(f"  fps_order : {orders.min():,} .. {orders.max():,}")
print(f"  n_atoms   : {sizes.min()} .. {sizes.max()}  (mean {sizes.mean():.1f})")
print(f"  info keys : {sorted(selected[0][1].info)}")
