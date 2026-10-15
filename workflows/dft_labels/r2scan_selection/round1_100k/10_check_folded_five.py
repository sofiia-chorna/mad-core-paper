import re

import numpy as np

release_dir = "${STORE_ROOT}/mad-core/release/mad-core/"
project_dir = "${DEV_ROOT}/phace-atomic"
shared_path = f"{project_dir}/cscs-madcore_first100k_non-mad_max300 (3).xyz"

rank_pattern = re.compile(r"fps_rank=(\d+)")

image_flag_rows = np.load(release_dir + "image_flag_rows.npy")
image_flags = np.memmap(release_dir + "image_flags.bin", dtype=np.int64, mode="r").reshape(-1, 3)
image_flag_cells = np.memmap(release_dir + "image_flag_cells.bin", dtype=np.float64, mode="r").reshape(-1, 3, 3)

na = np.load(release_dir + "na.npy", mmap_mode="r")
positions = np.memmap(release_dir + "x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
dataset_group = np.load(release_dir + "dataset_group.npy", mmap_mode="r")

flag_offsets = np.zeros(len(image_flag_rows) + 1, dtype=np.int64)
flag_offsets[1:] = np.cumsum([na[row + 1] - na[row] for row in image_flag_rows])

target_ranks = [60919, 63706, 96880, 100156, 100985]
shared_positions = {}
with open(shared_path) as handle:
    line = handle.readline()
    while line:
        n_atoms = int(line)
        comment = handle.readline()
        rank = int(rank_pattern.search(comment).group(1))
        rows = [handle.readline() for _ in range(n_atoms)]
        if rank in target_ranks:
            shared_positions[rank] = np.array([[float(v) for v in row.split()[1:4]] for row in rows])
        line = handle.readline()

print(f"{'rank':>8} {'group':>10} {'atoms':>6} {'max |orig-shared|':>20} {'max |x|_shared':>16}")
for rank in target_ranks:
    slot = int(np.flatnonzero(image_flag_rows == rank)[0])
    flags = np.asarray(image_flags[flag_offsets[slot]:flag_offsets[slot + 1]], dtype=np.float64)
    cell = np.asarray(image_flag_cells[slot], dtype=np.float64)
    folded = np.asarray(positions[na[rank]:na[rank + 1]], dtype=np.float64)

    original = folded + flags @ cell
    delta = np.abs(original - shared_positions[rank]).max()
    print(f"{rank:>8} {dataset_group[rank].decode():>10} {len(folded):>6} "
          f"{delta:>20.3e} {np.abs(shared_positions[rank]).max():>16.3e}")
