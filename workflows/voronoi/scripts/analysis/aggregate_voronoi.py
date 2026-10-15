import glob
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
assign_dir = f"{voronoi_dir}/out/assign"
analysis_dir = f"{voronoi_dir}/out/analysis"
MAX_LEVEL = 24
COMPOSITION_LEVEL = 16
NUM_WORKERS = 64

DIST_BINS = np.concatenate([[0.0], np.geomspace(1e-4, 64.0, 2048)])
level_offsets = np.concatenate([[0], np.cumsum([2 ** k for k in range(MAX_LEVEL + 1)])])
families = sorted(d for d in os.listdir(assign_dir) if os.path.isdir(f"{assign_dir}/{d}"))
family_index = {name: i for i, name in enumerate(families)}


def aggregate_chunk(file_chunk):
    populations = np.zeros(level_offsets[-1], dtype=np.int64)
    dist_hist = np.zeros((MAX_LEVEL + 1, len(DIST_BINS) - 1), dtype=np.int64)
    dist_max = np.zeros(MAX_LEVEL + 1, dtype=np.float64)
    composition = np.zeros((2 ** COMPOSITION_LEVEL, len(families)), dtype=np.int64)
    for path in file_chunk:
        columns = [f"a{k:02d}" for k in range(MAX_LEVEL + 1)] + [f"d{k:02d}" for k in range(MAX_LEVEL + 1)] + ["family"]
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        family_id = family_index[table.column("family")[0].as_py()]
        for k in range(MAX_LEVEL + 1):
            assigns = table.column(f"a{k:02d}").to_numpy()
            populations[level_offsets[k]:level_offsets[k + 1]] += np.bincount(assigns, minlength=2 ** k)
            dists = table.column(f"d{k:02d}").to_numpy()
            dist_hist[k] += np.histogram(dists, bins=DIST_BINS)[0]
            dist_max[k] = max(dist_max[k], float(dists.max()))
            if k == COMPOSITION_LEVEL:
                np.add.at(composition[:, family_id], assigns, 1)
    return populations, dist_hist, dist_max, composition


files = sorted(glob.glob(f"{assign_dir}/*/*.parquet"))
chunks = [files[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
print(f"aggregating {len(files):,} files with {NUM_WORKERS} workers", flush=True)

total_pop = np.zeros(level_offsets[-1], dtype=np.int64)
total_hist = np.zeros((MAX_LEVEL + 1, len(DIST_BINS) - 1), dtype=np.int64)
total_max = np.zeros(MAX_LEVEL + 1, dtype=np.float64)
total_comp = np.zeros((2 ** COMPOSITION_LEVEL, len(families)), dtype=np.int64)
with Pool(NUM_WORKERS) as pool:
    for i, (pop, hist, dmax, comp) in enumerate(pool.imap_unordered(aggregate_chunk, chunks)):
        total_pop += pop
        total_hist += hist
        total_max = np.maximum(total_max, dmax)
        total_comp += comp
        print(f"  worker {i + 1}/{NUM_WORKERS} merged", flush=True)

os.makedirs(analysis_dir, exist_ok=True)
np.savez_compressed(f"{analysis_dir}/populations.npz",
                    populations=total_pop, level_offsets=level_offsets)
np.savez_compressed(f"{analysis_dir}/distance_hist.npz",
                    hist=total_hist, bins=DIST_BINS, dist_max=total_max)
np.savez_compressed(f"{analysis_dir}/composition_l{COMPOSITION_LEVEL}.npz",
                    composition=total_comp, families=np.array(families, dtype=object))

n_total = int(total_pop[level_offsets[24]:level_offsets[25]].sum())
print(f"\ntotal points (level-24 populations sum): {n_total:,}", flush=True)
for k in (0, 8, 16, 20, 24):
    pop = total_pop[level_offsets[k]:level_offsets[k + 1]]
    print(f"level {k:>2}: nonempty cells {int((pop > 0).sum()):,}/{2 ** k:,}  "
          f"mean {pop.mean():,.0f}  median {int(np.median(pop)):,}  max {int(pop.max()):,}", flush=True)
print("DONE", flush=True)
