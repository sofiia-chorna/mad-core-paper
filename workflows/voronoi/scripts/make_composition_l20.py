import glob
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
assign_dir = f"{voronoi_dir}/out/assign"
analysis_dir = f"{voronoi_dir}/out/analysis"
LEVEL = 20
N_LEVEL = 2 ** LEVEL
NPROC = int(os.environ.get("COMP_NPROC", "128"))

families = [str(f) for f in np.load(f"{analysis_dir}/composition_l16.npz",
                                    allow_pickle=True)["families"]]
family_index = {f: i for i, f in enumerate(families)}


def scan(path):
    family = path.split("/assign/")[1].split("/")[0]
    a20 = pq.read_table(path, columns=[f"a{LEVEL:02d}"]).column(f"a{LEVEL:02d}").to_numpy()
    cells, counts = np.unique(a20, return_counts=True)
    return family, cells.astype(np.int64), counts.astype(np.int64)


def main():
    files = sorted(glob.glob(f"{assign_dir}/*/*.parquet"))
    seen = sorted({p.split("/assign/")[1].split("/")[0] for p in files})
    missing = [f for f in seen if f not in family_index]
    if missing:
        raise RuntimeError(f"families not in composition_l16 axis: {missing}")
    print(f"scanning {len(files):,} files, {len(seen)} families, {NPROC} workers", flush=True)

    composition = np.zeros((N_LEVEL, len(families)), dtype=np.int64)
    total = 0
    with Pool(NPROC) as pool:
        for index, (family, cells, counts) in enumerate(
                pool.imap_unordered(scan, files, chunksize=4)):
            composition[cells, family_index[family]] += counts
            total += int(counts.sum())
            if (index + 1) % 1000 == 0:
                print(f"  {index + 1:,}/{len(files):,} files, {total:,} points", flush=True)

    print(f"\ntotal points: {total:,}", flush=True)
    np.savez_compressed(f"{analysis_dir}/composition_l20.npz",
                        composition=composition, families=np.array(families, dtype=object))

    counts20 = composition.sum(1)
    reference = np.load(f"{analysis_dir}/populations_level20.npy")
    print(f"row sums match populations_level20.npy: {bool(np.array_equal(counts20, reference))}", flush=True)

    top = np.argsort(counts20)[::-1][:5]
    print("\nMEMBER composition of the biggest level-20 cells:")
    for rank, cell in enumerate(top):
        row = composition[cell]
        order = np.argsort(row)[::-1][:6]
        print(f"\n  rank {rank}: cell {cell:,} -> {counts20[cell]:,} points "
              f"({100 * counts20[cell] / total:.2f}% of library)")
        for j in order:
            if row[j] > 0:
                print(f"        {row[j]:>14,}  {100 * row[j] / counts20[cell]:>6.2f}%  {families[j]}")


if __name__ == "__main__":
    main()
