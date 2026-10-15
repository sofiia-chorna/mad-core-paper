import glob
import os
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

V = "${DEV_ROOT}/phace-atomic/voronoi"
LEVELS = [13, 14, 15, 16]
NPROC = int(os.environ.get("NPROC", "128"))


def scan(path):
    cols = [f"a{L:02d}" for L in LEVELS]
    t = pq.read_table(path, columns=cols)
    return [np.bincount(t.column(f"a{L:02d}").to_numpy(), minlength=2 ** L) for L in LEVELS]


def main():
    files = sorted(glob.glob(f"{V}/out/assign/*/*.parquet"))
    acc = {L: np.zeros(2 ** L, dtype=np.int64) for L in LEVELS}
    with Pool(NPROC) as pool:
        for i, parts in enumerate(pool.imap_unordered(scan, files, chunksize=4)):
            for L, b in zip(LEVELS, parts):
                acc[L] += b
            if (i + 1) % 2000 == 0:
                print(f"  {i+1:,}/{len(files):,}", flush=True)

    pops = np.load(f"{V}/out/analysis/populations.npz")
    p, lo = pops["populations"], pops["level_offsets"]
    print(f"\n{'level':>6} {'truth sum':>16} {'npz sum':>16} {'identical':>10} {'truth max%':>11} {'npz max%':>10}")
    for L in LEVELS:
        truth = acc[L]
        npz = p[lo[L]:lo[L + 1]]
        same = bool(np.array_equal(truth, npz))
        print(f"{L:>6} {truth.sum():>16,} {npz.sum():>16,} {str(same):>10} "
              f"{100*truth.max()/truth.sum():>10.2f}% {100*npz.max()/npz.sum():>9.2f}%")
        np.save(f"{V}/out/analysis/populations_level{L}_verified.npy", truth)
        if not same:
            d = np.where(truth != npz)[0]
            print(f"        MISMATCH in {d.size:,}/{2**L:,} cells; first: cell {d[0]} truth={truth[d[0]]:,} npz={npz[d[0]]:,}")


if __name__ == "__main__":
    main()
