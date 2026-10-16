from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description="rebuild data/ from the large source files")
parser.add_argument("--store-root", required=True,
                    help="root of the aa002 mad-core store, the directory holding release/, mad_union/ and accrete/")
STORE = parser.parse_args().store_root
INFO = f"{STORE}/mad_union/mad_union_full.parquet"


def main():
    t = pq.read_table(INFO, columns=["fps_order", "subset"])
    order = np.argsort(t.column("fps_order").to_numpy())
    subset = np.asarray(t.column("subset").to_pylist())[order]
    n = len(subset)
    ks = [1 << k for k in range(1, int(np.floor(np.log2(n))) + 1)]

    if ks[-1] != n:
        ks.append(n)
    names = sorted(set(subset.tolist()))
    cum = {name: np.cumsum(subset == name) for name in names}
    out = HERE / "data" / "subset_counts.tsv"

    with open(out, "w") as fh:
        fh.write("k_set_size\t" + "\t".join(names) + "\n")

        for k in ks:
            fh.write(f"{k}\t" + "\t".join(str(int(cum[name][k - 1])) for name in names) + "\n")
    print(f"wrote {out}: {len(ks)} k-sets, {len(names)} subsets, {n:,} structures")


if __name__ == "__main__":
    main()
