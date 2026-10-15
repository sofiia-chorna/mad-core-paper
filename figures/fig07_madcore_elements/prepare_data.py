"""Data of the 2^24 periodic table (fig:coreset224-elements).

Counts, over the full 2^24 ordering, the structures that contain each element and writes
data/element_structure_counts.tsv, which plot.py reads. Reads the store copy of the 2^24
accretion run.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase.data import chemical_symbols

HERE = Path(__file__).resolve().parent

import argparse

parser = argparse.ArgumentParser(description="rebuild data/ from the large source files")
parser.add_argument("--store-root", required=True,
                    help="root of the aa002 mad-core store, the directory holding release/, mad_union/ and accrete/")
STORE = parser.parse_args().store_root
# the 341 MB coreset_224_light.parquet carries the same non-feature columns as the 37 GB full file
INFO = f"{STORE}/accrete/phace_atomic_ep92/coreset224/out/coreset_224_light.parquet"
MAX_Z = 118


def main():
    els = pq.read_table(INFO, columns=["elements"]).column("elements").to_pylist()
    presence = np.zeros(MAX_Z + 1, dtype=np.int64)
    for e in els:
        for z in set(int(z) for z in e):
            if 0 < z <= MAX_Z:
                presence[z] += 1
    out = HERE / "data" / "element_structure_counts.tsv"
    with open(out, "w") as fh:
        fh.write("element\tstructures\n")
        for z in range(1, MAX_Z + 1):
            if presence[z] > 0:
                fh.write(f"{chemical_symbols[z]}\t{int(presence[z])}\n")
    print(f"wrote {out}: {int((presence > 0).sum())} elements, {len(els):,} structures")


if __name__ == "__main__":
    main()
