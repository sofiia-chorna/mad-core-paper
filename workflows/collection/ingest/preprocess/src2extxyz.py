import argparse
import os

from ase import Atoms
from ase.io import read, write


def main():
    ap = argparse.ArgumentParser(description="Read cif/xyz via ASE and write clean extxyz for the nc_ingest xyz reader.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    frames = []
    for src in args.inputs:
        stem = os.path.splitext(os.path.basename(src))[0]
        for i, a in enumerate(read(src, index=":")):
            clean = Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc)
            clean.info = {"entry_id": f"{stem}:{i}"}
            frames.append(clean)

    write(args.out, frames, format="extxyz")
    print(f"[src2extxyz] wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
