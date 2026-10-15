import os
import re
import sys
import numpy as np
from pathlib import Path

from mad.madcore_vs_random_20k.xyz_write import vacuum_axes

SPLITS = ("ttt", "fff", "surfaces")


def frames_of(path):
    with open(path) as fh:
        while True:
            count_line = fh.readline()
            if not count_line:
                break
            n = int(count_line)
            comment = fh.readline()
            atom_lines = [fh.readline() for _ in range(n)]
            yield count_line, comment, atom_lines


def reclassify(comment, atom_lines):
    pbc_str = re.search(r'pbc="([TF ]+)"', comment).group(1)
    n_periodic = pbc_str.count("T")

    if n_periodic == 0:
        return "fff", -1
    if n_periodic < 3:
        return "surfaces", -1

    lattice = re.search(r'Lattice="([^"]+)"', comment).group(1)
    cell = np.array(lattice.split(), dtype=np.float64).reshape(3, 3)
    positions = np.array([line.split()[1:4] for line in atom_lines], dtype=np.float64)

    vac = vacuum_axes(positions, cell)
    if vac == 0:
        return "ttt", 0
    if vac == 3:
        return "fff", 3
    return "surfaces", vac


def main(out_dir):
    out_dir = Path(out_dir)

    result_by_id = {}
    counts = {name: 0 for name in SPLITS}
    new_files = {name: open(out_dir / f"{name}.extxyz.new", "w") for name in SPLITS}

    for name in SPLITS:
        for count_line, comment, atom_lines in frames_of(out_dir / f"{name}.extxyz"):
            split, vac = reclassify(comment, atom_lines)
            counts[split] += 1

            cid = re.search(r'configuration_id="([^"]+)"', comment).group(1)
            result_by_id[cid] = (vac, split)

            new_files[split].write(count_line + comment + "".join(atom_lines))

    for fh in new_files.values():
        fh.close()

    manifest_path = out_dir / "selection.tsv"
    with open(manifest_path) as fh:
        header = next(fh).rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in fh]

    id_col = header.index("configuration_id")
    split_col = header.index("split_file")

    if "vacuum_axes" not in header:
        header.insert(split_col, "vacuum_axes")
        for row in rows:
            row.insert(split_col, "")

    vac_col = header.index("vacuum_axes")
    split_col = header.index("split_file")

    for row in rows:
        vac, split = result_by_id[row[id_col]]
        row[vac_col] = str(vac)
        row[split_col] = split

    with open(out_dir / "selection.tsv.new", "w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")

    for name in SPLITS:
        os.replace(out_dir / f"{name}.extxyz.new", out_dir / f"{name}.extxyz")
    os.replace(out_dir / "selection.tsv.new", manifest_path)

    print(f"resplit {out_dir}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main(sys.argv[1])
