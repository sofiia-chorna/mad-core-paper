import numpy as np
from pathlib import Path

from mad.madcore_vs_random_20k.xyz_write import classify, write_frame

RELEASE = Path("${STORE_ROOT}/mad-core/release/mad-core")
OUT = Path(__file__).resolve().parent / "out" / "madcore_25k"
TARGET = 25000
MAX_ATOMS = 300
SCAN_ROWS = 65536


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    na = np.asarray(np.load(RELEASE / "na.npy", mmap_mode="r")[: SCAN_ROWS + 1])
    natoms = np.diff(na)
    group = np.asarray(np.load(RELEASE / "dataset_group.npy", mmap_mode="r")[:SCAN_ROWS])
    pbc = np.asarray(np.load(RELEASE / "pbc.npy", mmap_mode="r")[:SCAN_ROWS])

    is_mad = group == b"mad-next"
    eligible = is_mad | (natoms < MAX_ATOMS)
    picked = np.flatnonzero(eligible)[:TARGET]
    assert len(picked) == TARGET, f"only {len(picked)} eligible rows in first {SCAN_ROWS}"

    last_row = picked[-1]
    print(f"selected {TARGET} rows from fps_order 0..{last_row}: "
          f"{int(is_mad[picked].sum())} mad-next (no atom cap), "
          f"{int((~is_mad[picked]).sum())} others (< {MAX_ATOMS} atoms)")
    print(f"skipped: {int((~eligible[: last_row + 1]).sum())} non-mad rows with >= {MAX_ATOMS} atoms")

    atom_end = na[last_row + 1]
    a = np.memmap(RELEASE / "a.bin", dtype=np.int32, mode="r", shape=(atom_end,))
    x = np.memmap(RELEASE / "x.bin", dtype=np.float32, mode="r", shape=(atom_end, 3))
    c = np.memmap(RELEASE / "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3)
    cid = np.asarray(np.load(RELEASE / "configuration_id.npy", mmap_mode="r")[: last_row + 1])
    dataset_id = np.asarray(np.load(RELEASE / "dataset_id.npy", mmap_mode="r")[: last_row + 1])

    image_flag_rows = np.load(RELEASE / "image_flag_rows.npy")
    folded = np.intersect1d(picked, image_flag_rows)
    if len(folded):
        print(f"note: {len(folded)} selected rows have folded coordinates (image-flag rows): {folded.tolist()}")

    files = {name: open(OUT / f"{name}.extxyz", "w") for name in ("ttt", "fff", "surfaces")}
    manifest = open(OUT / "selection.tsv", "w")
    manifest.write("fps_order\tconfiguration_id\tdataset_group\tdataset_id\tnatoms\tpbc\tvacuum_axes\tsplit_file\n")

    counts = {"ttt": 0, "fff": 0, "surfaces": 0}

    for row in picked:
        split, vac = classify(x[na[row]: na[row + 1]], c[row], pbc[row])
        counts[split] += 1

        info = {
            "configuration_id": cid[row].decode(),
            "dataset_group": group[row].decode(),
            "dataset_id": dataset_id[row].decode(),
            "fps_order": str(row),
        }
        write_frame(files[split], a[na[row]: na[row + 1]], x[na[row]: na[row + 1]],
                    c[row], pbc[row], info)

        pbc_str = "".join("T" if flag else "F" for flag in pbc[row])
        manifest.write(f"{row}\t{info['configuration_id']}\t{info['dataset_group']}\t"
                       f"{info['dataset_id']}\t{na[row + 1] - na[row]}\t{pbc_str}\t{vac}\t{split}\n")

    for fh in files.values():
        fh.close()
    manifest.close()

    print(f"written to {OUT}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
