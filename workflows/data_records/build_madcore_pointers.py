import argparse
import shutil
from pathlib import Path

import numpy as np

RELEASE = Path("${STORE_ROOT}/mad-core/release")
CELL_FPS = RELEASE / "mad-core" / "cell_fps"
MADCORE_ROWS = 16_777_216


def build(staging):
    offsets = np.load(CELL_FPS / "center_offsets.npy")
    seed_members = offsets[:-1]

    rel_dir_all = np.load(CELL_FPS / "release_directory.npy", mmap_mode="r")
    rel_row_all = np.load(CELL_FPS / "release_row_in_directory.npy", mmap_mode="r")

    rel_dir = np.asarray(rel_dir_all[seed_members])
    rel_row = np.asarray(rel_row_all[seed_members])
    assert rel_dir.dtype == np.int16 and rel_row.dtype == np.int32
    assert len(rel_dir) == MADCORE_ROWS

    names = np.load(CELL_FPS / "release_directory_names.npy")
    group = np.load(RELEASE / "mad-core" / "dataset_group.npy")

    group_mismatches = 0
    for d in np.unique(rel_dir):
        name = names[d].decode()
        expected_group = ("mad-next" if name == "mad-next-coreset"
                          else name.split("collection/", 1)[-1]).encode()
        rows = np.flatnonzero(rel_dir == d)
        got = np.unique(group[rows])
        if not (len(got) == 1 and got[0] == expected_group):
            group_mismatches += 1
            print(f"GROUP MISMATCH {name}: dataset_group values {got[:5]}")
    assert group_mismatches == 0, "pointer directory does not match dataset_group"
    print("pointer directory == dataset_group folder rule holds for all rows")

    staging.mkdir(parents=True, exist_ok=True)
    with open(staging / "release_directory.npy", "wb") as f:
        np.save(f, rel_dir)
    with open(staging / "release_row_in_directory.npy", "wb") as f:
        np.save(f, rel_row)
    shutil.copy2(CELL_FPS / "release_directory_names.npy",
                 staging / "release_directory_names.npy")
    print(f"staged pointer files in {staging}")
    return rel_dir, rel_row, names


def verify(rel_dir, rel_row, names):
    cid = np.load(RELEASE / "mad-core" / "configuration_id.npy")

    mismatches = 0
    for d in np.unique(rel_dir):
        dir_path = RELEASE / names[d].decode()
        dir_cid = np.load(dir_path / "configuration_id.npy", mmap_mode="r")

        rows = np.flatnonzero(rel_dir == d)
        order = np.argsort(rel_row[rows], kind="stable")
        rows = rows[order]

        resolved = np.asarray(dir_cid[rel_row[rows]])
        expected = cid[rows]
        equal = resolved == expected
        bad = int((~equal).sum())
        mismatches += bad
        status = "ok" if bad == 0 else f"{bad} MISMATCHES"
        print(f"  {names[d].decode()}: {len(rows):,} rows {status}", flush=True)

    assert mismatches == 0, f"{mismatches} id round-trip failures"
    print(f"FULL id round-trip passed for all {MADCORE_ROWS:,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()

    rel_dir, rel_row, names = build(args.staging)
    verify(rel_dir, rel_row, names)


if __name__ == "__main__":
    main()
