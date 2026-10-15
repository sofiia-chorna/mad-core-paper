import argparse
import glob
import hashlib
import json
import os

import numpy as np

OUT_ROOT = ("${DEV_ROOT}/phace-atomic/voronoi/out/"
            "cell_fps")
RELEASE = "${STORE_ROOT}/mad-core/release"
N_CELLS = 2 ** 24


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--allocate", action="store_true")
    args = parser.parse_args()
    os.makedirs(f"{args.out}/ids", exist_ok=True)

    # per-cell row counts, tallied independently during the release scan
    counts = np.zeros(N_CELLS, dtype=np.int64)
    parts = sorted(glob.glob(f"{OUT_ROOT}/rowid/logs/cell_rows_*.npy"))
    for path in parts:
        counts += np.load(path)
    n = int(counts.sum())
    print(f"cell_rows tallies merged: {len(parts)}   total rows {n:,}")

    population = np.load(f"{RELEASE}/coreset224/voronoi_population.npy")
    mismatch = int((population != counts).sum())
    print(f"vs coreset224/voronoi_population: {'MATCH' if mismatch == 0 else mismatch}"
          f" cells differ   (sum {int(population.sum()):,})")
    if mismatch:
        raise SystemExit("center_offsets would disagree with voronoi_population")

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(f"{args.out}/center_offsets.npy", offsets)
    print(f"wrote center_offsets.npy  [-1] = {int(offsets[-1]):,}")

    centre = np.load(f"{RELEASE}/coreset224/configuration_id.npy")
    np.save(f"{args.out}/center_configuration_id.npy", centre)
    print(f"wrote center_configuration_id.npy  {centre.dtype}  {centre.shape}")

    # directory names in index order, omitting sources deleted from the release
    # (build_binary shifts the per-row indices past them to match)
    with open(f"{OUT_ROOT}/plan.json") as fh:
        plan = json.load(fh)
    dead = set(plan.get("deleted_sources", {}).get("indices", ()))
    names = [s["name"] for i, s in enumerate(plan["sources"]) if i not in dead]
    folders = [n if n == "mad_coreset" else f"collection/{n}" for n in names]
    np.save(f"{args.out}/release_directory_names.npy",
            np.array(folders, dtype="S"))
    shas = []
    for folder in folders:
        manifest = f"{RELEASE}/{folder}/MANIFEST.json"
        shas.append(hashlib.sha256(open(manifest, "rb").read()).hexdigest())
    np.save(f"{args.out}/release_manifest_sha256.npy", np.array(shas, dtype="S64"))
    print(f"wrote release_directory_names.npy ({len(folders)}) and manifest hashes")

    if not args.allocate:
        estimate = n * (4 + 4 + 2 + 4 + 8) + n * 32.3
        print(f"dry-run: per-row arrays would need ~{estimate / 1e9:.1f} GB")
        return

    # real .npy files (header + data) so the format matches the spec and the
    # parallel builders can memmap them read-write
    for name, dtype in [("fps_order_in_cell", np.int32), ("epsilon_sq", np.float32),
                        ("release_directory", np.int16),
                        ("release_row_in_directory", np.int32)]:
        path = f"{args.out}/{name}.npy"
        if os.path.exists(path):
            existing = np.lib.format.open_memmap(path, mode="r")
            if existing.shape == (n,) and existing.dtype == dtype:
                print(f"{name} already allocated")
                continue
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n,))
        array[:] = -1 if dtype in (np.int32, np.int16) else np.nan
        array.flush()
        del array
        print(f"allocated {name} ({n * np.dtype(dtype).itemsize / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
