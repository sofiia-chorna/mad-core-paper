import argparse
import json
import os

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
N_LEVELS = 25
CHUNK = 4_000_000


def per_row_files():
    """(name, dtype, values-per-row) for files that must be exactly ns rows."""
    return [("ecumetric_features.bin", np.float32, 512),
            ("ecumetric_energy.bin", np.float32, 1),
            ("voronoi_a.bin", np.int32, N_LEVELS),
            ("voronoi_d.bin", np.float32, N_LEVELS)]


def check_directory(relative):
    path = f"{RELEASE}/{relative}"
    problems = []
    ns = int(np.load(f"{path}/ns.npy")[0])

    # --- na / per-atom integrity -------------------------------------------
    na = np.load(f"{path}/na.npy", mmap_mode="r")
    if len(na) != ns + 1:
        problems.append(f"na length {len(na)} != ns+1 {ns + 1}")
    if int(na[0]) != 0:
        problems.append(f"na[0] == {int(na[0])}, expected 0")
    n_atoms = int(na[-1])
    diffs_bad = 0
    for start in range(0, len(na) - 1, CHUNK):
        block = np.asarray(na[start:start + CHUNK + 1])
        d = np.diff(block)
        diffs_bad += int((d < 0).sum())
    if diffs_bad:
        problems.append(f"na not monotonic in {diffs_bad} places")

    for name, size in [("a.bin", 4), ("x.bin", 12)]:
        expected = n_atoms * size
        actual = os.path.getsize(f"{path}/{name}")
        if actual != expected:
            problems.append(f"{name} is {actual} bytes, expected {expected}")

    for name, dtype, per_row in per_row_files():
        target = f"{path}/{name}"
        if not os.path.exists(target):
            continue
        expected = ns * per_row * np.dtype(dtype).itemsize
        actual = os.path.getsize(target)
        if actual != expected:
            problems.append(f"{name} is {actual} bytes, expected {expected}")

    # --- atomic numbers / positions ----------------------------------------
    numbers = np.memmap(f"{path}/a.bin", dtype=np.int32, mode="r")
    positions = np.memmap(f"{path}/x.bin", dtype=np.float32, mode="r")
    bad_z = nonfinite_x = 0
    for start in range(0, n_atoms, CHUNK):
        stop = min(start + CHUNK, n_atoms)
        z = np.asarray(numbers[start:stop])
        bad_z += int(((z < 1) | (z > 118)).sum())
        block = np.asarray(positions[start * 3:stop * 3])
        nonfinite_x += int((~np.isfinite(block)).sum())
    if bad_z:
        problems.append(f"{bad_z} atomic numbers outside [1,118]")
    if nonfinite_x:
        problems.append(f"{nonfinite_x} non-finite position components")

    # --- voronoi invariants -------------------------------------------------
    va_path = f"{path}/voronoi_a.bin"
    level_violations = monotonic_violations = 0
    if os.path.exists(va_path):
        a = np.memmap(va_path, dtype=np.int32, mode="r", shape=(ns, N_LEVELS))
        d = np.memmap(f"{path}/voronoi_d.bin", dtype=np.float32, mode="r",
                      shape=(ns, N_LEVELS))
        bounds = (1 << np.arange(N_LEVELS)).astype(np.int64)
        for start in range(0, ns, CHUNK):
            stop = min(start + CHUNK, ns)
            block_a = np.asarray(a[start:stop]).astype(np.int64)
            level_violations += int(((block_a < 0) | (block_a >= bounds)).sum())
            block_d = np.asarray(d[start:stop]).astype(np.float64)
            # distance to nearest centre cannot grow as levels are added
            monotonic_violations += int((np.diff(block_d, axis=1) > 1e-4).sum())
        if level_violations:
            problems.append(f"voronoi_a: {level_violations} entries outside [0, 2^k)")
        if monotonic_violations:
            problems.append(f"voronoi_d: {monotonic_violations} non-monotonic steps")

    # --- coreset-tier invariants -------------------------------------------
    if os.path.exists(f"{path}/fps_order.npy"):
        order = np.load(f"{path}/fps_order.npy", mmap_mode="r")
        mismatch = 0
        for start in range(0, ns, CHUNK):
            stop = min(start + CHUNK, ns)
            mismatch += int((np.asarray(order[start:stop])
                             != np.arange(start, stop)).sum())
        if mismatch:
            problems.append(f"fps_order != arange in {mismatch} rows")

        # epsilon_sq is non-increasing only over the FPS part; coreset224's first
        # 856,012 rows are the accrete set in accrete order, where it need not be.
        eps = np.load(f"{path}/epsilon_sq.npy", mmap_mode="r")
        rises, first_rise, previous = 0, None, None
        for start in range(0, ns, CHUNK):
            stop = min(start + CHUNK, ns)
            block = np.asarray(eps[start:stop]).astype(np.float64)
            if previous is not None:
                block = np.concatenate([[previous], block])
            step = np.diff(block)
            hit = np.nonzero(step > 1e-9 + 1e-5 * np.abs(block[:-1]))[0]
            if len(hit) and first_rise is None:
                first_rise = int(start + hit[0])
            rises += len(hit)
            previous = block[-1]
        if rises:
            problems.append(f"epsilon_sq rises at {rises} steps, first at row {first_rise}")

    # --- filter_code domain -------------------------------------------------
    counts = {}
    if os.path.exists(f"{path}/filter_code.npy"):
        code = np.load(f"{path}/filter_code.npy", mmap_mode="r")
        for start in range(0, ns, CHUNK):
            block = np.asarray(code[start:min(start + CHUNK, ns)])
            values, n = np.unique(block, return_counts=True)
            for v, k in zip(values, n):
                counts[int(v)] = counts.get(int(v), 0) + int(k)
        outside = {k: v for k, v in counts.items() if k not in (0, 1, 2, 3)}
        if outside:
            problems.append(f"filter_code values outside 0..3: {outside}")

    return {"dir": relative, "ns": ns, "n_atoms": n_atoms,
            "filter_counts": counts, "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dirs-file", default=None,
                        help="check only the directories listed here, one per line")
    args = parser.parse_args()

    if args.dirs_file:
        with open(args.dirs_file) as fh:
            everything = [line.strip() for line in fh if line.strip()]
    else:
        everything = ([f"collection/{n}"
                       for n in sorted(os.listdir(f"{RELEASE}/collection"))]
                      + ["mad_coreset", "coreset224"])
    directories = everything[args.task_id::args.num_tasks]

    results = []
    for index, relative in enumerate(directories):
        try:
            results.append(check_directory(relative))
        except Exception as error:
            results.append({"dir": relative, "problems": [f"EXCEPTION {error!r}"]})
        if index % 10 == 0:
            print(f"task {args.task_id}: {index + 1}/{len(directories)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/check_{args.task_id:04d}.json", "w") as fh:
        json.dump(results, fh, indent=1)

    bad = [r for r in results if r["problems"]]
    print(f"task {args.task_id}: {len(results)} dirs, {len(bad)} with problems", flush=True)
    for r in bad[:5]:
        print(f"   {r['dir']}: {r['problems'][:2]}", flush=True)


if __name__ == "__main__":
    main()
