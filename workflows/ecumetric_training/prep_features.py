import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

from extxyz_to_memmap import extxyz_to_memmap


DATA = Path("${DEV_ROOT}/phace-atomic/data")
AA = Path("${STORE_ROOT}/universal_features/mad-1.5")

SUBSETS = {
    "mad15": (str(AA / "mad-1.5-r2scan-all.xyz"), "atomization_energy"),
    "mad_next": (str(DATA / "mad_next" / "mad_next.xyz"), "corrected_energy"),
    "mad_sol": (str(DATA / "mad_sol" / "mad_sol.xyz"), "corrected_energy"),
}
ORDER = ["mad15", "mad_next", "mad_sol"]

TEACHERS = {"pet": 512, "mace": 144, "uma": 128, "dpa_omat": 240, "dpa_omol": 240}

VAL_FRAC, TEST_FRAC, SEED = 0.05, 0.05, 42


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(DATA / "atomic_combined"))
    p.add_argument("--mm", default=str(DATA / "_subset_mm"))
    p.add_argument("--chunk", type=int, default=2_000_000, help="rows per streaming chunk")
    p.add_argument("--limit", type=int, default=None,
                   help="first N structures PER SUBSET (small end-to-end test); default = all")
    return p.parse_args()


def build_subset_mm(subset, mm_root, limit=None):
    xyz, ekey = SUBSETS[subset]
    out = Path(mm_root) / subset
    if (out / "na.npy").exists() and (out / f"{ekey}.bin").exists() and (out / "forces.bin").exists():
        print(f"[mm/{subset}] exists, skip")
        return out, ekey
    tmp = Path(mm_root) / f"{subset}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    index = ":" if not limit else f":{limit}"
    print(f"[mm/{subset}] converting {xyz} (energy_key={ekey}, index={index})")
    extxyz_to_memmap(
        xyz, str(tmp),
        index=index, energy_key=ekey, forces_key="forces",
        has_forces=True, has_stress=False, has_magmoms=False,
    )
    os.rename(tmp, out)   # atomic: `out` exists only when the subset is fully written
    return out, ekey


def combine_systems(mm_root, out, ns_list, na_list, ekeys):
    total_atoms = int(sum(na[-1] for na in na_list))
    total_ns = int(sum(ns_list))

    na = np.zeros(total_ns + 1, dtype=np.int64)
    aoff = soff = 0
    for na_s in na_list:
        n_s = len(na_s) - 1
        na[soff + 1: soff + n_s + 1] = na_s[1:] + aoff
        aoff += int(na_s[-1])
        soff += n_s
    np.save(out / "na.npy", na)
    np.save(out / "ns.npy", total_ns)

    a = np.memmap(out / "a.bin", dtype="int32", mode="w+", shape=(total_atoms,))
    x = np.memmap(out / "x.bin", dtype="float32", mode="w+", shape=(total_atoms, 3))
    c = np.memmap(out / "c.bin", dtype="float32", mode="w+", shape=(total_ns, 3, 3))
    e = np.memmap(out / "e.bin", dtype="float32", mode="w+", shape=(total_ns, 1))
    f = np.memmap(out / "f.bin", dtype="float32", mode="w+", shape=(total_atoms, 3))

    aoff = soff = 0
    for subset, na_s, n_struct, ekey in zip(ORDER, na_list, ns_list, ekeys):
        d = Path(mm_root) / subset
        n_at = int(na_s[-1])
        a[aoff:aoff + n_at] = np.memmap(d / "a.bin", dtype="int32", mode="r", shape=(n_at,))
        x[aoff:aoff + n_at] = np.memmap(d / "x.bin", dtype="float32", mode="r", shape=(n_at, 3))
        f[aoff:aoff + n_at] = np.memmap(d / "forces.bin", dtype="float32", mode="r", shape=(n_at, 3))
        c[soff:soff + n_struct] = np.memmap(d / "c.bin", dtype="float32", mode="r", shape=(n_struct, 3, 3))
        e[soff:soff + n_struct] = np.memmap(d / f"{ekey}.bin", dtype="float32", mode="r", shape=(n_struct, 1))
        aoff += n_at
        soff += n_struct
    for arr in (a, x, c, e, f):
        arr.flush()
    print(f"[systems] combined ns={total_ns} atoms={total_atoms}")
    return total_atoms, total_ns


def find_teacher_files(subset, teacher):
    d = DATA / subset / teacher
    npy = sorted(glob.glob(str(d / "*_ll.npy"))) or sorted(glob.glob(str(d / "*.npy")))
    ids = sorted(glob.glob(str(d / "*_ids.npz"))) or sorted(glob.glob(str(d / "*.npz")))
    if not npy:
        raise SystemExit(f"no .npy for teacher {teacher} in {d}")
    return npy[0], (ids[0] if ids else None)


def write_teacher_raw(teacher, dim, na_list, out_bin):
    dst = np.memmap(out_bin, dtype="float32", mode="w+", shape=(int(sum(na[-1] for na in na_list)), dim))
    aoff = 0
    for subset, na_s in zip(ORDER, na_list):
        n_at = int(na_s[-1])
        npy, ids = find_teacher_files(subset, teacher)
        raw = np.load(npy, mmap_mode="r")
        if raw.shape[1] != dim:
            raise SystemExit(f"{subset}/{teacher}: dim {raw.shape[1]} != {dim}")
        if ids is None:
            # full teacher (no ids) rows atoms of structs 0..n_struct-1, in order
            dst[aoff:aoff + n_at] = raw[:n_at]
            n_nan = 0
        else:
            n_struct = len(na_s) - 1
            kept = np.load(ids)["kept"]
            kept = kept[kept < n_struct]  # limit-aware: only structures in range
            block = np.full((n_at, dim), np.nan, dtype="float32")
            src = 0
            for s in kept:
                a0, a1 = int(na_s[s]), int(na_s[s + 1])
                block[a0:a1] = raw[src:src + (a1 - a0)]
                src += a1 - a0
            dst[aoff:aoff + n_at] = block
            n_nan = n_at - src
            del block
        print(f"  [{teacher}/{subset}] atoms={n_at} nan={n_nan} ({100 * n_nan / n_at:.2f}%)")
        aoff += n_at
    dst.flush()
    del dst


def normalize_inplace(out_bin, n_rows, dim, chunk):
    arr = np.memmap(out_bin, dtype="float32", mode="r+", shape=(n_rows, dim))
    acc = np.zeros(dim, dtype=np.float64)
    nval = 0
    for i in range(0, n_rows, chunk):
        b = np.asarray(arr[i:i + chunk], dtype=np.float64)
        m = ~np.isnan(b).any(axis=1)
        acc += b[m].sum(axis=0)
        nval += int(m.sum())
    mean = acc / nval
    tot = 0.0
    for i in range(0, n_rows, chunk):
        b = np.asarray(arr[i:i + chunk], dtype=np.float64)
        m = ~np.isnan(b).any(axis=1)
        tot += np.linalg.norm(b[m] - mean, axis=1).sum()
    scale = tot / nval
    if scale == 0.0:
        raise SystemExit("zero scale")
    for i in range(0, n_rows, chunk):
        b = (np.asarray(arr[i:i + chunk], dtype=np.float64) - mean) / scale
        arr[i:i + chunk] = b.astype(np.float32)
    arr.flush()
    del arr
    return float(scale), nval, n_rows - nval


def build_splits(out, ns_list, limit=None):
    rng = np.random.default_rng(SEED)
    val_all, test_all = [], []
    soff = 0
    for subset, n in zip(ORDER, ns_list):
        if subset == "mad15" and not limit:
            val = np.loadtxt(AA / "val_indices_in_all.txt", dtype=int)
            test = np.loadtxt(AA / "test_indices_in_all.txt", dtype=int)
        else:
            perm = rng.permutation(n)
            n_val = int(round(VAL_FRAC * n))
            n_test = int(round(TEST_FRAC * n))
            val = perm[:n_val]
            test = perm[n_val:n_val + n_test]
        val_all.append(np.asarray(val) + soff)
        test_all.append(np.asarray(test) + soff)
        soff += n
    np.savetxt(out / "val_indices_in_all.txt", np.concatenate(val_all), fmt="%d")
    np.savetxt(out / "test_indices_in_all.txt", np.concatenate(test_all), fmt="%d")
    print(f"[split] val={sum(len(v) for v in val_all)} test={sum(len(t) for t in test_all)} "
          f"train={int(sum(ns_list)) - sum(len(v) for v in val_all) - sum(len(t) for t in test_all)}")


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    na_list, ns_list, ekeys = [], [], []
    for subset in ORDER:
        d, ekey = build_subset_mm(subset, args.mm, args.limit)
        na_s = np.load(d / "na.npy")
        na_list.append(na_s)
        ns_list.append(int(np.load(d / "ns.npy")))
        ekeys.append(ekey)

    total_atoms, total_ns = combine_systems(args.mm, out, ns_list, na_list, ekeys)

    stats = {}
    for teacher, dim in TEACHERS.items():
        out_bin = out / f"{teacher}.bin"
        if out_bin.exists():
            print(f"[{teacher}] exists, skip")
            continue
        print(f"[{teacher}] building (dim={dim})")
        tmp_bin = out / f"{teacher}.bin.tmp"
        write_teacher_raw(teacher, dim, na_list, tmp_bin)
        scale, nval, nnan = normalize_inplace(tmp_bin, total_atoms, dim, args.chunk)
        os.rename(tmp_bin, out_bin)   # atomic: out_bin exists only when fully built+normalized
        stats[teacher] = {"dim": dim, "scale": scale, "n_valid": nval, "n_nan_atoms": nnan}
        print(f"[{teacher}] done scale={scale:.6g} nan_atoms={nnan} ({100 * nnan / total_atoms:.2f}%)")

    build_splits(out, ns_list, args.limit)

    with open(out / "build_stats.json", "w") as fh:
        json.dump({"order": ORDER, "ns": ns_list, "total_atoms": total_atoms,
                   "total_ns": total_ns, "teachers": stats}, fh, indent=1)
    print(f"done. combined dataset: {out}")


if __name__ == "__main__":
    main()
