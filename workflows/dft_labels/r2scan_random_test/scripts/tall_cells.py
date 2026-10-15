import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

VALID = Path("${DEV_ROOT}/phace-atomic/mad-core-valid")
SETS = VALID / "lda_pbe" / "sets"
INDEX = VALID / "delta_pbe_10k_vs_mad16" / "manifests" / "index"
RECORD = VALID / "sources" / "mad-1.6" / "record_2026.167_v4"
BOTR = VALID.parent / "mad" / "madcore-r2scan" / "madcore_r2scan_fps_ordered.xyz"
KEY = re.compile(r'(\w+)=("[^"]*"|\S+)')


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def frame_at(fh, offset):
    fh.seek(offset)
    n = int(fh.readline())
    info = {m.group(1): m.group(2).strip('"') for m in KEY.finditer(fh.readline().decode("utf-8", "replace"))}
    pos = np.array([[float(v) for v in fh.readline().split()[1:4]] for _ in range(n)])
    return n, info, pos


def frames(path):
    with open(path, "rb") as fh:
        while True:
            off = fh.tell()
            if not fh.readline():
                return
            yield off, frame_at(fh, off)


def gaps(cell, pos):
    frac = pos @ np.linalg.inv(cell)
    frac -= np.floor(frac)
    out = []
    for k in range(3):
        f = np.sort(frac[:, k])
        d = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        height = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3]))
        out.append(float(d.max() * height))
    return out


def describe(tag, info, n, pos):
    cell = np.array(info["Lattice"].split(), float).reshape(3, 3)
    g = gaps(cell, pos)
    L = np.linalg.norm(cell, axis=1)
    print(f"  {tag:60s} n={n:4d} |a|,|b|,|c|={L[0]:6.1f},{L[1]:6.1f},{L[2]:6.1f} vacuum gap a,b,c={g[0]:5.1f},{g[1]:5.1f},{g[2]:5.1f} pbc={info.get('pbc','')}")


print("== our sets: tall cells / vacuum-flagged geometries outside the surfaces file")
for arm, split in (("madcore_25k", "surfaces"), ("madcore_25k", "ttt"), ("madcore_25k", "fff"), ("voronoi_25k", "ttt"), ("voronoi_25k", "surfaces")):
    for off, (n, info, pos) in frames(SETS / arm / f"{split}.extxyz"):
        if "Lattice" not in info:
            continue
        cell = np.array(info["Lattice"].split(), float).reshape(3, 3)
        L = np.linalg.norm(cell, axis=1)
        pbc = info.get("pbc", "T T T")
        if cell[2][2] > 80 or L[0] > 80 or L[1] > 80 or "F" in pbc:
            describe(f"{arm}/{split} {info['configuration_id']} va={info.get('vacuum_axes','')}", info, n, pos)

print("\n== MAD-1.6 + BotR: frames with c > 80 A in subsets MAD does NOT dipole-correct (mc3d, mc3d_rattled, mc3d_ext, madcore_3d)")
for name, path in (("mad16_pbe", RECORD / "mad-1.6-pbe.xyz"), ("mad16_r2scan_train", RECORD / "mad-1.6-r2scan-train.xyz"),
                   ("mad16_r2scan_val", RECORD / "mad-1.6-r2scan-val.xyz"), ("mad16_r2scan_test", RECORD / "mad-1.6-r2scan-test.xyz"),
                   ("botr_r2scan", BOTR)):
    idx = read_tsv(INDEX / f"{name}.tsv")
    gap_class = Counter()
    with open(path, "rb") as fh:
        shown = 0
        for r in idx:
            if r["subset"] in ("mc2d", "mc3d_surface", "madcat") or r["pbc"] == "FFF":
                continue
            fh.seek(int(r["offset"]))
            fh.readline()
            info = {m.group(1): m.group(2).strip('"') for m in KEY.finditer(fh.readline().decode("utf-8", "replace"))}
            if "Lattice" not in info:
                continue
            cell = np.array(info["Lattice"].split(), float).reshape(3, 3)
            if cell[2][2] <= 80:
                continue
            n, info, pos = frame_at(fh, int(r["offset"]))
            g = gaps(cell, pos)
            cls = "slab_like(gap_c>6)" if g[2] > 6 else ("gap_a_or_b>6" if max(g[:2]) > 6 else "dense_tall_cell")
            gap_class[(r["subset"], cls)] += 1
            if shown < 4:
                describe(f"{name} {r['subset']}#{r['frame_id']} {r['configuration_id']}", info, n, pos)
                shown += 1
    print(f"  {name}: {dict(gap_class)}")

print("\n== madcat sample: first 300 frames of the r2scan train file, vacuum gap along c")
idx = [r for r in read_tsv(INDEX / "mad16_r2scan_train.tsv") if r["subset"] == "madcat"][:300]
cls = Counter()
with open(RECORD / "mad-1.6-r2scan-train.xyz", "rb") as fh:
    for r in idx:
        n, info, pos = frame_at(fh, int(r["offset"]))
        cell = np.array(info["Lattice"].split(), float).reshape(3, 3)
        g = gaps(cell, pos)
        cls[("gap_c>6" if g[2] > 6 else "no_gap_c", f"c={cell[2][2]:.0f}", info.get("pbc"))] += 1
print(" ", dict(cls))
