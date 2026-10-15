import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VALID = Path("${DEV_ROOT}/phace-atomic/mad-core-valid")
SETS = VALID / "lda_pbe" / "sets"
LABELLED = VALID / "lda_pbe" / "labelled"
INDEX = VALID / "delta_pbe_10k_vs_mad16" / "manifests" / "index"
RECORD = VALID / "sources" / "mad-1.6" / "record_2026.167_v4"
BOTR = VALID.parent / "mad" / "madcore-r2scan" / "madcore_r2scan_fps_ordered.xyz"
SAMPLE = VALID / "r2scan_random_test" / "manifests" / "test_1000.tsv"
KEY = re.compile(r'(\w+)=("[^"]*"|\S+)')
TTF_MIN_C = 80.0


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def headers(path):
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            n = int(line)
            info = {m.group(1): m.group(2).strip('"') for m in KEY.finditer(fh.readline().decode("utf-8", "replace"))}
            for _ in range(n):
                fh.readline()
            yield n, info


def header_at(fh, offset):
    fh.seek(offset)
    fh.readline()
    return {m.group(1): m.group(2).strip('"') for m in KEY.finditer(fh.readline().decode("utf-8", "replace"))}


def geom(info):
    cell = np.array(info["Lattice"].split(), float).reshape(3, 3) if "Lattice" in info else None
    pbc = [c == "T" for c in info["pbc"].split()] if "pbc" in info else [False] * 3
    return cell, pbc


def rule_bucket(split_file, cell, pbc):
    if split_file == "fff":
        return "isolated_fff"
    if split_file == "ttt":
        return "periodic_ttt"
    if (not pbc[2]) or cell[2][2] > TTF_MIN_C:
        return "surfaces_ttf"
    return "periodic_ttt"


def flags(cell, pbc):
    if cell is None:
        return "no_cell"
    out = []
    if not pbc[2]:
        out.append("pbc_z_F")
    if not (pbc[0] and pbc[1]):
        out.append("pbc_xy_F")
    if cell[2][2] > TTF_MIN_C:
        out.append("c_gt_80")
    lengths = np.linalg.norm(cell, axis=1)
    if lengths[0] > TTF_MIN_C or lengths[1] > TTF_MIN_C:
        out.append("a_or_b_gt_80")
    return "+".join(out) if out else "plain"


report = {}

print("== A. our PBE campaign: manifest dft_bucket vs the 80 A rule applied to sets/ geometries", flush=True)
for arm in ("madcore_25k", "random_25k", "voronoi_25k", "random_test_6k"):
    man = read_tsv(LABELLED / arm / "manifest.tsv")
    by_key = {}
    scope = Counter()
    for split in ("ttt", "surfaces", "fff"):
        for row, (n, info) in enumerate(headers(SETS / arm / f"{split}.extxyz")):
            cell, pbc = geom(info)
            by_key[(split, str(row))] = (rule_bucket(split, cell, pbc), flags(cell, pbc))
            scope[(split, flags(cell, pbc))] += 1
    mism = Counter()
    examples = defaultdict(list)
    checked = 0
    for r in man:
        if not r["dft_bucket"]:
            continue
        checked += 1
        exp, fl = by_key[(r["split_file"], r["split_row"])]
        if exp != r["dft_bucket"]:
            mism[(r["split_file"], fl, exp, r["dft_bucket"])] += 1
            if len(examples[(exp, r["dft_bucket"])]) < 3:
                examples[(exp, r["dft_bucket"])].append(r["configuration_id"])
    print(f"{arm}: {checked} labelled rows checked; mismatches: {sum(mism.values())}")
    for k, v in sorted(mism.items()):
        print(f"   split={k[0]} geometry={k[1]} rule_says={k[2]} computed_as={k[3]}: {v}  e.g. {examples[(k[2], k[3])]}")
    print("   geometries outside the rule's scope (ttt/fff files with c>80 or pbc_z F, never dipole-corrected):")
    for (split, fl), v in sorted(scope.items()):
        if split != "surfaces" and fl not in ("plain", "no_cell"):
            print(f"      {split} {fl}: {v}")
    print("   surfaces file by geometry flag:", {fl: v for (s, fl), v in scope.items() if s == "surfaces"})
    report[arm] = {"checked": checked, "mismatches": sum(mism.values())}

print("\n== B. MAD-1.6 record files: subset (MAD dipole rule = mc2d + mc3d_surface) vs cell/pbc geometry", flush=True)
for name, path in (("mad16_pbe", RECORD / "mad-1.6-pbe.xyz"), ("mad16_r2scan_train", RECORD / "mad-1.6-r2scan-train.xyz"),
                   ("mad16_r2scan_val", RECORD / "mad-1.6-r2scan-val.xyz"), ("mad16_r2scan_test", RECORD / "mad-1.6-r2scan-test.xyz"),
                   ("botr_r2scan", BOTR)):
    idx = read_tsv(INDEX / f"{name}.tsv")
    tab = Counter()
    with open(path, "rb") as fh:
        for r in idx:
            info = header_at(fh, int(r["offset"]))
            cell, pbc = geom(info)
            tab[(r["subset"], flags(cell, pbc))] += 1
    print(f"{name}: {len(idx)} frames")
    for (subset, fl), v in sorted(tab.items()):
        dip = subset in ("mc2d", "mc3d_surface")
        rule = "dipole" if ("c_gt_80" in fl or "pbc_z_F" in fl) else ("n/a" if fl == "no_cell" else "no_dipole")
        mad = "dipole" if dip else "no_dipole"
        mark = "" if (rule == "n/a" or rule == mad) else "   <-- differs from MAD subset rule" if name != "botr_r2scan" else "   <-- rule says dipole; BotR treatment unknown"
        print(f"   {subset:18s} {fl:22s} {v:7d}  80A-rule={rule:9s} MAD-subset-rule={mad}{mark}")

print("\n== C. the 1,000 sample and the whole test arm: periodic frames vs the rule", flush=True)
sample = {r["configuration_id"]: r for r in read_tsv(SAMPLE)}
tab = Counter()
worst = []
for n, info in headers(LABELLED / "random_test_6k" / "pbe.xyz"):
    cell, pbc = geom(info)
    if cell is None:
        continue
    fl = flags(cell, pbc)
    in_sample = info["configuration_id"] in sample
    tab[(in_sample, info.get("vacuum_axes", ""), fl)] += 1
    if in_sample and info.get("vacuum_axes") in ("1", "2"):
        worst.append((float(cell[2][2]), float(max(np.linalg.norm(cell, axis=1))), info["configuration_id"]))
for k, v in sorted(tab.items()):
    print(f"   in_sample={k[0]!s:5s} vacuum_axes={k[1]:2s} geometry={k[2]:20s} {v}")
worst.sort(reverse=True)
print("   sample vacuum-gap frames: largest c and largest lattice length:", worst[:3])
