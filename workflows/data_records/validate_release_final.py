import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

RELEASE = Path("${STORE_ROOT}/mad-core/release")
REPO = Path("${DEV_ROOT}/phace-atomic")
AUDIT_OUT = RELEASE / "audit" / "final_validation"
REPO_OUT = REPO / "release" / "build" / "audit" / "final_validation"

COLLECTION_ROWS = 1_693_646_343
RELEASE_ROWS = 1_694_502_355
MADCORE_ROWS = 16_777_216
MADNEXT_ROWS = 856_012
ACCRETE_ROWS = 856_012
CLEAN_COLLECTION_ROWS = 1_641_975_594 - MADNEXT_ROWS

FOLDED_DIRS = {
    "mad-core": 29,
    "collection/nomad": 68_993,
    "collection/mcloudarchive__2ftdw-8cr18": 1_088,
    "collection/mcloudarchive__b4kgx-rtd78": 3,
}
NO_CELL_OUTLIER_DIR = "collection/mcloudarchive__vv6sy-98m49"
MAX_COORDINATE = 1e4
SPLIT_COUNTS = {b"train": 180_184, b"val": 18_305, b"test": 18_314, b"": 639_209}
MAD_DATASET_IDS = {b"MAD-1.5", b"mad_next", b"mad_sol"}

CHUNK_ROWS = 1 << 22
failures = []


def check(condition, message):
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {message}", flush=True)
    if not condition:
        failures.append(message)


def collection_dirs():
    return sorted(p for p in (RELEASE / "collection").iterdir() if p.is_dir())


def load_geometry(d):
    ns = int(np.load(d / "ns.npy")[0])
    na = np.load(d / "na.npy", mmap_mode="r")
    return ns, na


SPLIT_WEIGHT = 8_000_000_000


def light_checks(dir_path):
    d = RELEASE / dir_path
    result = {"dir": dir_path, "errors": [], "notes": []}
    err = result["errors"].append

    ns, na = load_geometry(d)
    n_atoms = int(na[-1])
    result["ns"] = ns
    result["n_atoms"] = n_atoms

    if not (len(na) == ns + 1 and na[0] == 0 and np.all(np.diff(np.asarray(na)) > 0)):
        err("na not cumulative strictly increasing from 0")

    with open(d / "MANIFEST.json") as f:
        manifest = json.load(f)
    if manifest["ns"] != ns:
        err(f"MANIFEST ns {manifest['ns']} != ns.npy {ns}")
    if "row_order_shards" in manifest:
        shard_sum = sum(count for _, count in manifest["row_order_shards"])
        if shard_sum != ns:
            err(f"row_order_shards sum {shard_sum} != ns {ns}")

    for name, entry in manifest["files"].items():
        path = d / name
        if not path.exists():
            err(f"manifest file missing on disk: {name}")
            continue
        actual = path.stat().st_size
        if actual != entry["bytes"]:
            err(f"{name} size {actual} != manifest {entry['bytes']}")
    known_extra = {"MANIFEST.json", "x.bin.prefold", "image_flag_rows.npy",
                   "image_flags.bin", "image_flag_cells.bin", "extxyz", "cell_fps",
                   "release_directory.npy", "release_row_in_directory.npy",
                   "release_directory_names.npy"}
    extra = [p.name for p in d.iterdir()
             if p.name not in manifest["files"] and p.name not in known_extra]
    if extra:
        err(f"files on disk not in manifest: {sorted(extra)}")

    expected_sizes = {
        "a.bin": n_atoms * 4,
        "x.bin": n_atoms * 12,
        "c.bin": ns * 36,
        "ecumetric_energy.bin": ns * 4,
        "ecumetric_nc_forces.bin": n_atoms * 12,
        "ecumetric_features.bin": ns * 2048,
    }
    for name, expected in expected_sizes.items():
        if (d / name).exists() and (d / name).stat().st_size != expected:
            err(f"{name} size != {expected}")

    result["id_width"] = int(np.load(d / "configuration_id.npy", mmap_mode="r").dtype.itemsize)

    if (d / "source_reference_offsets.npy").exists():
        off = np.load(d / "source_reference_offsets.npy", mmap_mode="r")
        if len(off) != ns + 1 or off[0] != 0 or not np.all(np.diff(np.asarray(off)) >= 0):
            err("source_reference_offsets not monotone (ns+1,) from 0")
        elif int(off[-1]) != (d / "source_reference_bytes.bin").stat().st_size:
            err("source_reference_bytes.bin size != offsets[-1]")
    elif dir_path != "mad-next-coreset":
        err("source_reference_offsets.npy missing")

    if dir_path.startswith("collection/"):
        fc = np.load(d / "filter_code.npy")
        dup = np.load(d / "is_duplicate.npy")
        values = np.unique(fc)
        if not set(values.tolist()) <= {0, 1, 2, 3}:
            err(f"unexpected filter_code values {values.tolist()}")
        result["filter_counts"] = {int(v): int((fc == v).sum()) for v in values}
        result["duplicates"] = int(dup.sum())
        result["clean_rows"] = int(((fc == 0) & ~dup).sum())

    flagged = FOLDED_DIRS.get(dir_path)
    has_flags = (d / "image_flag_rows.npy").exists()
    if (flagged is not None) != has_flags:
        err(f"image flag files present={has_flags}, expected={flagged is not None}")
    if flagged is not None and has_flags:
        rows = np.load(d / "image_flag_rows.npy")
        flags_size = (d / "image_flags.bin").stat().st_size
        cells_size = (d / "image_flag_cells.bin").stat().st_size
        na_arr = np.asarray(na)
        flagged_atoms = int((na_arr[rows + 1] - na_arr[rows]).sum())
        if len(rows) != flagged:
            err(f"{len(rows)} flagged rows != expected {flagged}")
        if not (np.all(np.diff(rows) > 0) and rows[0] >= 0 and rows[-1] < ns):
            err("image_flag_rows not sorted unique in range")
        if flags_size != flagged_atoms * 24:
            err(f"image_flags.bin {flags_size} != {flagged_atoms * 24}")
        if cells_size != len(rows) * 72:
            err(f"image_flag_cells.bin {cells_size} != {len(rows) * 72}")
        flags = np.memmap(d / "image_flags.bin", dtype=np.int64, mode="r").reshape(-1, 3)
        spans = np.concatenate([[0], np.cumsum(na_arr[rows + 1] - na_arr[rows])])
        zero_rows = sum(
            1 for j in range(len(rows))
            if not np.any(flags[spans[j]:spans[j + 1]])
        )
        if zero_rows:
            err(f"{zero_rows} flagged rows with all-zero flags")

    return result


def range_scan(dir_path, lo, hi):
    d = RELEASE / dir_path
    ns, na = load_geometry(d)
    partial = {"dir": dir_path, "errors": [], "max_abs": 0.0, "far_rows": 0,
               "empty_ids": 0, "pbc_true_zero": 0, "pbc_false_nonzero": 0,
               "dataset_ids": [], "d_nan": 0, "d_negative": 0,
               "level_max": None, "level_min": None, "hist": None}
    err = partial["errors"].append

    a_lo, a_hi = int(na[lo]), int(na[hi])
    x = np.memmap(d / "x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
    max_abs = 0.0
    far_rows = set()
    atoms_per_chunk = CHUNK_ROWS * 8
    for start in range(a_lo, a_hi, atoms_per_chunk):
        stop = min(start + atoms_per_chunk, a_hi)
        block = np.abs(x[start:stop])
        block_max = float(block.max()) if stop > start else 0.0
        max_abs = max(max_abs, block_max)
        if block_max > MAX_COORDINATE:
            bad_atoms = start + np.flatnonzero((block > MAX_COORDINATE).any(axis=1))
            far_rows.update((np.searchsorted(np.asarray(na), bad_atoms, side="right") - 1).tolist())
    partial["max_abs"] = max_abs
    partial["far_rows"] = len(far_rows)
    partial["far_row_list"] = sorted(far_rows)[:200]

    c = np.memmap(d / "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3)
    pbc = np.load(d / "pbc.npy", mmap_mode="r")
    for start in range(lo, hi, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, hi)
        cell_axis_nonzero = (np.abs(c[start:stop]) > 0).any(axis=2)
        p = np.asarray(pbc[start:stop])
        partial["pbc_true_zero"] += int((p & ~cell_axis_nonzero).sum())
        partial["pbc_false_nonzero"] += int((~p & cell_axis_nonzero).sum())

    cid = np.load(d / "configuration_id.npy", mmap_mode="r")
    minted = re.compile(rb"^[^:]+:[0-9a-f]{32}$")
    bad_format = 0
    for start in range(lo, hi, CHUNK_ROWS):
        block = np.asarray(cid[start:min(start + CHUNK_ROWS, hi)])
        partial["empty_ids"] += int((block == b"").sum())
        sample = block[:: max(1, len(block) // 2000)]
        bad_format += sum(1 for s in sample if not minted.match(bytes(s)))
    if bad_format:
        err(f"{bad_format} sampled configuration_ids not in minted <family>:<hex32> form")

    dataset_ids = set()
    did = np.load(d / "dataset_id.npy", mmap_mode="r")
    for start in range(lo, hi, CHUNK_ROWS):
        dataset_ids.update(np.unique(np.asarray(did[start:min(start + CHUNK_ROWS, hi)])).tolist())
    partial["dataset_ids"] = sorted(v.decode() for v in dataset_ids)

    if dir_path.startswith("collection/") or dir_path == "mad-next-coreset":
        va = np.memmap(d / "voronoi_a.bin", dtype=np.int32, mode="r").reshape(ns, 25)
        vd = np.memmap(d / "voronoi_d.bin", dtype=np.float32, mode="r").reshape(ns, 25)
        hist = np.zeros(MADCORE_ROWS, dtype=np.int64)
        level_max = np.full(25, -1, dtype=np.int64)
        level_min = np.full(25, 2**31, dtype=np.int64)
        for start in range(lo, hi, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, hi)
            block = np.asarray(va[start:stop])
            counts = np.bincount(block[:, 24], minlength=MADCORE_ROWS)
            if len(counts) > MADCORE_ROWS:
                err(f"voronoi_a level-24 value >= 2^24 in rows {start}:{stop}")
                counts = counts[:MADCORE_ROWS]
            hist += counts
            level_max = np.maximum(level_max, block.max(axis=0))
            level_min = np.minimum(level_min, block.min(axis=0))
            dist = np.asarray(vd[start:stop])
            partial["d_nan"] += int(np.isnan(dist).sum())
            partial["d_negative"] += int((dist < -1e-6).sum())
        partial["level_max"] = level_max.tolist()
        partial["level_min"] = level_min.tolist()
        nonzero = np.flatnonzero(hist)
        if len(nonzero) < 2_000_000:
            partial["hist"] = ("sparse", nonzero, hist[nonzero])
        else:
            partial["hist"] = ("dense", hist)

    return partial


def finalize_dir(light, partials, dir_path):
    result = light
    err = result["errors"].append

    for p in partials:
        result["errors"].extend(p.get("errors", []))

    result["max_abs_coordinate"] = max([p.get("max_abs", 0.0) for p in partials], default=0.0)
    result["rows_above_1e4"] = sum(p.get("far_rows", 0) for p in partials)
    if result["rows_above_1e4"] and dir_path != NO_CELL_OUTLIER_DIR:
        allowed = False
        group_path = RELEASE / dir_path / "dataset_group.npy"
        far_list = [r for p in partials for r in p.get("far_row_list", [])]
        if group_path.exists() and len(far_list) == result["rows_above_1e4"]:
            group = np.load(group_path, mmap_mode="r")
            values = {bytes(group[r]) for r in far_list}
            allowed = values <= {NO_CELL_OUTLIER_DIR.split("/", 1)[1].encode()}
            if allowed:
                result["notes"].append(
                    f"{len(far_list)} far rows are documented vv6sy no-cell molecules: {far_list}")
        if not allowed:
            err(f"{result['rows_above_1e4']} rows with |coordinate| > 1e4 "
                f"(max {result['max_abs_coordinate']:.3g})")

    empty_ids = sum(p.get("empty_ids", 0) for p in partials)
    if empty_ids:
        err(f"{empty_ids} empty configuration_ids")

    pbc_true_zero = sum(p.get("pbc_true_zero", 0) for p in partials)
    if pbc_true_zero:
        err(f"{pbc_true_zero} pbc-true axes with zero cell row")
    result["pbc_false_nonzero_cell_axes"] = sum(p.get("pbc_false_nonzero", 0) for p in partials)

    result["dataset_ids"] = sorted(set().union(*[set(p.get("dataset_ids", [])) for p in partials]))

    with_voronoi = [p for p in partials if p.get("level_max") is not None]
    if with_voronoi:
        level_max = np.max([p["level_max"] for p in with_voronoi], axis=0)
        level_min = np.min([p["level_min"] for p in with_voronoi], axis=0)
        limits = 2 ** np.arange(25, dtype=np.int64)
        if (level_max >= limits).any() or (level_min < 0).any():
            err("voronoi_a level k outside [0, 2^k)")
        d_negative = sum(p["d_negative"] for p in with_voronoi)
        if d_negative:
            err(f"{d_negative} voronoi_d values below -1e-6")
        result["voronoi_d_nan"] = sum(p["d_nan"] for p in with_voronoi)

    return result


def plan_tasks(dirs):
    tasks = []
    for dir_path in dirs:
        with open(RELEASE / dir_path / "MANIFEST.json") as f:
            manifest = json.load(f)
        ns = manifest["ns"]
        tasks.append(("light", dir_path, 0, 0))

        weight = manifest["n_atoms"] * 12 + ns * 200
        nparts = min(64, max(1, int(weight // SPLIT_WEIGHT) + 1))
        if nparts == 1:
            tasks.append(("range", dir_path, 0, ns))
            continue

        na = np.load(RELEASE / dir_path / "na.npy")
        targets = np.linspace(0, na[-1], nparts + 1)
        cuts = np.unique(np.searchsorted(np.asarray(na), targets))
        cuts[0], cuts[-1] = 0, ns
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi > lo:
                tasks.append(("range", dir_path, int(lo), int(hi)))
    return tasks


def scan_task(task):
    kind, dir_path, lo, hi = task
    try:
        if kind == "light":
            return task, light_checks(dir_path)
        return task, range_scan(dir_path, lo, hi)
    except Exception as exc:
        return task, {"dir": dir_path, "exception": True,
                      "errors": [f"exception in {kind} [{lo}:{hi}]: {exc!r}"]}


def run_scan(dirs, workers):
    tasks = plan_tasks(dirs)
    print(f"{len(dirs)} dirs -> {len(tasks)} tasks", flush=True)

    hist = np.zeros(MADCORE_ROWS, dtype=np.int64)
    lights = {}
    partials = {}
    done = 0
    with Pool(min(workers, len(tasks))) as pool:
        for task, out in pool.imap_unordered(scan_task, tasks):
            kind, dir_path = task[0], task[1]
            if kind == "light" and not out.get("exception"):
                lights[dir_path] = out
            else:
                packed = out.pop("hist", None)
                if packed is not None:
                    if packed[0] == "sparse":
                        np.add.at(hist, packed[1], packed[2])
                    else:
                        hist += packed[1]
                partials.setdefault(dir_path, []).append(out)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tasks done", flush=True)

    results = []
    for dir_path in dirs:
        light = lights.get(dir_path)
        if light is None:
            light = {"dir": dir_path, "errors": ["light checks missing (exception)"],
                     "ns": 0, "n_atoms": 0}
        results.append(finalize_dir(light, partials.get(dir_path, []), dir_path))
    return results, hist


def all_scan_dirs():
    return [f"collection/{p.name}" for p in collection_dirs()] + ["mad-next-coreset", "mad-core"]


def plan_shards(nshards):
    weights = []
    for dir_path in all_scan_dirs():
        with open(RELEASE / dir_path / "MANIFEST.json") as f:
            manifest = json.load(f)
        weights.append((manifest["n_atoms"] * 12 + manifest["ns"] * 200, dir_path))

    shards = [[] for _ in range(nshards)]
    loads = [0] * nshards
    for weight, dir_path in sorted(weights, reverse=True):
        lightest = loads.index(min(loads))
        shards[lightest].append(dir_path)
        loads[lightest] += weight
    return shards, loads


def part_scan_shard(shard, nshards, workers):
    shards, loads = plan_shards(nshards)
    dirs = shards[shard]
    print(f"shard {shard}/{nshards}: {len(dirs)} dirs, ~{loads[shard] / 1e9:.0f} GB weight",
          flush=True)

    results, hist = run_scan(dirs, workers)

    os.makedirs(REPO_OUT, exist_ok=True)
    with open(REPO_OUT / f"scan_shard_{shard}.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(REPO_OUT / f"scan_hist_{shard}.npy", "wb") as f:
        np.save(f, hist)

    bad = {r["dir"]: r["errors"] for r in results if r.get("errors")}
    check(not bad, f"shard {shard}: per-directory checks clean ({len(bad)} dirs with errors)")
    for dir_path, errors in sorted(bad.items()):
        for error in errors:
            print(f"    {dir_path}: {error}", flush=True)
    return {"shard": shard, "dirs": len(dirs)}


def part_scan_merge(nshards):
    results = []
    hist = np.zeros(MADCORE_ROWS, dtype=np.int64)
    for shard in range(nshards):
        with open(REPO_OUT / f"scan_shard_{shard}.json") as f:
            results.extend(json.load(f))
        hist += np.load(REPO_OUT / f"scan_hist_{shard}.npy")

    scanned = {r["dir"] for r in results}
    expected = set(all_scan_dirs())
    check(scanned == expected,
          f"shards cover all 456 directories exactly once "
          f"(missing {sorted(expected - scanned)[:3]}, extra {sorted(scanned - expected)[:3]})")

    return global_scan_checks(results, hist)


def part_scan(workers):
    dirs = all_scan_dirs()
    check(len(dirs) == 456, f"456 directories to scan (found {len(dirs)})")
    results, hist = run_scan(dirs, workers)
    return global_scan_checks(results, hist)


def global_scan_checks(results, hist):
    by_dir = {r["dir"]: r for r in results}
    bad = {r["dir"]: r["errors"] for r in results if r.get("errors")}
    check(not bad, f"per-directory checks clean ({len(bad)} dirs with errors)")
    for dir_path, errors in sorted(bad.items()):
        for error in errors:
            print(f"    {dir_path}: {error}", flush=True)

    total_ns = sum(r["ns"] for r in results if r["dir"].startswith("collection/"))
    check(total_ns == COLLECTION_ROWS, f"collection rows {total_ns:,} == {COLLECTION_ROWS:,}")

    clean = sum(r.get("clean_rows", 0) for r in results)
    check(clean == CLEAN_COLLECTION_ROWS,
          f"collection clean rows {clean:,} == {CLEAN_COLLECTION_ROWS:,}")

    population = np.load(RELEASE / "mad-core" / "voronoi_population.npy")
    check(np.array_equal(hist, population),
          "voronoi_a level-24 histogram over all members == voronoi_population (closure)")

    metadata_ids = set()
    with open(RELEASE / "dataset_metadata.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        id_col = header.index("dataset_id")
        for line in f:
            metadata_ids.add(line.rstrip("\n").split("\t")[id_col])
    seen_ids = set()
    for r in results:
        if r["dir"].startswith("collection/"):
            seen_ids.update(r.get("dataset_ids", []))
    uncovered = seen_ids - metadata_ids
    check(not uncovered, f"all collection dataset_ids covered by dataset_metadata.tsv "
                         f"({len(seen_ids)} seen, uncovered: {sorted(uncovered)[:5]})")

    vv6sy = by_dir.get(NO_CELL_OUTLIER_DIR, {})
    print(f"documented outlier dir {NO_CELL_OUTLIER_DIR}: "
          f"{vv6sy.get('rows_above_1e4')} rows above 1e4, "
          f"max {vv6sy.get('max_abs_coordinate'):.4g}", flush=True)

    return {"directories": results,
            "closure_ok": bool(np.array_equal(hist, population)),
            "collection_rows": total_ns,
            "clean_rows": clean}


def part_coreset():
    d = RELEASE / "mad-core"
    ns, na = load_geometry(d)
    check(ns == MADCORE_ROWS, f"mad-core ns == {MADCORE_ROWS:,}")

    fps = np.load(d / "fps_order.npy")
    check(np.array_equal(fps, np.arange(ns)), "fps_order == 0..ns-1 (identity permutation)")

    kset = np.load(d / "k_set.npy")
    counts = np.bincount(kset)
    cumulative = np.cumsum(counts)
    levels = np.flatnonzero(counts)
    nested_ok = all(cumulative[k] == 2**k for k in levels if 2**k <= ns)
    check(nested_ok, f"k_set nesting: cumulative counts == 2^k for levels {levels.tolist()}")

    eps = np.load(d / "epsilon_sq.npy")
    tail = eps[ACCRETE_ROWS:]
    diffs = np.diff(tail)
    rise_rows = (ACCRETE_ROWS + np.flatnonzero(diffs > 0)).tolist()
    beyond = int((diffs > 1e-3 * np.abs(tail[:-1]) + 2e-6).sum())
    check(beyond == 0,
          f"epsilon_sq non-increasing beyond accrete block within resume-boundary tolerance "
          f"(sub-tolerance resume steps at rows {rise_rows})")
    block_rises = int((np.diff(eps[:ACCRETE_ROWS]) > 0).sum())
    print(f"  epsilon_sq rises inside accrete block [0, {ACCRETE_ROWS}): {block_rises} "
          f"(accrete order, informational)", flush=True)

    population = np.load(d / "voronoi_population.npy")
    check(int(population.sum()) == RELEASE_ROWS and population.min() >= 1,
          f"voronoi_population sums to {RELEASE_ROWS:,}, min >= 1")

    for name, dtype, shape in [("ecumetric_energy.bin", np.float32, (-1, 1)),
                               ("ecumetric_nc_forces.bin", np.float32, (-1, 3))]:
        arr = np.memmap(d / name, dtype=dtype, mode="r").reshape(shape)
        nan_count = 0
        for start in range(0, len(arr), CHUNK_ROWS * 4):
            nan_count += int(np.isnan(arr[start:start + CHUNK_ROWS * 4]).sum())
        check(nan_count == 0, f"{name} finite everywhere")

    features = np.memmap(d / "ecumetric_features.bin", dtype=np.float32, mode="r").reshape(ns, 512)
    nan_count = 0
    for start in range(0, ns, CHUNK_ROWS // 4):
        nan_count += int(np.isnan(features[start:start + CHUNK_ROWS // 4]).sum())
    check(nan_count == 0, "ecumetric_features.bin finite everywhere")

    for name in ("zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy"):
        arr = np.load(d / f"{name}.npy")
        check(not np.isnan(arr).any(), f"{name}.npy has no NaN")

    group = np.load(d / "dataset_group.npy")
    families = {p.name.encode() for p in collection_dirs()}
    group_values = set(np.unique(group).tolist())
    stray = group_values - families - {b"mad-next"}
    check(not stray, f"dataset_group values are collection dirs + mad-next (stray: {stray})")
    check(int((group == b"mad-next").sum()) == 340_555, "340,555 mad-next rows in mad-core")

    cid = np.load(d / "configuration_id.npy", mmap_mode="r")
    mn_cid = np.load(RELEASE / "mad-next-coreset" / "configuration_id.npy")
    mn_row = {v: i for i, v in enumerate(mn_cid.tolist())}
    overlap_rows = np.flatnonzero(group == b"mad-next")

    va = np.memmap(RELEASE / "mad-next-coreset" / "voronoi_a.bin",
                   dtype=np.int32, mode="r").reshape(MADNEXT_ROWS, 25)
    vd = np.memmap(RELEASE / "mad-next-coreset" / "voronoi_d.bin",
                   dtype=np.float32, mode="r").reshape(MADNEXT_ROWS, 25)
    missing = 0
    self_index_bad = 0
    self_distance_bad = 0
    for r in overlap_rows.tolist():
        j = mn_row.get(bytes(cid[r]))
        if j is None:
            missing += 1
        else:
            if int(va[j, 24]) != r:
                self_index_bad += 1
            if float(vd[j, 24]) > 1e-2:
                self_distance_bad += 1
    check(missing == 0, f"all 340,555 overlap ids present in mad-next-coreset (missing {missing})")
    check(self_index_bad == 0,
          f"overlap rows' mad-next voronoi_a[24] == own mad-core row (bad {self_index_bad})")
    check(self_distance_bad == 0,
          f"overlap rows' mad-next voronoi_d[24] ~ 0 (above 1e-2 noise ceiling: {self_distance_bad})")

    off = np.load(d / "source_reference_offsets.npy")
    blob = np.memmap(d / "source_reference_bytes.bin", dtype=np.uint8, mode="r")
    check(len(off) == ns + 1 and off[0] == 0 and int(off[-1]) == len(blob)
          and np.all(np.diff(off) > 0), "source_reference offsets monotone, no empty reference")

    if (d / "release_directory.npy").exists():
        rel_dir = np.load(d / "release_directory.npy")
        rel_row = np.load(d / "release_row_in_directory.npy")
        names = np.load(d / "release_directory_names.npy")
        check(rel_dir.dtype == np.int16 and rel_row.dtype == np.int32
              and len(rel_dir) == ns == len(rel_row) and len(names) == 455,
              "pointer files present with expected dtypes and lengths")
        rng = np.random.default_rng(2)
        sample = rng.integers(0, ns, 2000)
        bad = 0
        for r in sample.tolist():
            dir_cid = np.load(RELEASE / names[rel_dir[r]].decode() / "configuration_id.npy",
                              mmap_mode="r")
            if dir_cid[rel_row[r]] != cid[r]:
                bad += 1
        check(bad == 0, f"2000 sampled pointer round-trips resolve (mismatches {bad})")
    else:
        print("  pointer files not present yet (staged pending copy-in)", flush=True)

    return {"k_set_levels": levels.tolist(),
            "fps_resume_rise_rows": rise_rows,
            "accrete_block_eps_rises": block_rises,
            "overlap_checked": len(overlap_rows)}


def part_crosstier(samples, seed):
    d = RELEASE / "mad-core"
    cell_fps = d / "cell_fps"
    offsets = np.load(cell_fps / "center_offsets.npy", mmap_mode="r")
    rel_dir = np.load(cell_fps / "release_directory.npy", mmap_mode="r")
    rel_row = np.load(cell_fps / "release_row_in_directory.npy", mmap_mode="r")
    names = np.load(cell_fps / "release_directory_names.npy")
    cid = np.load(d / "configuration_id.npy", mmap_mode="r")

    rng = np.random.default_rng(seed)
    rows = set(rng.integers(0, MADCORE_ROWS, samples).tolist())
    rows.update(int(r) for r in np.load(d / "image_flag_rows.npy"))
    rows = sorted(rows)

    by_dir = {}
    for r in rows:
        member = int(offsets[r])
        by_dir.setdefault(int(rel_dir[member]), []).append((r, int(rel_row[member])))

    mismatches = 0
    far_self_bad = 0
    flagged = set(int(r) for r in np.load(d / "image_flag_rows.npy"))
    for dir_index, pairs in by_dir.items():
        dir_path = RELEASE / names[dir_index].decode()
        dir_cid = np.load(dir_path / "configuration_id.npy", mmap_mode="r")
        ns_dir = len(dir_cid)
        va = np.memmap(dir_path / "voronoi_a.bin", dtype=np.int32, mode="r").reshape(ns_dir, 25)
        vd = np.memmap(dir_path / "voronoi_d.bin", dtype=np.float32, mode="r").reshape(ns_dir, 25)
        for r, row in pairs:
            if dir_cid[row].tobytes().rstrip(b"\x00") != cid[r].tobytes().rstrip(b"\x00"):
                mismatches += 1
            elif r in flagged:
                if int(va[row, 24]) != r or float(vd[row, 24]) > 1e-2:
                    far_self_bad += 1

    check(mismatches == 0,
          f"{len(rows)} sampled mad-core ids resolve to identical ids in their release dir "
          f"(mismatches {mismatches})")
    check(far_self_bad == 0,
          f"all {len(flagged)} folded centers self-assigned with ~zero distance "
          f"(bad {far_self_bad})")
    return {"sampled_centers": len(rows)}


def part_madnext():
    d = RELEASE / "mad-next-coreset"
    ns, na = load_geometry(d)
    check(ns == MADNEXT_ROWS, f"mad-next-coreset ns == {MADNEXT_ROWS:,}")

    split = np.load(d / "split.npy")
    values, counts = np.unique(split, return_counts=True)
    observed = dict(zip(values.tolist(), counts.tolist()))
    check(observed == SPLIT_COUNTS, f"split counts {observed} == documented")

    fps = np.load(d / "fps_order.npy")
    check(np.array_equal(fps, np.arange(ns)), "fps_order == 0..ns-1")

    eps = np.load(d / "epsilon_sq.npy")
    rises = int((np.diff(eps) > 0).sum())
    print(f"  mad-next epsilon_sq rises along rows: {rises} (informational)", flush=True)

    kset = np.load(d / "k_set.npy")
    counts = np.bincount(kset)
    cumulative = np.cumsum(counts)
    levels = np.flatnonzero(counts)
    nested_ok = all(cumulative[k] == 2**k for k in levels if 2**k <= ns)
    check(nested_ok, f"k_set nesting for levels {levels.tolist()}")

    check(not (d / "image_flag_rows.npy").exists(), "no image flag files (no displaced rows)")
    check(not (d / "source_reference_offsets.npy").exists(), "no source_reference (documented)")

    group = np.load(d / "dataset_group.npy")
    values = set(np.unique(group).tolist())
    check(values == {b"mad15", b"mad_next", b"mad_sol"}, f"dataset_group values {values}")

    return {"split_counts": {k.decode() or "(empty)": v for k, v in observed.items()},
            "eps_rises": rises}


FRAME_RANGE = re.compile(r"frames_r(\d+)-(\d+)\.extxyz$")


def walk_extxyz_frames(path):
    count = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line)
            f.readline()
            for _ in range(n):
                f.readline()
            count += 1
    return count


def parse_frame(path, frame_index):
    with open(path, "rb") as f:
        for _ in range(frame_index):
            n = int(f.readline())
            f.readline()
            for _ in range(n):
                f.readline()
        n = int(f.readline())
        header = f.readline().decode()
        positions = np.array(
            [f.readline().split()[1:4] for _ in range(n)], dtype=np.float64)
    fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', header))
    return n, fields, positions


def check_extxyz_chunk(args):
    chunk_path, start_row, tier_dir, sample_count, seed = args
    d = RELEASE / tier_dir
    ns, na = load_geometry(d)
    x = np.memmap(d / "x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
    cid = np.load(d / "configuration_id.npy", mmap_mode="r")
    group = (np.load(d / "dataset_group.npy", mmap_mode="r")
             if (d / "dataset_group.npy").exists() else None)

    errors = []
    frames = walk_extxyz_frames(chunk_path)

    rng = np.random.default_rng(seed)
    for frame in sorted(rng.integers(0, frames, sample_count).tolist()):
        row = start_row + frame
        n, fields, positions = parse_frame(chunk_path, frame)
        expected_n = int(na[row + 1] - na[row])
        if n != expected_n:
            errors.append(f"{chunk_path.name} frame {frame}: {n} atoms != na {expected_n}")
            continue
        stored = x[na[row]:na[row + 1]].astype(np.float64)
        if np.abs(positions - stored).max() > 1e-6:
            errors.append(f"{chunk_path.name} frame {frame}: positions deviate "
                          f"{np.abs(positions - stored).max():.3g}")
        if fields.get("configuration_id") != cid[row].decode():
            errors.append(f"{chunk_path.name} frame {frame}: configuration_id mismatch")
        if fields.get("fps_order") != str(row):
            errors.append(f"{chunk_path.name} frame {frame}: fps_order "
                          f"{fields.get('fps_order')} != {row}")
        if group is not None and fields.get("dataset_group") != group[row].decode():
            errors.append(f"{chunk_path.name} frame {frame}: dataset_group "
                          f"{fields.get('dataset_group')} != {group[row].decode()}")
    return chunk_path.name, frames, errors


def part_extxyz(workers, seed):
    chunks = sorted((RELEASE / "mad-core" / "extxyz").glob("frames_r*.extxyz"))
    check(len(chunks) == 256, f"256 mad-core extxyz chunks (found {len(chunks)})")

    jobs = []
    for chunk in chunks:
        match = FRAME_RANGE.search(chunk.name)
        start_row, end_row = int(match.group(1)), int(match.group(2))
        jobs.append((chunk, start_row, "mad-core", 4, seed + start_row))

    with Pool(workers) as pool:
        outputs = pool.map(check_extxyz_chunk, jobs)

    frame_total = sum(frames for _, frames, _ in outputs)
    errors = [e for _, _, errs in outputs for e in errs]
    bad_counts = [(name, frames) for name, frames, _ in outputs if frames != 65_536]
    check(not bad_counts, f"every chunk holds 65,536 frames (bad: {bad_counts[:3]})")
    check(frame_total == MADCORE_ROWS, f"chunk frames total {frame_total:,} == {MADCORE_ROWS:,}")
    check(not errors, f"sampled frame contents match binaries ({len(errors)} errors)")
    for e in errors[:10]:
        print(f"    {e}", flush=True)

    single = RELEASE / "mad-next-coreset" / "extxyz" / "mad-next-coreset.extxyz"
    name, frames, errors = check_extxyz_chunk((single, 0, "mad-next-coreset", 32, seed))
    check(frames == MADNEXT_ROWS, f"mad-next-coreset.extxyz frames {frames:,} == {MADNEXT_ROWS:,}")
    check(not errors, f"mad-next extxyz sampled frames match binaries ({len(errors)} errors)")

    return {"madcore_frames": frame_total, "madnext_frames": frames}


def run_parts(parts, args):
    os.makedirs(AUDIT_OUT, exist_ok=True)
    os.makedirs(REPO_OUT, exist_ok=True)
    summary = {}

    for part in parts:
        print(f"=== part: {part} ===", flush=True)
        if part == "coreset":
            summary[part] = part_coreset()
        elif part == "madnext":
            summary[part] = part_madnext()
        elif part == "crosstier":
            summary[part] = part_crosstier(args.samples, args.seed)
        elif part == "extxyz":
            summary[part] = part_extxyz(args.workers, args.seed)
        elif part == "scan":
            if args.shard is not None:
                summary[part] = part_scan_shard(args.shard, args.nshards, args.workers)
            else:
                summary[part] = part_scan(args.workers)
        elif part == "scan-merge":
            summary[part] = part_scan_merge(args.nshards)

    summary["failures"] = failures
    label = "_".join(parts)
    if args.shard is not None:
        label += f"_shard{args.shard}"
    for out_dir in (AUDIT_OUT, REPO_OUT):
        with open(out_dir / f"{label}.json", "w") as f:
            json.dump(summary, f, indent=1, default=str)

    if failures:
        print(f"VALIDATION FAILED: {len(failures)} checks failed", flush=True)
        for message in failures:
            print(f"  FAIL {message}", flush=True)
        sys.exit(1)
    print("VALIDATION PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="all",
                        choices=["all", "light", "coreset", "madnext", "crosstier",
                                 "extxyz", "scan", "scan-merge"])
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--nshards", type=int, default=6)
    args = parser.parse_args()

    if args.part == "all":
        parts = ["coreset", "madnext", "crosstier", "extxyz", "scan"]
    elif args.part == "light":
        parts = ["coreset", "madnext", "crosstier"]
    else:
        parts = [args.part]
    run_parts(parts, args)


if __name__ == "__main__":
    main()
