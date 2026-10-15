import multiprocessing as mp
import numpy as np
from pathlib import Path

from mad.madcore_vs_random_20k.xyz_write import classify, write_frame

RELEASE = Path("${STORE_ROOT}/mad-core/release")
MAD_TIER = "mad-next-coreset"
TOTAL_ROWS = 1694502355
MAX_ATOMS = 300
WORKERS = 24


def tsv_ids(path):
    with open(path) as fh:
        header = next(fh).rstrip("\n").split("\t")
        col = header.index("configuration_id")
        return {line.rstrip("\n").split("\t")[col] for line in fh}


def group_dirs():
    names = sorted("collection/" + d.name for d in (RELEASE / "collection").iterdir() if d.is_dir())
    names.append(MAD_TIER)
    return names


_w = {}


def _init_worker(names):
    _w["names"] = names
    _w["groups"] = {}


def _group(gi):
    if gi not in _w["groups"]:
        g = RELEASE / _w["names"][gi]
        is_mad = _w["names"][gi] == MAD_TIER
        _w["groups"][gi] = {
            "path": g,
            "is_mad": is_mad,
            "na": np.load(g / "na.npy", mmap_mode="r"),
            "cid": np.load(g / "configuration_id.npy", mmap_mode="r"),
            "fc": None if is_mad else np.load(g / "filter_code.npy", mmap_mode="r"),
            "dup": None if is_mad else np.load(g / "is_duplicate.npy", mmap_mode="r"),
            "geom": None,
        }
    return _w["groups"][gi]


def _geom(grp):
    if grp["geom"] is None:
        g = grp["path"]
        grp["geom"] = {
            "a": np.memmap(g / "a.bin", dtype=np.int32, mode="r"),
            "x": np.memmap(g / "x.bin", dtype=np.float32, mode="r").reshape(-1, 3),
            "c": np.memmap(g / "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3),
            "pbc": np.load(g / "pbc.npy", mmap_mode="r"),
            "dataset_id": np.load(g / "dataset_id.npy", mmap_mode="r"),
        }
    return grp["geom"]


def _check(task):
    gi, rows = task
    grp = _group(gi)
    na = grp["na"]
    natoms = np.asarray(na[rows + 1]) - np.asarray(na[rows])

    if grp["is_mad"]:
        ok = np.ones(len(rows), dtype=bool)
        rejects = {"filtered": 0, "duplicate": 0, "too_big": 0}
    else:
        ok_filter = np.asarray(grp["fc"][rows]) == 0
        ok_dup = ~np.asarray(grp["dup"][rows])
        ok_size = natoms < MAX_ATOMS
        ok = ok_filter & ok_dup & ok_size
        rejects = {
            "filtered": int((~ok_filter).sum()),
            "duplicate": int((ok_filter & ~ok_dup).sum()),
            "too_big": int((ok_filter & ok_dup & ~ok_size).sum()),
        }

    cids = [c.decode() for c in np.asarray(grp["cid"][rows[ok]])]
    return ok, natoms, cids, rejects


def _fetch(task):
    gi, row = task
    grp = _group(gi)
    geom = _geom(grp)
    na = grp["na"]

    return {
        "group": _w["names"][gi].split("/")[-1],
        "numbers": np.asarray(geom["a"][na[row]: na[row + 1]]),
        "positions": np.asarray(geom["x"][na[row]: na[row + 1]]),
        "cell": np.asarray(geom["c"][row]),
        "pbc": np.asarray(geom["pbc"][row]),
        "dataset_id": geom["dataset_id"][row].decode(),
    }


def run_sampler(out_name, target, seed, exclude_tsvs):
    base = Path(__file__).resolve().parent
    out = base / "out" / out_name
    out.mkdir(parents=True, exist_ok=True)

    names = group_dirs()
    ns = np.array([int(np.load(RELEASE / n / "ns.npy")[0]) for n in names])
    cum = np.cumsum(ns)
    assert cum[-1] == TOTAL_ROWS, cum[-1]

    excluded = set()
    for path in exclude_tsvs:
        excluded |= tsv_ids(base / path)
    print(f"pool {TOTAL_ROWS} rows in {len(names)} groups; excluding {len(excluded)} ids", flush=True)

    rng = np.random.default_rng(seed)
    pool = mp.Pool(WORKERS, _init_worker, (names,))

    accepted = []
    seen_rows = set()
    seen_ids = set(excluded)
    rejected = {"filtered": 0, "duplicate": 0, "too_big": 0, "collision": 0}

    while len(accepted) < target:
        draws = rng.integers(0, TOTAL_ROWS, size=(target - len(accepted)) * 2 + 1000)
        draw_gi = np.searchsorted(cum, draws, side="right")
        draw_row = draws - (cum[draw_gi] - ns[draw_gi])

        sels, tasks = [], []
        for gi in np.unique(draw_gi):
            sel = np.flatnonzero(draw_gi == gi)
            sels.append(sel)
            tasks.append((int(gi), draw_row[sel]))

        ok_all = np.zeros(len(draws), dtype=bool)
        nat_all = np.zeros(len(draws), dtype=np.int64)
        cid_all = np.empty(len(draws), dtype=object)

        for sel, (ok, natoms, cids, rejects) in zip(sels, pool.map(_check, tasks)):
            ok_all[sel] = ok
            nat_all[sel] = natoms
            cid_all[sel[ok]] = cids
            for key in rejects:
                rejected[key] += rejects[key]

        for i in np.flatnonzero(ok_all):
            if len(accepted) >= target:
                break

            key = (int(draw_gi[i]), int(draw_row[i]))
            if key in seen_rows or cid_all[i] in seen_ids:
                rejected["collision"] += 1
                continue

            seen_rows.add(key)
            seen_ids.add(cid_all[i])
            accepted.append((key[0], key[1], cid_all[i], int(nat_all[i])))

        print(f"accepted {len(accepted)}/{target}  rejected {rejected}", flush=True)

    frames = pool.map(_fetch, [(gi, row) for gi, row, _, _ in accepted], chunksize=8)
    pool.close()
    pool.join()

    order = rng.permutation(target)

    files = {name: open(out / f"{name}.extxyz", "w") for name in ("ttt", "fff", "surfaces")}
    manifest = open(out / "selection.tsv", "w")
    manifest.write("configuration_id\tdataset_group\tdataset_id\trow_in_group\tnatoms\tpbc\tvacuum_axes\tsplit_file\n")

    counts = {"ttt": 0, "fff": 0, "surfaces": 0}
    n_mad = 0

    for idx in order:
        gi, row, cid_str, natoms = accepted[idx]
        f = frames[idx]
        n_mad += f["group"] == MAD_TIER

        split, vac = classify(f["positions"], f["cell"], f["pbc"])
        counts[split] += 1

        info = {
            "configuration_id": cid_str,
            "dataset_group": f["group"],
            "dataset_id": f["dataset_id"],
        }
        write_frame(files[split], f["numbers"], f["positions"], f["cell"], f["pbc"], info)

        pbc_str = "".join("T" if flag else "F" for flag in f["pbc"])
        manifest.write(f"{cid_str}\t{f['group']}\t{f['dataset_id']}\t{row}\t"
                       f"{natoms}\t{pbc_str}\t{vac}\t{split}\n")

    for fh in files.values():
        fh.close()
    manifest.close()

    print(f"mad-tier rows sampled: {n_mad}")
    print(f"written to {out}: " + ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)
