import multiprocessing as mp
import numpy as np
from pathlib import Path

from mad.madcore_vs_random_20k.xyz_write import classify, write_frame

RELEASE = Path("${STORE_ROOT}/mad-core/release")
CELL_FPS = RELEASE / "mad-core" / "cell_fps"
BASE = Path(__file__).resolve().parent
OUT = BASE / "out" / "voronoi_25k"
EXCLUDE_TSVS = [BASE / "out" / "random_25k" / "selection.tsv",
                BASE / "out" / "random_test_6k" / "selection.tsv"]
TARGET = 25000
ALPHA = 0.25
MAX_ATOMS = 300
SEED = 20260813
CELL_BUFFER = 2 * TARGET
MAX_MEMBERS_SCANNED = 256
WORKERS = 24


def tsv_ids(path):
    with open(path) as fh:
        header = next(fh).rstrip("\n").split("\t")
        col = header.index("configuration_id")
        return {line.rstrip("\n").split("\t")[col] for line in fh}


_w = {}


def _init_worker(excluded):
    _w["excluded"] = excluded
    _w["offsets"] = np.load(CELL_FPS / "center_offsets.npy", mmap_mode="r")
    _w["order"] = np.load(CELL_FPS / "fps_order_in_cell.npy", mmap_mode="r")
    _w["dir"] = np.load(CELL_FPS / "release_directory.npy", mmap_mode="r")
    _w["rowdir"] = np.load(CELL_FPS / "release_row_in_directory.npy", mmap_mode="r")
    _w["cid_off"] = np.load(CELL_FPS / "configuration_id_offsets.npy", mmap_mode="r")
    _w["cid_blob"] = np.memmap(CELL_FPS / "configuration_id_bytes.bin", dtype=np.uint8, mode="r")
    _w["dirnames"] = [name.decode() for name in np.load(CELL_FPS / "release_directory_names.npy")]
    _w["folders"] = {}


def _folder(di):
    if di not in _w["folders"]:
        g = RELEASE / _w["dirnames"][di]
        _w["folders"][di] = {
            "na": np.load(g / "na.npy", mmap_mode="r"),
            "a": np.memmap(g / "a.bin", dtype=np.int32, mode="r"),
            "x": np.memmap(g / "x.bin", dtype=np.float32, mode="r").reshape(-1, 3),
            "c": np.memmap(g / "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3),
            "pbc": np.load(g / "pbc.npy", mmap_mode="r"),
            "dataset_id": np.load(g / "dataset_id.npy", mmap_mode="r"),
        }
    return _w["folders"][di]


def scan_cell(cell):
    lo, hi = int(_w["offsets"][cell]), int(_w["offsets"][cell + 1])
    scan_hi = min(hi, lo + MAX_MEMBERS_SCANNED)
    member_order = np.asarray(_w["order"][lo:scan_hi])

    for j in range(scan_hi - lo):
        if member_order[j] < 0:
            break

        di = int(_w["dir"][lo + j])
        row = int(_w["rowdir"][lo + j])
        na = _folder(di)["na"]
        natoms = int(na[row + 1] - na[row])
        is_mad = _w["dirnames"][di] == "mad-next-coreset"

        if not is_mad and natoms >= MAX_ATOMS:
            continue

        cid_str = bytes(_w["cid_blob"][_w["cid_off"][lo + j]: _w["cid_off"][lo + j + 1]]).decode()
        if cid_str in _w["excluded"]:
            continue

        return (int(cell), int(member_order[j]), di, row, cid_str, natoms)

    return None


def fetch_pick(pick):
    cell, rank, di, row, cid_str, natoms = pick
    fa = _folder(di)
    na = fa["na"]

    return {
        "cell": cell,
        "rank": rank,
        "folder": _w["dirnames"][di],
        "row": row,
        "cid": cid_str,
        "natoms": natoms,
        "numbers": np.asarray(fa["a"][na[row]: na[row + 1]]),
        "positions": np.asarray(fa["x"][na[row]: na[row + 1]]),
        "cell_matrix": np.asarray(fa["c"][row]),
        "pbc": np.asarray(fa["pbc"][row]),
        "dataset_id": fa["dataset_id"][row].decode(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    pop = np.load(RELEASE / "mad-core" / "voronoi_population.npy")
    rng = np.random.default_rng(SEED)

    gumbel = -np.log(-np.log(rng.random(len(pop))))
    scores = ALPHA * np.log(pop.astype(np.float64)) + gumbel
    ranked = np.argpartition(-scores, CELL_BUFFER)[:CELL_BUFFER]
    ranked = ranked[np.argsort(-scores[ranked])]

    excluded = set()
    for path in EXCLUDE_TSVS:
        excluded |= tsv_ids(path)
    print(f"excluding {len(excluded)} ids from prior selections", flush=True)

    pool = mp.Pool(WORKERS, _init_worker, (excluded,))

    candidates = []
    for i, result in enumerate(pool.imap(scan_cell, ranked.tolist(), chunksize=32)):
        candidates.append(result)
        if (i + 1) % 5000 == 0:
            print(f"scanned {i + 1}/{CELL_BUFFER} cells", flush=True)

    picks = []
    seen_ids = set()
    stats = {"cells_skipped": 0, "dup_across_cells": 0}

    for cand in candidates:
        if len(picks) >= TARGET:
            break
        if cand is None:
            stats["cells_skipped"] += 1
            continue
        if cand[4] in seen_ids:
            stats["dup_across_cells"] += 1
            continue
        seen_ids.add(cand[4])
        picks.append(cand)

    assert len(picks) == TARGET, f"only {len(picks)} picks from {CELL_BUFFER} cells"
    print(f"picked {TARGET}  {stats}", flush=True)
    print(f"centers chosen (member_rank 0): {sum(1 for p in picks if p[1] == 0)}", flush=True)

    frames = pool.map(fetch_pick, picks, chunksize=16)
    pool.close()
    pool.join()

    order = rng.permutation(TARGET)

    files = {name: open(OUT / f"{name}.extxyz", "w") for name in ("ttt", "fff", "surfaces")}
    manifest = open(OUT / "selection.tsv", "w")
    manifest.write("configuration_id\trelease_folder\tdataset_id\trow_in_folder\t"
                   "cell\tcell_population\tmember_rank\tnatoms\tpbc\tvacuum_axes\tsplit_file\n")

    counts = {"ttt": 0, "fff": 0, "surfaces": 0}

    for idx in order:
        f = frames[idx]

        split, vac = classify(f["positions"], f["cell_matrix"], f["pbc"])
        counts[split] += 1

        info = {
            "configuration_id": f["cid"],
            "dataset_group": f["folder"].split("/")[-1],
            "dataset_id": f["dataset_id"],
            "voronoi_cell": str(f["cell"]),
            "member_rank": str(f["rank"]),
        }
        write_frame(files[split], f["numbers"], f["positions"], f["cell_matrix"], f["pbc"], info)

        pbc_str = "".join("T" if flag else "F" for flag in f["pbc"])
        manifest.write(f"{f['cid']}\t{f['folder']}\t{f['dataset_id']}\t{f['row']}\t"
                       f"{f['cell']}\t{int(pop[f['cell']])}\t{f['rank']}\t{f['natoms']}\t"
                       f"{pbc_str}\t{vac}\t{split}\n")

    for fh in files.values():
        fh.close()
    manifest.close()

    print(f"written to {OUT}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
