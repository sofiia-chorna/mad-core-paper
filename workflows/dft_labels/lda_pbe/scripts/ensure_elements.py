import os
import numpy as np
from pathlib import Path

from mad.madcore_vs_random_20k.check_elements import TARGET_Z, elements_of_dir
from mad.madcore_vs_random_20k.xyz_write import SYMBOLS, classify, write_frame

RELEASE = Path("${STORE_ROOT}/mad-core/release")
BASE = Path(__file__).resolve().parent
OUT = BASE / "out" / "random_25k"
EXCLUDE_TSVS = ["out/madcore_25k/selection.tsv", "out/random_test_6k/selection.tsv",
                "out/voronoi_25k/selection.tsv"]
CANDIDATE_DIRS = [
    "collection/nomad",
    "collection/Carolina_Materials",
    "collection/N_O_F_columns_non-bonded_vdW_potential_JCP2023",
    "collection/mcloudarchive__55jga-d0447",
    "collection/mcloudarchive__m0-zg",
    "collection/mcloudarchive__nz8sk-6h010",
    "collection/mcloudarchive__pkhy1-ted45",
    "mad-next-coreset",
]
MAX_ATOMS = 300
SEED = 20260815


def tsv_ids(path):
    with open(path) as fh:
        header = next(fh).rstrip("\n").split("\t")
        col = header.index("configuration_id")
        return {line.rstrip("\n").split("\t")[col] for line in fh}


def candidate_rows(missing):
    candidates = {z: [] for z in missing}
    missing_arr = np.array(sorted(missing))

    for rel in CANDIDATE_DIRS:
        g = RELEASE / rel
        na = np.load(g / "na.npy")
        a = np.asarray(np.memmap(g / "a.bin", dtype=np.int32, mode="r"))
        natoms = np.diff(na)

        is_mad = rel == "mad-next-coreset"
        if is_mad:
            clean = np.ones(len(natoms), dtype=bool)
        else:
            fc = np.load(g / "filter_code.npy")
            dup = np.load(g / "is_duplicate.npy")
            clean = (fc == 0) & (~dup) & (natoms < MAX_ATOMS)

        hits = np.flatnonzero(np.isin(a, missing_arr))
        rows = np.searchsorted(na, hits, side="right") - 1
        for atom_idx, row in zip(hits, rows):
            if clean[row]:
                candidates[int(a[atom_idx])].append((rel, int(row)))

    return {z: sorted(set(rows)) for z, rows in candidates.items()}


def fetch_frame(rel, row):
    g = RELEASE / rel
    na = np.load(g / "na.npy", mmap_mode="r")
    a = np.memmap(g / "a.bin", dtype=np.int32, mode="r")
    x = np.memmap(g / "x.bin", dtype=np.float32, mode="r").reshape(-1, 3)
    c = np.memmap(g / "c.bin", dtype=np.float32, mode="r").reshape(-1, 3, 3)
    pbc = np.load(g / "pbc.npy", mmap_mode="r")
    dataset_id = np.load(g / "dataset_id.npy", mmap_mode="r")
    cid = np.load(g / "configuration_id.npy", mmap_mode="r")

    return {
        "group": rel.split("/")[-1],
        "row": row,
        "cid": cid[row].decode(),
        "dataset_id": dataset_id[row].decode(),
        "numbers": np.asarray(a[na[row]: na[row + 1]]),
        "positions": np.asarray(x[na[row]: na[row + 1]]),
        "cell": np.asarray(c[row]),
        "pbc": np.asarray(pbc[row]),
    }


def read_frames(out_dir):
    frames = []
    for name in ("ttt", "fff", "surfaces"):
        with open(out_dir / f"{name}.extxyz") as fh:
            while True:
                count_line = fh.readline()
                if not count_line:
                    break
                n = int(count_line)
                comment = fh.readline()
                atom_lines = [fh.readline() for _ in range(n)]
                cid = comment.split('configuration_id="')[1].split('"')[0]
                zs = {line.split()[0] for line in atom_lines}
                frames.append((cid, name, count_line + comment + "".join(atom_lines), zs))
    return frames


def main():
    present = elements_of_dir(OUT)
    missing = sorted(TARGET_Z - present)
    if not missing:
        print("all 102 elements already present")
        return

    print("missing:", [SYMBOLS[z] for z in missing], flush=True)

    excluded = set()
    for path in EXCLUDE_TSVS:
        if (BASE / path).exists():
            excluded |= tsv_ids(BASE / path)
    excluded |= tsv_ids(OUT / "selection.tsv")

    rng = np.random.default_rng(SEED)
    candidates = candidate_rows(set(missing))

    additions = []
    covered = set(present)

    for z in sorted(missing, key=lambda z: len(candidates[z])):
        if z in covered:
            continue

        pool = candidates[z]
        chosen = None
        for i in rng.permutation(len(pool)):
            rel, row = pool[i]
            frame = fetch_frame(rel, row)
            if frame["cid"] not in excluded:
                chosen = frame
                break

        assert chosen is not None, f"no eligible candidate for Z={z}"
        excluded.add(chosen["cid"])
        covered |= set(chosen["numbers"].tolist())
        additions.append((z, chosen))
        print(f"add {SYMBOLS[z]}: {chosen['cid']} ({chosen['group']}, row {chosen['row']}, "
              f"{len(chosen['numbers'])} atoms)", flush=True)

    frames = read_frames(OUT)

    symbol_counts = {}
    for _, _, _, zs in frames:
        for s in zs:
            symbol_counts[s] = symbol_counts.get(s, 0) + 1
    for _, frame in additions:
        for s in {SYMBOLS[z] for z in np.unique(frame["numbers"])}:
            symbol_counts[s] = symbol_counts.get(s, 0) + 1

    victims = set()
    order = rng.permutation(len(frames))
    for idx in order:
        if len(victims) >= len(additions):
            break
        cid, _, _, zs = frames[idx]
        if all(symbol_counts[s] >= 2 for s in zs):
            victims.add(cid)
            for s in zs:
                symbol_counts[s] -= 1

    assert len(victims) == len(additions)

    with open(OUT / "selection.tsv") as fh:
        header_line = next(fh)
        manifest_rows = [line for line in fh]

    files = {name: open(OUT / f"{name}.extxyz.new", "w") for name in ("ttt", "fff", "surfaces")}
    counts = {"ttt": 0, "fff": 0, "surfaces": 0}

    for cid, name, text, _ in frames:
        if cid in victims:
            continue
        files[name].write(text)
        counts[name] += 1

    kept_manifest = [row for row in manifest_rows if row.split("\t")[0] not in victims]

    topup_log = open(OUT / "element_topup.tsv", "w")
    topup_log.write("element\tadded_configuration_id\tadded_group\tremoved_configuration_id\n")

    for (z, frame), victim in zip(additions, sorted(victims)):
        split, vac = classify(frame["positions"], frame["cell"], frame["pbc"])
        counts[split] += 1

        info = {
            "configuration_id": frame["cid"],
            "dataset_group": frame["group"],
            "dataset_id": frame["dataset_id"],
        }
        write_frame(files[split], frame["numbers"], frame["positions"], frame["cell"],
                    frame["pbc"], info)

        pbc_str = "".join("T" if flag else "F" for flag in frame["pbc"])
        kept_manifest.append(f"{frame['cid']}\t{frame['group']}\t{frame['dataset_id']}\t"
                             f"{frame['row']}\t{len(frame['numbers'])}\t{pbc_str}\t{vac}\t{split}\n")
        topup_log.write(f"{SYMBOLS[z]}\t{frame['cid']}\t{frame['group']}\t{victim}\n")

    topup_log.close()

    with open(OUT / "selection.tsv.new", "w") as fh:
        fh.write(header_line)
        fh.writelines(kept_manifest)

    for name in files:
        files[name].close()
        os.replace(OUT / f"{name}.extxyz.new", OUT / f"{name}.extxyz")
    os.replace(OUT / "selection.tsv.new", OUT / "selection.tsv")

    total = sum(counts.values())
    print(f"swapped {len(additions)} structures; total {total}: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
