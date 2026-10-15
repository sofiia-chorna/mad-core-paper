import json
from collections import Counter

import numpy as np

from paths import (BOTR_XYZ, DELTA_INDEX, LOGS, MAD16_R2SCAN, MANIFESTS, PBE_XYZ, make_logger, parse_frames, read_tsv, sha256, soft_hash, write_tsv)

SOURCES = {"mad16_r2scan_train": MAD16_R2SCAN["train"], "mad16_r2scan_val": MAD16_R2SCAN["val"], "mad16_r2scan_test": MAD16_R2SCAN["test"],
           "botr_r2scan": BOTR_XYZ}


def frame_at(path, offset):
    with open(path, "rb") as fh:
        fh.seek(offset)
        for frame in parse_frames(fh):
            return frame


def read_frame(path, offset):
    with open(path, "rb") as fh:
        fh.seek(offset)
        n = int(fh.readline())
        header = fh.readline().decode()
        lines = [fh.readline().split() for _ in range(n)]
    symbols = [l[0].decode() for l in lines]
    positions = np.array([[float(v) for v in l[1:4]] for l in lines])
    forces = np.array([[float(v) for v in l[4:7]] for l in lines])
    return header, symbols, positions, forces


def main():
    say = make_logger(LOGS / "find_existing_r2scan.log")
    sample = read_tsv(MANIFESTS / "test_1000.tsv")
    wanted = {r["geometry_hash"]: r for r in sample}
    say(f"sample: {len(sample)} structures, {len(wanted)} distinct geometry hashes")

    hits = []
    for name, path in SOURCES.items():
        index = read_tsv(DELTA_INDEX / f"{name}.tsv")
        found = [r for r in index if r["hash"] in wanted]
        say(f"{name}: {len(index)} indexed frames (hash = delta_pbe_10k_vs_mad16 soft_hash of {path}), {len(found)} hash matches")
        for r in found:
            s = wanted[r["hash"]]
            header, symbols, positions, forces = read_frame(path, int(r["offset"]))
            _, _, numbers, pos_pbe, cell_pbe, pbc_pbe = frame_at(PBE_XYZ, 0) if False else next(
                f for i, f in enumerate(parse_frames(PBE_XYZ)) if i == int(s["pbe_frame_idx"]))
            assert soft_hash(numbers, pos_pbe, cell_pbe, pbc_pbe) == r["hash"]
            dpos = float(np.abs(positions - pos_pbe).max())
            hits.append({**{k: s[k] for k in ("sample_rank", "configuration_id", "geometry_hash", "natoms")}, "r2scan_source": name,
                         "r2scan_idx": r["idx"], "r2scan_offset": r["offset"], "r2scan_subset": r["subset"], "r2scan_frame_id": r["frame_id"],
                         "r2scan_energy": r["energy"], "r2scan_atomization_energy": r["atomization_energy"], "fmax_r2scan": r["fmax"],
                         "r2scan_pbc": r["pbc"], "pbe_pbc": s["pbc"], "max_abs_position_diff_A": dpos})

    by_cid = Counter(h["configuration_id"] for h in hits)
    multi = {c: n for c, n in by_cid.items() if n > 1}
    say(f"structures with an existing r2SCAN label: {len(by_cid)} ({len(hits)} hits, {len(multi)} matched in more than one source)")

    columns = ["sample_rank", "configuration_id", "geometry_hash", "natoms", "r2scan_source", "r2scan_idx", "r2scan_offset", "r2scan_subset",
               "r2scan_frame_id", "r2scan_energy", "r2scan_atomization_energy", "fmax_r2scan", "r2scan_pbc", "pbe_pbc", "max_abs_position_diff_A"]
    write_tsv(MANIFESTS / "existing_r2scan_hits.tsv", sorted(hits, key=lambda h: int(h["sample_rank"])), columns)

    status = {}
    for r in sample:
        mine = [h for h in hits if h["configuration_id"] == r["configuration_id"]]
        if mine:
            preferred = sorted(mine, key=lambda h: list(SOURCES).index(h["r2scan_source"]))[0]
            status[r["configuration_id"]] = ("existing", preferred["r2scan_source"])
        else:
            status[r["configuration_id"]] = ("required", "")
    for r in sample:
        r["r2scan_status"], r["r2scan_source"] = status[r["configuration_id"]]
    cols = list(sample[0].keys())
    write_tsv(MANIFESTS / "test_1000_status.tsv", sample, cols)

    summary = {"sources": {n: {"path": str(p), "sha256": sha256(p)} for n, p in SOURCES.items()},
               "index_files": {n: str(DELTA_INDEX / f"{n}.tsv") for n in SOURCES},
               "match_rule": "identical delta_pbe_10k_vs_mad16 soft_hash (species order + positions and cell rounded to 1e-4 A); positions re-read and compared",
               "existing": sum(1 for v in status.values() if v[0] == "existing"),
               "required": sum(1 for v in status.values() if v[0] == "required"),
               "existing_by_source": dict(Counter(v[1] for v in status.values() if v[0] == "existing")),
               "hits": len(hits), "hit_ids": sorted(by_cid)}
    with open(MANIFESTS / "EXISTING_R2SCAN.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    say(json.dumps({k: v for k, v in summary.items() if k not in ("sources", "index_files")}, indent=1))


if __name__ == "__main__":
    main()
