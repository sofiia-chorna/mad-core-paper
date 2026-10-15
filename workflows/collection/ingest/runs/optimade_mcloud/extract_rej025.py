#!/usr/bin/env python
"""Extract per-dataset rej025 (rejected@0.35, passed@0.25) xyz for the OPTIMADE MC datasets.

The upstream optimade filter writes the 0.25 pass as ONE combined flat file
(download_optimade/filtered_xyz_0.25/part_0000.xyz) across ALL providers, NOT per
dataset — so the absence of a per-dataset filtered_xyz_0.25/<ds>/ folder does NOT mean
rej025 is empty. A dataset's rej025 = (its rejected@0.35 structure_ids) ∩ (the combined
passed@0.25 structure_ids). This pulls those frames out into a per-dataset xyz so they can
be ingested+featurized as the `filtered_rej_passes_at_025` variant and joined into the FPS
pool (both-variants rule). Writes to a user-owned staging dir (does NOT touch the upstream
download_optimade tree).

Result (verified 2026-05-29): only mcloudarchive__m0-zg yields any (3 structs:
MAT00001563, MAT00007268, MAT00036645). The other 11 MC datasets -> 0 (no file written).
Run on CSCS (login node is fine; pure text)."""
import os

DLO = "${STORE_ROOT}/dataset_collection/download_optimade"
PASS25 = f"{DLO}/filtered_xyz_0.25/part_0000.xyz"
OUT_BASE = "${STORE_USER_ROOT}/optimade_rej025"
MC = [
    "mcloud__2dtopo", "mcloud__autowannier", "mcloud__curated-cofs", "mcloud__mc2d",
    "mcloud__pyrene-mofs", "mcloud__stoceriaitf", "mcloud__tin-antimony-sulfoiodide",
    "mcloudarchive__1z-pd", "mcloudarchive__ed-g2", "mcloudarchive__jk-9v",
    "mcloudarchive__m0-zg", "mcloudarchive__zt-z4",
]


def frames(path):
    """Yield (block_text, structure_id) for each xyz frame."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            natoms = int(line.strip())
            comment = f.readline()
            atoms = [f.readline() for _ in range(natoms)]
            sid = None
            for tok in comment.split():
                if tok.startswith("structure_id="):
                    sid = tok.split("=", 1)[1]
                    break
            yield (line + comment + "".join(atoms)), sid


def rejected_ids(ds):
    rd = f"{DLO}/rejected_xyz/{ds}"
    ids = set()
    if not os.path.isdir(rd):
        return ids
    for fn in os.listdir(rd):
        if fn.endswith(".xyz"):
            for _, sid in frames(os.path.join(rd, fn)):
                if sid:
                    ids.add(sid)
    return ids


def main():
    # combined passed@0.25 frames, indexed by structure_id
    pass_by_id = {}
    for block, sid in frames(PASS25):
        if sid:
            pass_by_id[sid] = block
    print(f"combined passed@0.25: {len(pass_by_id)} structures")
    total = 0
    for ds in MC:
        rej = rejected_ids(ds)
        hit = sorted(rej & pass_by_id.keys())
        if not hit:
            print(f"  {ds:38s} rej@035={len(rej):<4d} rej025=0")
            continue
        out = f"{OUT_BASE}/{ds}/part_0000.xyz"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            for sid in hit:
                f.write(pass_by_id[sid])
        total += len(hit)
        print(f"  {ds:38s} rej@035={len(rej):<4d} rej025={len(hit)} -> {out}  ids={hit}")
    print(f"DONE: wrote {total} rej025 structs across MC datasets")


if __name__ == "__main__":
    main()
