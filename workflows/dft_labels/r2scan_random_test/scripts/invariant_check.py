from collections import defaultdict

import numpy as np

from paths import BOTR_XYZ, LOGS, MAD16_R2SCAN, MANIFESTS, PBE_XYZ, make_logger, parse_frames, read_tsv, write_tsv

import sys
sys.path.insert(0, str(MANIFESTS.parents[2] / "experiments" / "prod_experiment" / "scripts"))
from common import invariant_fingerprint  # noqa: E402

SOURCES = {"mad16_r2scan_train": MAD16_R2SCAN["train"], "mad16_r2scan_val": MAD16_R2SCAN["val"], "mad16_r2scan_test": MAD16_R2SCAN["test"],
           "botr_r2scan": BOTR_XYZ}


def main():
    say = make_logger(LOGS / "invariant_check.log")
    sample = read_tsv(MANIFESTS / "test_1000.tsv")
    idx_to_row = {int(r["pbe_frame_idx"]): r for r in sample}
    fps = {}
    comps = defaultdict(list)
    for i, (_, info, numbers, positions, cell, pbc) in enumerate(parse_frames(PBE_XYZ)):
        if i in idx_to_row:
            cid = idx_to_row[i]["configuration_id"]
            fp = invariant_fingerprint(numbers, positions, cell, pbc)
            assert fp == idx_to_row[i]["invariant_fp"], cid
            fps[fp] = cid
            comps[tuple(sorted(numbers.tolist()))].append(cid)
    say(f"sample fingerprints recomputed and verified: {len(fps)}")

    hits = []
    for name, path in SOURCES.items():
        same_comp = 0
        for i, (_, info, numbers, positions, cell, pbc) in enumerate(parse_frames(path)):
            key = tuple(sorted(numbers.tolist()))
            if key not in comps:
                continue
            same_comp += 1
            fp = invariant_fingerprint(numbers, positions, cell, pbc)
            if fp in fps:
                hits.append({"configuration_id": fps[fp], "source": name, "idx": i, "subset": info.get("subset", ""), "frame_id": info.get("frame_id", "")})
        say(f"{name}: {same_comp} frames share a composition with a sample structure, {len([h for h in hits if h['source'] == name])} invariant-fingerprint matches")
    write_tsv(MANIFESTS / "invariant_fp_hits.tsv", hits, ["configuration_id", "source", "idx", "subset", "frame_id"])
    say(f"total invariant-fingerprint matches: {len(hits)}")


if __name__ == "__main__":
    main()
