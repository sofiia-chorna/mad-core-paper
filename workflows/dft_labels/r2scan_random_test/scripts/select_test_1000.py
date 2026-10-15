import json
import platform
from collections import Counter, defaultdict

import numpy as np

from paths import (DELTA_MANIFESTS, LOGS, MANIFESTS, PARENT, PBE_MANIFEST, PBE_XYZ, REPORTS, REPORT_ONLY_SETS, SAMPLE_SIZE, SEED,
                    TEST_COLUMNS, TRAIN_VAL_SETS, make_logger, parse_frames, read_tsv, sha256, soft_hash, write_tsv)


def main():
    MANIFESTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    say = make_logger(LOGS / "select_test_1000.log")

    parent = read_tsv(PARENT)
    say(f"parent {PARENT}: {len(parent)} rows, sha256 {sha256(PARENT)}")
    parent_ids = [r["configuration_id"] for r in parent]
    assert len(set(parent_ids)) == len(parent_ids), "duplicate configuration_id in the parent manifest"

    labelled = {r["configuration_id"]: r for r in read_tsv(PBE_MANIFEST)}
    eligible_check = [c for c, r in labelled.items()
                      if r["has_pbe"] == "1" and r["has_lda"] == "1" and float(r["fmax_pbe"]) <= 100 and float(r["fmax_lda"]) <= 100]
    say(f"labelled manifest rows with PBE and LDA labels and max force <= 100 eV/A at both: {len(eligible_check)}; "
        f"equal to the parent id set: {set(eligible_check) == set(parent_ids)}")

    frames = {}
    for idx, (offset, info, numbers, positions, cell, pbc) in enumerate(parse_frames(PBE_XYZ)):
        cid = info["configuration_id"]
        if cid in labelled and cid in set(parent_ids):
            frames[cid] = {"idx": idx, "hash": soft_hash(numbers, positions, cell, pbc), "natoms": len(numbers),
                           "pbc": "".join("T" if p else "F" for p in pbc), "info": info}
    say(f"PBE geometries read from {PBE_XYZ} (sha256 {sha256(PBE_XYZ)}): {len(frames)} of {len(parent)} parent rows found")
    assert len(frames) == len(parent)

    rows = []
    for r in parent:
        f = frames[r["configuration_id"]]
        assert int(r["natoms"]) == f["natoms"]
        m = labelled[r["configuration_id"]]
        rows.append({**r, "geometry_hash": f["hash"], "pbc": f["pbc"], "dft_bucket": m["dft_bucket"], "pbe_frame_idx": f["idx"],
                     "frozen_configuration_id": m["frozen_configuration_id"], "free_energy_pbe": f["info"]["free_energy"],
                     "e_pbe_frame": f["info"]["energy"]})
        assert abs(float(rows[-1]["e_pbe_frame"]) - float(r["e_pbe"])) < 1e-6

    exclusions = []
    by_hash = defaultdict(list)
    by_fp = defaultdict(list)
    for r in rows:
        by_hash[r["geometry_hash"]].append(r["configuration_id"])
        by_fp[r["invariant_fp"]].append(r["configuration_id"])
    exact_dups = {h: sorted(c) for h, c in by_hash.items() if len(c) > 1}
    fp_dups = {h: sorted(c) for h, c in by_fp.items() if len(c) > 1}
    say(f"exact geometry duplicates within the parent (soft hash): {len(exact_dups)} groups, "
        f"{sum(len(c) for c in exact_dups.values())} rows")
    say(f"permutation/translation-invariant duplicates (invariant_fp): {len(fp_dups)} groups, {sum(len(c) for c in fp_dups.values())} rows")
    for group in list(exact_dups.values()) + list(fp_dups.values()):
        for cid in group[1:]:
            exclusions.append({"configuration_id": cid, "reason": "geometry_duplicate", "detail": f"duplicate of {group[0]}"})

    for name in TRAIN_VAL_SETS:
        rows_set = read_tsv(DELTA_MANIFESTS / f"{name}.tsv")
        hashes = {r["hash"] for r in rows_set}
        cids = {r["configuration_id"] for r in rows_set}
        hit_h = [r["configuration_id"] for r in rows if r["geometry_hash"] in hashes]
        hit_c = [r["configuration_id"] for r in rows if r["configuration_id"] in cids or r["frozen_configuration_id"] in cids]
        say(f"overlap with {name} ({len(rows_set)} rows): {len(hit_h)} by geometry hash, {len(hit_c)} by configuration_id")
        for cid in sorted(set(hit_h) | set(hit_c)):
            exclusions.append({"configuration_id": cid, "reason": f"overlap_{name}", "detail": "geometry hash or id match"})

    excluded_ids = {e["configuration_id"] for e in exclusions}
    write_tsv(MANIFESTS / "exclusions.tsv", exclusions, ["configuration_id", "reason", "detail"])
    say(f"exclusions written: {len(exclusions)} rows, {len(excluded_ids)} distinct structures")

    eligible = sorted((r for r in rows if r["configuration_id"] not in excluded_ids), key=lambda r: r["configuration_id"])
    for k, r in enumerate(eligible):
        r["eligible_rank"] = k
    say(f"eligible unique structures after exclusions, sorted by configuration_id: {len(eligible)}")

    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(eligible), size=SAMPLE_SIZE, replace=False)
    picked = sorted((eligible[i] for i in picks), key=lambda r: r["configuration_id"])
    for k, r in enumerate(picked):
        r["sample_rank"] = k
    assert len({r["configuration_id"] for r in picked}) == SAMPLE_SIZE

    write_tsv(MANIFESTS / "test_1000.tsv", picked, TEST_COLUMNS)
    write_tsv(MANIFESTS / "eligible_pool.tsv", eligible, ["eligible_rank", "configuration_id", "geometry_hash", "natoms", "dataset_group"])
    say(f"sample written: {MANIFESTS / 'test_1000.tsv'} sha256 {sha256(MANIFESTS / 'test_1000.tsv')}")

    overlap_report = {}
    for name in REPORT_ONLY_SETS:
        rows_set = read_tsv(DELTA_MANIFESTS / f"{name}.tsv")
        hashes = {r["hash"] for r in rows_set}
        cids = {r["configuration_id"] for r in rows_set}
        hit = [r["configuration_id"] for r in picked if r["geometry_hash"] in hashes or r["configuration_id"] in cids]
        overlap_report[name] = {"rows": len(rows_set), "overlap_with_test_1000": len(hit), "ids": hit}
        say(f"report-only overlap of the sample with {name}: {len(hit)}")

    summary = {
        "parent_manifest": {"path": str(PARENT), "sha256": sha256(PARENT), "rows": len(parent),
                            "definition": "prod_experiment test set: random_test_6k structures with converged PBE and LDA labels and "
                                          "max per-atom force <= 100 eV/A at both functionals (5,847 of 6,000 selected)"},
        "geometry_source": {"path": str(PBE_XYZ), "sha256": sha256(PBE_XYZ), "labelled_manifest_sha256": sha256(PBE_MANIFEST)},
        "geometry_hash": "blake2b-16 of species order + positions and cell rounded to 1e-4 A (delta_pbe_10k_vs_mad16 soft_hash, pbc excluded)",
        "exclusions": {"exact_duplicate_groups": exact_dups, "invariant_fp_duplicate_groups": fp_dups,
                       "counts": dict(Counter(e["reason"] for e in exclusions)), "distinct_structures_excluded": len(excluded_ids)},
        "eligible_after_exclusions": len(eligible),
        "procedure": "sort eligible by configuration_id (python string order); numpy.random.default_rng(20260908).choice(n, size=1000, "
                     "replace=False); output sorted by configuration_id",
        "seed": SEED, "sample_size": SAMPLE_SIZE, "numpy_version": np.__version__, "python_version": platform.python_version(),
        "sample_manifest": {"path": str(MANIFESTS / "test_1000.tsv"), "sha256": sha256(MANIFESTS / "test_1000.tsv")},
        "sample_summary": {"dataset_groups": dict(Counter(r["dataset_group"] for r in picked)),
                           "dft_bucket": dict(Counter(r["dft_bucket"] for r in picked)),
                           "pbc": dict(Counter(r["pbc"] for r in picked)),
                           "total_atoms": int(sum(int(r["natoms"]) for r in picked)),
                           "natoms_max": max(int(r["natoms"]) for r in picked),
                           "elements": len({e for r in picked for e in r["elements"].split()})},
        "report_only_overlap": overlap_report,
    }
    with open(MANIFESTS / "SELECTION.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    say(json.dumps(summary["sample_summary"], indent=1))


if __name__ == "__main__":
    main()
