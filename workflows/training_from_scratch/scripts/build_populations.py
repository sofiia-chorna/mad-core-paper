import csv
import hashlib
import json
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parent.parent
PARENT = S.parent / "voronoi_weighted" / "pbe_20k" / "manifests"
V2 = S.parent / "scratch_scaling_v2" / "manifests"
OUT = S / "reports" / "populations"
MANIFESTS = OUT / "manifests"
SEED = 20261008
BALANCED_RANDOM = 2000
KEYS = ("configuration_id", "frozen_configuration_id", "geom_hash", "invariant_fp")
COLUMNS = ("population", "source_test", "rank", "configuration_id", "frozen_configuration_id", "natoms", "geom_hash", "invariant_fp")
FROZEN_SHA256 = {
    "test_random": "3e3290a671939699e31117b4a76fb71f86983ba1e85fb5e256e4a76f767f1a77",
    "test_madcore": "0dc1b2c4729fee780bfc62055776ead3aff987315bfa728f09e0cd8c9530466a",
}
TRAINING = [f"{arm}_{size}_ordinary" for arm in ("madcore", "random") for size in ("2k", "5k", "10k", "20k")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def tsv_text(rows):
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row[c]) for c in COLUMNS))
    return "\n".join(lines) + "\n"


def as_rows(population, source_test, rows):
    return [{"population": population, "source_test": source_test, **{c: r[c] for c in COLUMNS[2:]}} for r in rows]


# a frozen manifest is written once, later runs must rebuild exactly the same bytes
def freeze(path, text):
    if path.exists():
        assert path.read_text() == text, f"{path} exists and differs from the rebuilt selection"
        return "verified"
    path.write_text(text)
    return "written"


def counts(rows):
    atoms = sum(int(r["natoms"]) for r in rows)
    return {"structures": len(rows), "atoms": atoms, "force_components": 3 * atoms}


def overlaps(rows_a, rows_b):
    return {k: len({r[k] for r in rows_a if r[k]} & {r[k] for r in rows_b if r[k]}) for k in KEYS}


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    # frozen tests, checked against the checksums recorded when the split was built
    tests = {}
    for name, expected in FROZEN_SHA256.items():
        path = PARENT / f"{name}.tsv"
        assert sha256(path) == expected, f"{path} changed"
        tests[name] = sorted(read_tsv(path), key=lambda r: int(r["rank"]))
    assert len(tests["test_random"]) == 5847 and len(tests["test_madcore"]) == 2000

    # balanced draw from the random test by position in rank order only, no prediction is read here
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(len(tests["test_random"]), size=BALANCED_RANDOM, replace=False))
    random_subset = [tests["test_random"][i] for i in picked]

    populations = {
        "union_7847": as_rows("union_7847", "test_random", tests["test_random"]) + as_rows("union_7847", "test_madcore", tests["test_madcore"]),
        "balanced_4000": as_rows("balanced_4000", "test_random", random_subset) + as_rows("balanced_4000", "test_madcore", tests["test_madcore"]),
    }

    status = {}
    for name, rows in populations.items():
        status[name] = freeze(MANIFESTS / f"{name}.tsv", tsv_text(rows))
    checksums = "".join(f"{sha256(MANIFESTS / f'{name}.tsv')}  {name}.tsv\n" for name in populations)
    (MANIFESTS / "CHECKSUMS.sha256").write_text(checksums)

    # validation and training sets of round 3, for the leakage check
    holdouts = {name: read_tsv(S / "manifests" / f"{name}.tsv") for name in TRAINING}
    holdouts["val"] = read_tsv(PARENT / "val.tsv")
    holdouts["val_madcore_250"] = read_tsv(V2 / "val_madcore_250.tsv")
    training_union = [r for name in TRAINING for r in holdouts[name]]
    validation_union = holdouts["val"] + holdouts["val_madcore_250"]

    summary = {
        "seed": SEED,
        "draw": f"numpy default_rng({SEED}).choice(5847, size={BALANCED_RANDOM}, replace=False) over test_random rows in rank order, indices sorted; selection by id only",
        "frozen_tests": {name: {"manifest": str(PARENT / f"{name}.tsv"), "sha256": FROZEN_SHA256[name]} for name in FROZEN_SHA256},
        "manifests": {name: {"path": str(MANIFESTS / f"{name}.tsv"), "sha256": sha256(MANIFESTS / f"{name}.tsv"), "status": status[name]} for name in populations},
        "balanced_random_rank_range": [int(random_subset[0]["rank"]), int(random_subset[-1]["rank"])],
        "populations": {},
    }

    for name, rows in populations.items():
        parts = {test: [r for r in rows if r["source_test"] == test] for test in FROZEN_SHA256}
        total = counts(rows)
        entry = {"total": total, "parts": {}, "checks": {}}
        for test, part in parts.items():
            c = counts(part)
            entry["parts"][test] = {
                **c,
                "share_of_structures": round(c["structures"] / total["structures"], 4),
                "share_of_atoms": round(c["atoms"] / total["atoms"], 4),
                "share_of_force_components": round(c["force_components"] / total["force_components"], 4),
                "mean_atoms_per_structure": round(c["atoms"] / c["structures"], 2),
            }

        # disjointness of the two parts, uniqueness inside the population, and no overlap with training or validation
        entry["checks"]["test_random_x_test_madcore"] = overlaps(parts["test_random"], parts["test_madcore"])
        entry["checks"]["unique_configuration_id"] = len({r["configuration_id"] for r in rows}) == len(rows)
        entry["checks"]["unique_geom_hash"] = len({r["geom_hash"] for r in rows}) == len(rows)
        entry["checks"]["x_round3_training_union"] = overlaps(rows, training_union)
        entry["checks"]["x_round3_validation"] = overlaps(rows, validation_union)
        for key in KEYS[:3]:
            assert entry["checks"]["test_random_x_test_madcore"][key] == 0, (name, key)
            assert entry["checks"]["x_round3_training_union"][key] == 0, (name, key)
            assert entry["checks"]["x_round3_validation"][key] == 0, (name, key)
        assert entry["checks"]["unique_configuration_id"] and entry["checks"]["unique_geom_hash"]
        summary["populations"][name] = entry

        print(f"{name}: {total['structures']:,} structures, {total['atoms']:,} atoms, {total['force_components']:,} force components ({status[name]})")
        for test, part in entry["parts"].items():
            print(f"  {test}: {part['structures']:,} structures ({part['share_of_structures']:.1%}), {part['atoms']:,} atoms ({part['share_of_atoms']:.1%})")
        print(f"  overlaps, tests {entry['checks']['test_random_x_test_madcore']}, training {entry['checks']['x_round3_training_union']}, validation {entry['checks']['x_round3_validation']}")

    (OUT / "POPULATIONS.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote", OUT / "POPULATIONS.json")


if __name__ == "__main__":
    main()
