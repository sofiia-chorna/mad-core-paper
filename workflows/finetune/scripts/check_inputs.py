import csv
import hashlib
import json
import random
from pathlib import Path

import metatensor.torch as mts
import metatomic.torch  # noqa: F401
import numpy as np
import torch
from ase.io import iread

D = Path(__file__).resolve().parent.parent
E = D.parent
VALID = E.parent
TRAIN_XYZ = E / "scratch_scaling_v3" / "manifests" / "xyz" / "madcore_20k_aug.xyz"
VAL_XYZ = E / "scratch_scaling_v2" / "manifests" / "xyz" / "val_combined_1250.xyz"
SETS = E / "scratch_scaling_v3" / "manifests" / "SETS.json"
ORDINARY = E / "scratch_scaling_v3" / "manifests" / "madcore_20k_ordinary.tsv"
REFERENCES = E / "scratch_scaling_v3" / "manifests" / "isolated_atom_references.tsv"
FROZEN = E / "voronoi_weighted" / "pbe_20k" / "manifests"
VAL_MADCORE = E / "scratch_scaling_v2" / "manifests" / "val_madcore_250.tsv"
BALANCED = E / "scratch_scaling_v3" / "reports" / "populations" / "manifests" / "balanced_4000.tsv"
PRETRAINED = E / "models" / "pet-omat-s-v1.0.0.ckpt"
PBE_LABELS = VALID / "dft" / "lda_pbe" / "final" / "madcore_25k" / "pbe.xyz"
MONOMERS = VALID / "dft" / "lda_pbe" / "references" / "monomers_pbe.xyz"
KEYS = ("configuration_id", "frozen_configuration_id", "geom_hash")
SAMPLE = 200
SEED = 20261009


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, columns):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in columns) + "\n")


def energy(atoms):
    return float(atoms.calc.results["energy"]) if atoms.calc is not None and "energy" in atoms.calc.results else float(atoms.info["energy"])


def forces(atoms):
    return atoms.calc.results["forces"] if atoms.calc is not None and "forces" in atoms.calc.results else atoms.arrays["forces"]


def main():
    checks, report = [], {}
    sets = json.loads(SETS.read_text())

    # the exact files of the scratch model, checked against their recorded checksums
    train_sha, val_sha = sha256(TRAIN_XYZ), sha256(VAL_XYZ)
    checks.append(("training file is the scratch model's file", train_sha == sets["sets"]["madcore_20k"]["xyz_sha256"]))
    checks.append(("validation file is the scratch model's file", val_sha == sets["validation"]["combined_xyz_sha256"]))
    report["files"] = {"training_xyz": str(TRAIN_XYZ), "training_sha256": train_sha, "validation_xyz": str(VAL_XYZ), "validation_sha256": val_sha,
                       "pretrained": str(PRETRAINED), "pretrained_sha256": sha256(PRETRAINED)}

    # split identifiers, written in the order of the files and checked against the scratch manifests
    train_frames = list(iread(str(TRAIN_XYZ)))
    val_frames = list(iread(str(VAL_XYZ)))
    ordinary, references = read_tsv(ORDINARY), read_tsv(REFERENCES)
    val_rows = read_tsv(FROZEN / "val.tsv") + read_tsv(VAL_MADCORE)
    train_ids = [a.info["configuration_id"] for a in train_frames]
    val_ids = [a.info["configuration_id"] for a in val_frames]
    checks.append(("training file holds exactly the 20,000 ordinary structures and the 102 references of the scratch manifests",
                   sorted(train_ids) == sorted([r["configuration_id"] for r in ordinary + references]) and len(train_ids) == 20102))
    checks.append(("validation file holds exactly the 1,000 random and 250 MAD-CORE validation structures",
                   sorted(val_ids) == sorted(r["configuration_id"] for r in val_rows) and len(val_ids) == 1250))
    reference_ids = {r["configuration_id"] for r in references}
    write_tsv(D / "manifests" / "train_ids.tsv", [{"index": i, "configuration_id": c, "kind": "isolated_atom_reference" if c in reference_ids else "madcore"}
                                                 for i, c in enumerate(train_ids)], ["index", "configuration_id", "kind"])
    random_val = {r["configuration_id"] for r in read_tsv(FROZEN / "val.tsv")}
    write_tsv(D / "manifests" / "val_ids.tsv", [{"index": i, "configuration_id": c, "kind": "random" if c in random_val else "madcore"}
                                               for i, c in enumerate(val_ids)], ["index", "configuration_id", "kind"])

    # disjointness of training, validation and both full frozen tests
    groups = {"train_ordinary": ordinary, "train_references": references, "val_random": read_tsv(FROZEN / "val.tsv"), "val_madcore": read_tsv(VAL_MADCORE),
              "test_random": read_tsv(FROZEN / "test_random.tsv"), "test_madcore": read_tsv(FROZEN / "test_madcore.tsv")}
    overlaps = {}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a} x {b}"] = {k: len({r[k] for r in groups[a] if r.get(k)} & {r[k] for r in groups[b] if r.get(k)}) for k in KEYS}
    report["overlaps"] = overlaps
    checks.append(("all training, validation and test groups disjoint by configuration id, frozen id and geometry hash",
                   all(sum(v.values()) == 0 for v in overlaps.values())))
    balanced = read_tsv(BALANCED)
    checks.append(("the balanced 4,000 test population lies inside the frozen tests",
                   {r["configuration_id"] for r in balanced} <= {r["configuration_id"] for r in groups["test_random"] + groups["test_madcore"]}))
    report["balanced_4000"] = {"manifest": str(BALANCED), "sha256": sha256(BALANCED)}

    # isolated-atom references: zero labels, one per element, every element known to the pretrained model
    refs = [a for a in train_frames if a.info["configuration_id"] in reference_ids]
    numbers = sorted(int(a.numbers[0]) for a in refs)
    checks.append(("102 references, one atom each, elements 1 to 102 once each", len(refs) == 102 and numbers == list(range(1, 103))
                   and all(len(a) == 1 for a in refs)))
    checks.append(("every reference has atomization energy 0 and zero forces",
                   all(float(a.info["atomization_energy"]) == 0.0 and not np.any(forces(a)) for a in refs)))
    pretrained = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    types = list(pretrained["model_data"]["dataset_info"].atomic_types)
    state = pretrained["best_model_state_dict"]
    block = mts.load_buffer(state["additive_models.0.energy_composition_buffer"]).block()
    weights = {int(t): float(v) for t, v in zip(block.samples.values[:, 0], block.values.flatten())}
    scale = float(mts.load_buffer(state["scaler.energy_scaler_buffer"]).block().values.flatten()[0])
    checks.append(("every reference element has a pretrained atomic type and composition weight", all(z in types and z in weights for z in numbers)))

    # one energy convention: the training atomization energies use the same PBE isolated-atom energies as the references
    e0 = {int(a.numbers[0]): energy(a) for a in iread(str(MONOMERS))}
    raw = {r["configuration_id"]: float(r["e_pbe"]) for r in references}
    checks.append(("the reference monomer energies equal the PBE isolated-atom energies of the campaign",
                   all(abs(raw[f"isolated_atom:{a.get_chemical_symbols()[0]}"] - e0[int(a.numbers[0])]) < 1e-9 for a in refs)))
    sample = set(random.Random(SEED).sample([a.info["configuration_id"] for a in train_frames if a.info["configuration_id"] not in reference_ids], SAMPLE))
    labelled = {a.info["configuration_id"]: a for a in iread(str(PBE_LABELS)) if a.info["configuration_id"] in sample}
    gap = max(abs(float(a.info["atomization_energy"]) - (energy(labelled[a.info["configuration_id"]]) - sum(e0[int(z)] for z in a.numbers)))
              for a in train_frames if a.info["configuration_id"] in sample)
    checks.append((f"{SAMPLE} sampled training atomization energies equal the PBE energy minus the same isolated-atom energies (max gap {gap:.1e} eV)", gap < 1e-6))

    # starting offset per atom against the inherited baseline, ordinary structures against references
    ordinary_offset = np.array([(float(a.info["atomization_energy"]) - sum(weights[int(z)] for z in a.numbers)) / len(a)
                                for a in train_frames if a.info["configuration_id"] not in reference_ids])
    reference_offset = np.array([-weights[int(a.numbers[0])] for a in refs])
    low, high = np.percentile(ordinary_offset, [1, 99])
    inside = float(np.mean((reference_offset >= low) & (reference_offset <= high)))
    report["inherited_baseline"] = {
        "scale_eV": scale,
        "offset_per_atom_eV": {
            "ordinary": {"median": float(np.median(ordinary_offset)), "p5": float(np.percentile(ordinary_offset, 5)), "p95": float(np.percentile(ordinary_offset, 95)),
                         "p1": float(low), "p99": float(high)},
            "references": {"median": float(np.median(reference_offset)), "p5": float(np.percentile(reference_offset, 5)), "p95": float(np.percentile(reference_offset, 95)),
                           "min": float(reference_offset.min()), "max": float(reference_offset.max())},
        },
        "references_inside_ordinary_1_99_percent": inside,
    }

    # non-periodic single atoms in the tests, reported because the references are isolated atoms of the same elements
    single = {}
    for t in ("test_random", "test_madcore"):
        rows = [r for r in groups[t] if int(r["natoms"]) == 1 and r["geometry_class"] == "isolated"]
        single[t] = {"count": len(rows), "elements": sorted({r["elements"] for r in rows})}
    report["isolated_single_atoms_in_tests"] = single

    passed = all(ok for _, ok in checks)
    report["checks"] = [{"check": c, "passed": ok} for c, ok in checks]
    (D / "reports" / "input_checks.json").write_text(json.dumps(report, indent=1) + "\n")
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    print(json.dumps(report["inherited_baseline"], indent=1))
    print("isolated single atoms in tests:", single)
    print("INPUT CHECKS", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
