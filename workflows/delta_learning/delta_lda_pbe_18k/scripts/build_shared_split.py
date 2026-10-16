import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

D = Path(__file__).resolve().parent.parent
E = D.parent
sys.path.insert(0, str(E / "delta_lda" / "scripts"))
import build_sets as B

spec = importlib.util.spec_from_file_location("lda20k", E / "delta_lda_18774" / "scripts" / "build_sets_lda.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)
spec = importlib.util.spec_from_file_location("overlap", E / "delta_lda_18774" / "scripts" / "check_overlap_invariant.py")
OV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OV)

OLD = E / "delta_lda_18774" / "manifests"
TEN_K = E / "delta_lda" / "manifests"
MANIFESTS = D / "manifests"
XYZ = MANIFESTS / "xyz"
SEED = 42
SIZES = {"train": 18000, "val": 1000, "test_madcore": 1881}
SMOKE = {"train": 256, "val": 64, "test_madcore": 64, "test_random": 64}
SCALARS = ("atomization_energy", "atomization_energy_lda", "atomization_energy_pbe", "delta_energy_lda", "delta_energy_pbe")
SIZE_BINS = ((1, 1, "1"), (2, 3, "2 to 3"), (4, 9, "4 to 9"), (10, 29, "10 to 29"), (30, 10 ** 6, "30 or more"))
LABEL_TOL = 1e-9


def size_bin(n):
    for lo, hi, name in SIZE_BINS:
        if lo <= n <= hi:
            return name


def describe(entries):
    natoms = np.array([e["natoms"] for e in entries])
    elements = set()
    for e in entries:
        elements.update(int(z) for z in e["r2scan_frame"]["numbers"])
    big = [e["configuration_id"] for e in entries if e["pbc"] != "TTT" and e["natoms"] >= 10]
    return {"structures": len(entries), "atoms": int(natoms.sum()), "distinct_elements": len(elements), "elements": [chemical_symbols[z] for z in sorted(elements)],
            "sources": dict(Counter(e["r2scan_source"] for e in entries)), "origins": dict(Counter(e["origin"] for e in entries)), "pbc": dict(Counter(e["pbc"] for e in entries)),
            "size_bins": {name: int(sum(lo <= n <= hi for n in natoms)) for lo, hi, name in SIZE_BINS}, "natoms_median": float(np.median(natoms)), "natoms_max": int(natoms.max()),
            "nonperiodic_molecules_10_atoms_or_more": len(big), "nonperiodic_molecules_10_atoms_or_more_ids": big,
            "single_atom_nonperiodic": [e["configuration_id"] for e in entries if e["natoms"] == 1 and e["pbc"] != "TTT"]}


def write_tsv(path, entries):
    columns = ["split", "stable_index", "origin", "configuration_id", "frozen_configuration_id", "hash", "r2scan_source", "natoms", "pbc", *SCALARS, "fmax_r2scan", "fmax_lda", "fmax_pbe"]
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for e in entries:
            fh.write("\t".join(str(e[c]) for c in columns) + "\n")


def main():
    XYZ.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "sizes": SIZES, "force_cut_eV_per_A": B.FORCE_CUT, "position_tolerance_A": B.POSITION_TOL, "sources": {}, "checks": []}
    checks = summary["checks"]

    assert sorted((B.FINAL / "SHA256SUMS").read_text().split("\n")) == sorted((B.STORE_FINAL / "SHA256SUMS").read_text().split("\n"))
    references = {f: B.monomer_energies(p) for f, p in B.MONOMERS.items()}
    references["r2scan"] = {int(z): float(v) for z, v in json.loads(B.E0_R2SCAN.read_text()).items()}
    for name, path in (*B.MONOMERS.items(), ("r2scan", B.E0_R2SCAN)):
        summary["sources"][f"references_{name}"] = {"path": str(path), "sha256": B.sha256(path), "elements": len(references[name])}

    def atomization(frame, functional):
        return float(frame["info"]["energy"]) - sum(references[functional][int(z)] for z in frame["numbers"])

    labels = {}
    for arm in ("madcore_25k", "random_test_6k", "random_25k"):
        for functional in ("lda", "pbe"):
            for frame in B.read_all(B.FINAL / arm / f"{functional}.xyz"):
                labels.setdefault(frame["info"]["configuration_id"], {})[functional] = frame
    pool_flags = {r["configuration_id"]: r for r in B.read_tsv(L.POOL)}

    indexes = {"botr": (B.BOTR_XYZ, {r["hash"]: int(r["offset"]) for r in B.read_tsv(B.INDEX / "botr_r2scan.tsv")})}
    for split in ("train", "val", "test"):
        indexes[f"mad16_{split}"] = (B.MAD16[split], {r["hash"]: int(r["offset"]) for r in B.read_tsv(B.INDEX / f"mad16_r2scan_{split}.tsv")})
    indexes["botr0"] = (L.BOTR0_XYZ, {r["hash"]: int(r["offset"]) for r in B.read_tsv(L.BOTR0_INDEX)})
    lookup = {k: indexes[k] for k in ("botr", "mad16_train", "mad16_val", "mad16_test", "botr0")}
    mad16_heldout = set(indexes["mad16_val"][1]) | set(indexes["mad16_test"][1])
    summary["sources"]["r2scan"] = {k: str(v[0]) for k, v in lookup.items()}
    summary["sources"]["lda_pbe_final"] = str(B.FINAL)

    def resolve(cid, frozen_cid, origin, r2scan_source=None, r2scan_frame=None):
        have = labels.get(cid, {})
        if "lda" not in have or "pbe" not in have:
            return None, "no lda or pbe label"
        lda, pbe = have["lda"], have["pbe"]
        key = B.frame_hash(lda)
        if r2scan_frame is None:
            r2scan_source, r2scan_frame = B.r2scan_lookup(key, lookup)
            if r2scan_frame is None:
                return None, "no r2scan label"
        if not (B.same_geometry(r2scan_frame, lda) and B.same_geometry(r2scan_frame, pbe)):
            return None, "geometry differs between functionals"
        pbcs = {fr["pbc"].replace(" ", "") for fr in (r2scan_frame, lda, pbe)}
        if len(pbcs) != 1:
            return None, "pbc differs between functionals"
        frames = {"r2scan": r2scan_frame, "lda": lda, "pbe": pbe}
        if not all(np.isfinite(float(fr["info"]["energy"])) and np.isfinite(fr["forces"]).all() for fr in frames.values()):
            return None, "non-finite label"
        if max(B.fmax(fr) for fr in frames.values()) > B.FORCE_CUT:
            return None, "force above 100 eV/A"
        if cid in pool_flags and pool_flags[cid]["fragmented"] == "1":
            return None, "fragmented"
        for name, fr in frames.items():
            if "free_energy" in fr["info"]:
                assert abs(float(fr["info"]["free_energy"]) - float(fr["info"]["energy"])) < 1e-9, (cid, name)
            file_atomization = fr["info"].get("atomization_energy", fr["info"].get("corrected_energy"))
            if file_atomization is not None:
                assert abs(float(file_atomization) - atomization(fr, name)) <= B.ATOMIZATION_TOL, (cid, name)
        a = {f: atomization(fr, f) for f, fr in frames.items()}
        entry = {"configuration_id": cid, "frozen_configuration_id": frozen_cid, "origin": origin, "r2scan_source": r2scan_source, "hash": key, "r2scan_frame": r2scan_frame,
                 "natoms": len(lda["numbers"]), "pbc": pbcs.pop(), "mad16_heldout_geometry": key in mad16_heldout,
                 "atomization_energy": a["r2scan"], "atomization_energy_lda": a["lda"], "atomization_energy_pbe": a["pbe"],
                 "delta_energy_lda": a["r2scan"] - a["lda"], "delta_energy_pbe": a["r2scan"] - a["pbe"],
                 "forces": r2scan_frame["forces"], "forces_lda": lda["forces"], "forces_pbe": pbe["forces"],
                 "delta_forces_lda": r2scan_frame["forces"] - lda["forces"], "delta_forces_pbe": r2scan_frame["forces"] - pbe["forces"],
                 "fmax_r2scan": B.fmax(r2scan_frame), "fmax_lda": B.fmax(lda), "fmax_pbe": B.fmax(pbe)}
        return entry, None

    old = {name: B.read_tsv(OLD / f"{name}.tsv") for name in ("train", "val", "test_madcore")}
    pool, failures, source_changes = [], Counter(), 0
    for origin, rows in (("old_train", old["train"]), ("old_val_250", B.read_tsv(B.VAL_MADCORE)), ("old_test", old["test_madcore"])):
        for r in rows:
            entry, reason = resolve(r["configuration_id"], r["frozen_configuration_id"], origin)
            if entry is None:
                failures[f"{origin}: {reason}"] += 1
                continue
            pool.append(entry)
    old_by_cid = {r["configuration_id"]: r for rows in old.values() for r in rows}
    by_cid = {e["configuration_id"]: e for e in pool}
    label_gap = max(abs(float(old_by_cid[c]["delta_energy_lda"]) - by_cid[c]["delta_energy_lda"]) for c in old_by_cid)
    source_changes = sum(old_by_cid[c]["r2scan_source"] != by_cid[c]["r2scan_source"] for c in old_by_cid)
    counts = Counter(e["origin"] for e in pool)
    summary["pool"] = {"origins": dict(counts), "structures": len(pool), "unresolved": dict(failures), "old_lda_label_gap_eV": label_gap, "r2scan_source_changes": source_changes}
    checks.append(("pool is 18,774 old train + 232 of the old validation 250 + 1,875 old test = 20,881, every structure paired for both functionals",
                   counts == Counter({"old_train": 18774, "old_val_250": 232, "old_test": 1875}) and len(pool) == 20881))
    checks.append(("old validation 250 loses exactly the 18 without r2scan label", failures == Counter({"old_val_250: no r2scan label": 18})))
    checks.append((f"lda deltas identical to the previous manifests (gap {label_gap:.1e} eV) with the same r2scan sources ({source_changes} changes)", label_gap < LABEL_TOL and source_changes == 0))
    assert len({e["configuration_id"] for e in pool}) == len(pool)

    # one random permutation of the pool in configuration id order
    pool.sort(key=lambda e: e["configuration_id"])
    for i, e in enumerate(pool):
        e["stable_index"] = i
    order = np.random.RandomState(SEED).permutation(len(pool))
    bounds = {"train": (0, SIZES["train"]), "val": (SIZES["train"], SIZES["train"] + SIZES["val"]), "test_madcore": (SIZES["train"] + SIZES["val"], len(pool))}
    sets = {}
    for name, (lo, hi) in bounds.items():
        members = sorted(order[lo:hi].tolist())
        sets[name] = [dict(pool[i], split=name) for i in members]
    summary["split_rule"] = "pool sorted by configuration_id, numpy RandomState(42).permutation(20881), positions 0 to 17,999 train, 18,000 to 18,999 val, 19,000 to 20,880 test_madcore, each split written in stable index order"

    # the random test, unchanged 973, both functionals
    random_r2scan = {f["info"]["configuration_id"]: f for f in B.read_all(B.R2SCAN_RANDOM_TEST)}
    frozen_random = {r["configuration_id"]: r for r in B.read_tsv(B.FROZEN / "test_random.tsv")}
    old_random = {r["configuration_id"]: r for r in B.read_tsv(OLD / "test_random.tsv")}
    random_rows, reasons = [], Counter()
    for cid, frame in random_r2scan.items():
        entry, reason = resolve(cid, frozen_random[cid]["frozen_configuration_id"], "r2scan_random_test", "r2scan_random_test", frame)
        if entry is None:
            reasons[reason] += 1
            continue
        random_rows.append(dict(entry, split="test_random", stable_index=-1))
    random_rows.sort(key=lambda e: int(frozen_random[e["configuration_id"]]["rank"]))
    sets["test_random"] = random_rows
    gap_random = max(abs(float(old_random[e["configuration_id"]][k]) - e[k]) for e in random_rows for k in ("atomization_energy", "atomization_energy_lda", "delta_energy_lda"))
    checks.append((f"random test is the unchanged 973 with both pairings valid, lda labels identical to the previous file (gap {gap_random:.1e} eV), unusable {dict(reasons)}",
                   len(random_rows) == 973 and set(old_random) == {e["configuration_id"] for e in random_rows} and gap_random < LABEL_TOL))

    # written files
    for name, entries in sets.items():
        write_tsv(MANIFESTS / f"{name}.tsv", entries)
        B.write_xyz(XYZ / f"{name}.xyz", entries)
        B.write_xyz(XYZ / f"smoke_{name}.xyz", entries[:SMOKE[name]])
        write_tsv(MANIFESTS / f"smoke_{name}.tsv", entries[:SMOKE[name]])

    # checks on the written files, labels and reconstruction identities
    worst = 0.0
    for name, entries in sets.items():
        written = B.read_all(XYZ / f"{name}.xyz")
        assert len(written) == len(entries)
        for e, f in zip(entries, written):
            assert f["info"]["configuration_id"].strip('"') == e["configuration_id"]
            for key in SCALARS:
                worst = max(worst, abs(float(f["info"][key]) - e[key]))
            for functional in ("lda", "pbe"):
                worst = max(worst, abs(float(f["info"][f"atomization_energy_{functional}"]) + float(f["info"][f"delta_energy_{functional}"]) - float(f["info"]["atomization_energy"])))
    checks.append((f"written labels equal the entries, lda plus delta_lda and pbe plus delta_pbe reproduce r2scan (worst gap {worst:.1e} eV)", worst < LABEL_TOL))
    checks.append(("counts 18,000 / 1,000 / 1,881 / 973", tuple(len(sets[n]) for n in ("train", "val", "test_madcore", "test_random")) == (18000, 1000, 1881, 973)))

    keys = {}
    for name, entries in sets.items():
        keys[name] = {"configuration_id": {e["configuration_id"] for e in entries}, "frozen_configuration_id": {e["frozen_configuration_id"] for e in entries},
                      "hash": {e["hash"] for e in entries}, "fingerprint": {OV.fingerprint(e["r2scan_frame"]) for e in entries}}
        for k, v in keys[name].items():
            checks.append((f"{name}: no internal duplicates by {k}", len(v) == len(entries)))
    names = list(sets)
    overlaps = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            o = {k: len(keys[names[a]][k] & keys[names[b]][k]) for k in keys[names[a]]}
            overlaps[f"{names[a]} x {names[b]}"] = o
            checks.append((f"{names[a]} x {names[b]}: zero overlap by ids, hash and lattice aware fingerprint", all(v == 0 for v in o.values())))
    summary["overlaps"] = overlaps
    train_elements = set()
    for e in sets["train"]:
        train_elements.update(int(z) for z in e["r2scan_frame"]["numbers"])
    summary["elements_absent_from_training"] = {name: [chemical_symbols[z] for z in sorted({int(z) for e in sets[name] for z in e["r2scan_frame"]["numbers"]} - train_elements)] for name in names if name != "train"}
    checks.append(("every element of the pool occurs in training", not summary["elements_absent_from_training"]["val"] and not summary["elements_absent_from_training"]["test_madcore"]))
    summary["mad16_heldout_geometries"] = {name: [e["configuration_id"] for e in sets[name] if e["mad16_heldout_geometry"]] for name in names}
    summary["composition"] = {name: describe(entries) for name, entries in sets.items()}
    summary["composition"]["pool"] = describe(pool)

    summary["passed"] = all(ok for _, ok in checks)
    summary["checks"] = [{"check": c, "passed": ok} for c, ok in checks]
    (MANIFESTS / "SETS.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    with open(MANIFESTS / "CHECKSUMS.sha256", "w") as fh:
        for p in sorted(list(MANIFESTS.glob("*.tsv")) + list(XYZ.glob("*.xyz"))):
            fh.write(f"{B.sha256(p)}  {p.relative_to(MANIFESTS)}\n")
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    print("mad16 heldout geometries by split", {k: len(v) for k, v in summary["mad16_heldout_geometries"].items()})
    print("BUILD", "PASSED" if summary["passed"] else "FAILED")


if __name__ == "__main__":
    main()
