import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import iread

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    BASELINE_OUTPUT,
    BASELINE_PT,
    CORRECTIONS,
    D,
    DELTA_OUTPUT,
    UPSTREAM_PT,
    VALID,
    Potential,
    atoms_call,
    lips_start,
)

OUT = D / "verification"
STATIC = D / "static"
FLOAT64 = OUT / "float64"
DELTA_EXP = VALID / "experiments" / "delta_lda"
TESTS = ("test_random", "test_madcore")
TEST_XYZ = {t: DELTA_EXP / "manifests" / "xyz" / f"{t}.xyz" for t in TESTS}
TRAIN_XYZ = DELTA_EXP / "manifests" / "xyz" / "train.xyz"
E0_R2SCAN = VALID / "experiments" / "reports" / "e0_r2scan_mad16_monomers.json"
EXPECTED = {
    "pet-mad-m-v1.6.0.ckpt": "044418af7fcaf7dc6e65f5587d2afede2835f13e680dd296b399220ff786f8e4",
    "lda model_best.pt": "3324b33e9a645d898c3eb47908da8a0e48d5de512420d5e0c2ef2caf07204b99",
    "lda model.ckpt": "837d031647d7c0200114461f31328288a20727fc07d1c5e9eb6d21193e2bf935",
    "pbe model_best.pt": "3b49634952dec43e49f137205874fe838eadec620ed5c07d76d6a5633ff23326",
    "pbe model.ckpt": "dd358427dd89964f861710c1085b2b5332fa5fc99ab974da4f26c81d62e4cb07",
}
TRAIN_SAMPLE = 500
SEED = 20261009
FD_STEPS = (0.001, 0.002, 0.005, 0.01)
FD_COMPONENTS = 12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checks, report = [], {"device": device}

    # identities of the models used, each correction against the records of the delta experiment
    hashes = {
        "pet-mad-m-v1.6.0.ckpt": sha256(D / "models" / "pet-mad-m-v1.6.0.ckpt"),
        "pet-mad-m-v1.6.0.fixed.pt": sha256(BASELINE_PT),
        "pet-mad-m-v1.6.0.upstream.pt": sha256(UPSTREAM_PT),
    }
    for name, corr in CORRECTIONS.items():
        hashes[f"{name} model_best.pt"] = sha256(corr["pt"])
        hashes[f"{name} model.ckpt"] = sha256(corr["run"] / "model.ckpt")
    report["sha256"] = hashes
    for name, value in EXPECTED.items():
        checks.append((f"{name} sha256 is {value[:16]}", hashes[name] == value))
    for name, corr in CORRECTIONS.items():
        for t in TESTS:
            provenance = json.loads((DELTA_EXP / "reports" / "preds" / f"{corr['preds']}@{t}.provenance.json").read_text())
            checks.append((f"{name} correction model_best.pt equals the model of its stored {t} predictions",
                           provenance["model_best_pt_sha256"] == hashes[f"{name} model_best.pt"] and provenance["checkpoint_sha256"] == hashes[f"{name} model.ckpt"]))

    baseline = Potential(BASELINE_PT, BASELINE_OUTPUT, device)
    upstream = Potential(UPSTREAM_PT, BASELINE_OUTPUT, device)
    inner32 = Potential(FLOAT64 / "baseline_inner.float32.pt", BASELINE_OUTPUT, device)
    baseline64 = Potential(FLOAT64 / "baseline_inner.float64.pt", BASELINE_OUTPUT, device)
    deltas = {name: Potential(corr["pt"], DELTA_OUTPUT, device) for name, corr in CORRECTIONS.items()}
    deltas64 = {name: Potential(FLOAT64 / f"delta_{name}.float64.pt", DELTA_OUTPUT, device) for name in CORRECTIONS}

    capabilities = {}
    for name, potential in [("baseline", baseline)] + [(f"{n} correction", p) for n, p in deltas.items()]:
        cap = potential.model.capabilities()
        capabilities[name] = {"output": potential.output, "unit": cap.outputs[potential.output].unit, "length_unit": cap.length_unit,
                              "dtype": cap.dtype, "interaction_range_A": cap.interaction_range, "atomic_types": list(cap.atomic_types),
                              "neighbor_lists": [[o.cutoff, o.full_list] for o in potential.model.requested_neighbor_lists()]}
    report["capabilities"] = capabilities
    checks.append(("all models in eV and angstrom, float32, the same 102 atomic types",
                   all(c["unit"] == "eV" and c["length_unit"] == "angstrom" and c["dtype"] == "float32" and c["atomic_types"] == list(range(1, 103))
                       for c in capabilities.values())))

    lips = [lips_start(k) for k in range(5)]
    tests = {t: list(iread(str(TEST_XYZ[t]))) for t in TESTS}

    # patched against plain PET code, and the uncertainty wrapper against the bare PET, one structure per call
    agreement = []
    for label, atoms in [(f"lips_start{k}", a) for k, a in enumerate(lips)] + [(f"{t}:{i}", tests[t][i]) for t in TESTS for i in range(10)]:
        e_fixed, f_fixed = atoms_call(baseline, atoms)
        e_up, f_up = atoms_call(upstream, atoms)
        e_inner, f_inner = atoms_call(inner32, atoms)
        agreement.append({"structure": label, "natoms": len(atoms), "dE_upstream_eV": abs(e_fixed - e_up), "dF_upstream_eV_A": float(np.abs(f_fixed - f_up).max()),
                          "dE_bare_pet_eV": abs(e_fixed - e_inner), "dF_bare_pet_eV_A": float(np.abs(f_fixed - f_inner).max())})
    report["code_agreement"] = agreement
    worst_e = max(r["dE_upstream_eV"] / r["natoms"] for r in agreement)
    worst_f = max(r["dF_upstream_eV_A"] for r in agreement)
    checks.append((f"patched and plain PET exports of PET-MAD-M agree one structure at a time (max {worst_e:.1e} eV/atom, {worst_f:.1e} eV/A)",
                   worst_e < 1e-4 and worst_f < 1e-3))
    wrap_e = max(r["dE_bare_pet_eV"] / r["natoms"] for r in agreement)
    wrap_f = max(r["dF_bare_pet_eV_A"] for r in agreement)
    checks.append((f"the uncertainty wrapper leaves energy and forces equal to the bare PET within float32 precision (max {wrap_e:.1e} eV/atom, {wrap_f:.1e} eV/A)",
                   wrap_e < 1e-4 and wrap_f < 1e-3))

    # reference values on the five LiPS starts, and float32 against float64 for each combined potential
    lips_ref, precision = {}, []
    for k, atoms in enumerate(lips):
        e_b, f_b = atoms_call(baseline, atoms)
        e_b64, f_b64 = atoms_call(baseline64, atoms)
        lips_ref[f"E_baseline_{k}"], lips_ref[f"F_baseline_{k}"] = e_b, f_b
        row = {"start": k, "E_baseline_eV": e_b, "dE_baseline_32_64_eV": abs(e_b - e_b64), "max_F_baseline_eV_A": float(np.linalg.norm(f_b, axis=1).max())}
        for name in CORRECTIONS:
            e_d, f_d = atoms_call(deltas[name], atoms)
            e_d64, f_d64 = atoms_call(deltas64[name], atoms)
            lips_ref[f"E_delta_{name}_{k}"], lips_ref[f"F_delta_{name}_{k}"] = e_d, f_d
            row.update({f"E_delta_{name}_eV": e_d, f"E_combined_{name}_eV": e_b - e_d, f"dE_delta_{name}_32_64_eV": abs(e_d - e_d64),
                        f"dF_combined_{name}_32_64_eV_A": float(np.abs((f_b - f_d) - (f_b64 - f_d64)).max()),
                        f"max_F_delta_{name}_eV_A": float(np.linalg.norm(f_d, axis=1).max()),
                        f"max_F_combined_{name}_eV_A": float(np.linalg.norm(f_b - f_d, axis=1).max())})
        precision.append(row)
    np.savez(OUT / "lips_reference.npz", **lips_ref)
    report["lips_float32_vs_float64"] = precision
    for name in CORRECTIONS:
        worst = max(p[f"dF_combined_{name}_32_64_eV_A"] for p in precision)
        checks.append((f"{name}: float32 combined forces equal float64 combined forces on the LiPS starts (max {worst:.1e} eV/A)", worst < 5e-3))

    # each combined force against central differences of its combined energy, float64 copies of the same weights
    atoms = lips[0]
    rng = np.random.default_rng(SEED)
    picks = [(int(i), int(c)) for i, c in zip(rng.choice(len(atoms), FD_COMPONENTS, replace=False), rng.integers(0, 3, FD_COMPONENTS))]
    _, f0_b = atoms_call(baseline64, atoms)
    report["finite_differences"], report["finite_difference_max_by_step"] = {}, {}
    for name in CORRECTIONS:
        _, f0_d = atoms_call(deltas64[name], atoms)
        analytic = f0_b - f0_d
        rows = []
        for h in FD_STEPS:
            for i, c in picks:
                energies = []
                for sign in (1, -1):
                    moved = atoms.copy()
                    moved.positions[i, c] += sign * h
                    energies.append(atoms_call(baseline64, moved)[0] - atoms_call(deltas64[name], moved)[0])
                numeric = -(energies[0] - energies[1]) / (2 * h)
                rows.append({"h_A": h, "atom": i, "component": c, "analytic_eV_A": float(analytic[i, c]), "numeric_eV_A": float(numeric),
                             "difference_eV_A": float(abs(numeric - analytic[i, c]))})
        best = {h: max(r["difference_eV_A"] for r in rows if r["h_A"] == h) for h in FD_STEPS}
        report["finite_differences"][name] = rows
        report["finite_difference_max_by_step"][name] = best
        checks.append((f"{name}: combined forces equal central differences of the combined energy in float64 (max {min(best.values()):.1e} eV/A at the best step)",
                       min(best.values()) < 1e-4))

    # the evaluator reproduces the stored predictions of both corrections on both tests, then the static predictions
    STATIC.mkdir(parents=True, exist_ok=True)
    for t in TESTS:
        stored = {}
        for name, corr in CORRECTIONS.items():
            with open(DELTA_EXP / "reports" / "preds" / f"{corr['preds']}@{t}.xyz") as fh:
                stored[name] = list(iread(fh, format="extxyz"))
            assert len(stored[name]) == len(tests[t])
        gap = {name: [0.0, 0.0, 0.0] for name in CORRECTIONS}
        out = {"configuration_id": [], "natoms": [], "E_baseline": [], "F_baseline": []}
        for name in CORRECTIONS:
            out[f"E_delta_{name}"], out[f"F_delta_{name}"] = [], []
        for i, atoms in enumerate(tests[t]):
            e_b, f_b = atoms_call(baseline, atoms)
            out["configuration_id"].append(atoms.info["configuration_id"])
            out["natoms"].append(len(atoms))
            out["E_baseline"].append(e_b)
            out["F_baseline"].append(f_b)
            for name in CORRECTIONS:
                ref = stored[name][i]
                e_d, f_d = atoms_call(deltas[name], atoms)
                gap[name][0] = max(gap[name][0], float(np.abs(atoms.positions - ref.positions).max()))
                gap[name][1] = max(gap[name][1], abs(e_d - float(ref.info[DELTA_OUTPUT])) / len(atoms))
                gap[name][2] = max(gap[name][2], float(np.abs(f_d + np.array(ref.info[f"{DELTA_OUTPUT}_positions_gradients"])).max()))
                out[f"E_delta_{name}"].append(e_d)
                out[f"F_delta_{name}"].append(f_d)
        np.savez(STATIC / f"predictions_{t}.npz", configuration_id=np.array(out["configuration_id"]), natoms=np.array(out["natoms"]),
                 **{k: np.array(v) for k, v in out.items() if k.startswith("E_")},
                 **{k: np.concatenate(v) for k, v in out.items() if k.startswith("F_")})
        for name, (pos_gap, e_gap, f_gap) in gap.items():
            checks.append((f"{t}: one-structure evaluation of the {name} correction reproduces its stored predictions "
                           f"(positions {pos_gap:.1e} A, energy {e_gap:.1e} eV/atom, forces {f_gap:.1e} eV/A)",
                           pos_gap < 1e-5 and e_gap < 1e-5 and f_gap < 1e-3))
        report[f"{t}_static_predictions"] = {"structures": len(tests[t]), "file": str(STATIC / f"predictions_{t}.npz")}

    # energy reference of PET-MAD-M against the r2SCAN atomization labels, on sampled training structures only
    e0 = {int(z): float(v) for z, v in json.loads(E0_R2SCAN.read_text()).items()}
    train = list(iread(str(TRAIN_XYZ)))
    sample = np.random.default_rng(SEED).choice(len(train), TRAIN_SAMPLE, replace=False)
    offsets = {"atomization": {}, "total": {}}
    for i in sample:
        atoms = train[int(i)]
        energy, _ = atoms_call(baseline, atoms)
        label = float(atoms.info["atomization_energy"])
        source = atoms.info["r2scan_source"]
        offsets["atomization"].setdefault(source, []).append((energy - label) / len(atoms) * 1000.0)
        offsets["total"].setdefault(source, []).append((energy - sum(e0[int(z)] for z in atoms.numbers) - label) / len(atoms) * 1000.0)
    summary = {}
    for convention, by_source in offsets.items():
        summary[convention] = {}
        for source, values in by_source.items():
            v = np.array(values)
            summary[convention][source] = {"structures": len(v), "mean_meV_per_atom": float(v.mean()), "median_meV_per_atom": float(np.median(v)),
                                           "std_meV_per_atom": float(v.std()), "mae_meV_per_atom": float(np.abs(v).mean())}
    report["energy_reference_train_sample"] = {
        "rule": f"{TRAIN_SAMPLE} training structures drawn with seed {SEED}, per-atom offset of the PET-MAD-M energy from the r2SCAN atomization label, "
                "read as an atomization energy (energy minus label) or as a total energy (energy minus the MAD-1.6 r2SCAN isolated-atom energies minus label)",
        "by_convention_and_r2scan_source": summary,
    }

    passed = all(ok for _, ok in checks)
    report["checks"] = [{"check": c, "passed": ok} for c, ok in checks]
    (OUT / "calculator_checks.json").write_text(json.dumps(report, indent=1) + "\n")
    for c, ok in checks:
        print("PASS" if ok else "FAIL", c)
    print(json.dumps(report["finite_difference_max_by_step"]))
    print("CALCULATOR CHECKS", "PASSED" if passed else "FAILED")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
