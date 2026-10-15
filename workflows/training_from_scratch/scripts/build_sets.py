import json
import re
import sys
from pathlib import Path

from ase.data import atomic_numbers

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/scripts")
from build_sets import COLUMNS  # noqa: E402
from common import index_xyz, read_tsv, sha256, write_tsv  # noqa: E402

S = Path(__file__).resolve().parent.parent
V2 = S.parent / "scratch_scaling_v2" / "manifests"
PARENT = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/voronoi_weighted/pbe_20k/manifests")
MONOMERS = Path("${DEV_ROOT}/phace-atomic/mad-core-valid/dft/lda_pbe/references/monomers_pbe.xyz")
OUT = S / "manifests"
XYZ = OUT / "xyz"
SIZES = {"2k": 2000, "5k": 5000, "10k": 10000, "20k": 20000}
ARMS = ("madcore", "random")
SMOKE_ORDINARY = 500
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def ids(rows):
    return [r["configuration_id"] for r in rows]


# the 102 isolated atom reference frames: the monomer calculations the atomization energies were built from, rewritten
# with the training targets (atomization_energy 0, forces 0), non periodic, no cell, traceable by their free energy
def build_references():
    frames, rows = [], []
    with open(MONOMERS) as fh:
        rank = 0
        while True:
            n = fh.readline()
            if not n:
                break
            header = dict(KV.findall(fh.readline()))
            parts = fh.readline().split()
            assert int(n) == 1 and header["pbc"].strip('"') == "F F F" and "Lattice" not in header
            symbol = parts[0]
            energy = float(header["energy"])
            assert float(header["free_energy"]) == energy and all(float(x) == 0.0 for x in parts[4:7])
            cid = f"isolated_atom:{symbol}"
            frames.append(f'1\nconfiguration_id="{cid}" dataset_group="isolated_atom_reference" dataset_id="lda_pbe/references/monomers_pbe.xyz" natoms="1" vacuum_axes="-1" split_file="fff" '
                          f'arm="isolated_atom_reference" label_source="monomers_pbe" functional="pbe" energy={energy!r} free_energy={energy!r} atomization_energy=0.0 corrected_energy=0.0 '
                          f'Properties=species:S:1:pos:R:3:forces:R:3 pbc="F F F"\n{symbol:2s}       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000       0.00000000\n')
            row = {c: "" for c in COLUMNS}
            row.update({"rank": rank, "configuration_id": cid, "frozen_configuration_id": cid, "arm": "isolated_atom_reference", "dataset_group": "isolated_atom_reference", "dataset_id": "monomers_pbe",
                        "natoms": 1, "geometry_class": "isolated", "vacuum_axes": -1, "n_species": 1, "elements": symbol, "e_pbe": repr(energy), "atomization_e_pbe": "0.0", "fmax_pbe": "0.0"})
            rows.append(row)
            rank += 1
    assert len(frames) == 102 and len({r["elements"] for r in rows}) == 102
    assert sorted(atomic_numbers[r["elements"]] for r in rows) == list(range(1, 103))
    write_tsv(OUT / "isolated_atom_references.tsv", rows, COLUMNS)
    (XYZ / "isolated_atom_references.xyz").write_text("".join(frames))
    return rows


def main():
    XYZ.mkdir(parents=True, exist_ok=True)
    references = build_references()
    ref_xyz = (XYZ / "isolated_atom_references.xyz").read_bytes()
    record = {"source_round2_manifests": str(V2), "source_round2_split_sha256": sha256(V2 / "SPLIT.json"), "monomers_pbe_xyz": str(MONOMERS), "monomers_pbe_sha256": sha256(MONOMERS),
              "references": {"structures": 102, "atoms": 102, "manifest_sha256": sha256(OUT / "isolated_atom_references.tsv"), "xyz_sha256": sha256(XYZ / "isolated_atom_references.xyz"),
                             "targets": "atomization_energy 0.0 eV and zero forces for every frame, applied to the complete prediction (composition baseline included)",
                             "convention": "the monomer free energies are the per element E0 subtracted in every training atomization energy (merge_labels_staged_20261005.py, TRAINING_HANDOFF.md section 3), so the zero target is exact by construction"},
              "sets": {}, "duplicates": {}, "checks": {}}
    v2_split = json.load(open(V2 / "SPLIT.json"))
    holdouts = {name: read_tsv(PARENT / f"{name}.tsv") for name in ("val", "test_random", "test_madcore")}
    holdouts["val_madcore_250"] = read_tsv(V2 / "val_madcore_250.tsv")
    for name, rows in holdouts.items():
        record["checks"][f"{name}_nonperiodic_single_atoms"] = sum(1 for r in rows if r["natoms"] == "1" and r["geometry_class"] == "isolated")
        record["checks"][f"{name}_ids_colliding_with_references"] = len({r["configuration_id"] for r in rows} & {r["configuration_id"] for r in references})
    species_all = set()
    for arm in ARMS:
        order = read_tsv(V2 / f"{arm}_20k.tsv")
        assert ids(read_tsv(V2 / f"{arm}_5k.tsv")) == ids(order)[:5000] and ids(read_tsv(V2 / f"{arm}_10k.tsv")) == ids(order)[:10000]
        index = index_xyz(V2 / "xyz" / f"{arm}_20k.xyz")
        source = (V2 / "xyz" / f"{arm}_20k.xyz").read_bytes()
        offsets = sorted(index.values()) + [len(source)]
        end_of = {cid: offsets[offsets.index(off) + 1] for cid, off in index.items()}
        for size, n in SIZES.items():
            rows = order[:n]
            name = f"{arm}_{size}"
            ordinary_xyz = b"".join(source[index[r["configuration_id"]]:end_of[r["configuration_id"]]] for r in rows)
            if size in ("5k", "10k", "20k"):
                assert sha256_bytes(ordinary_xyz) == sha256(V2 / "xyz" / f"{name}.xyz"), f"{name} ordinary frames differ from round 2"
            write_tsv(OUT / f"{name}_ordinary.tsv", rows, COLUMNS)
            aug_rows = rows + [{**r, "rank": len(rows) + i} for i, r in enumerate(references)]
            write_tsv(OUT / f"{name}_aug.tsv", aug_rows, COLUMNS)
            (XYZ / f"{name}_aug.xyz").write_bytes(ordinary_xyz + ref_xyz)
            species_before = {s for r in rows for s in r["elements"].split()}
            species_all |= species_before
            dup = [(r["configuration_id"], r["elements"]) for r in rows if r["natoms"] == "1" and r["geometry_class"] == "isolated"]
            record["duplicates"][name] = {"ordinary_nonperiodic_single_atoms_also_present_as_reference": dup}
            atoms_ordinary = sum(int(r["natoms"]) for r in rows)
            record["sets"][name] = {"ordinary_structures": n, "added_references": 102, "total_structures": n + 102, "atoms_ordinary": atoms_ordinary, "atoms_total": atoms_ordinary + 102,
                                    "elements_before": len(species_before), "elements_after": 102, "steps_per_epoch_batch_64_drop_last": (n + 102) // 64,
                                    "ordering": v2_split["sets"][f"{arm}_{size}" if size != "2k" else f"{arm}_5k"].get("ordering", "round 2 ordering"), "prefix_of": f"{arm}_20k (round 2)",
                                    "manifest_sha256": sha256(OUT / f"{name}_aug.tsv"), "xyz_sha256": sha256(XYZ / f"{name}_aug.xyz"), "ordinary_manifest_sha256": sha256(OUT / f"{name}_ordinary.tsv")}
            print(f"{name:12s} ordinary {n:6d} ({atoms_ordinary:7d} atoms) + 102 references, elements {len(species_before):3d} -> 102, steps/epoch {(n + 102) // 64}, isolated duplicates {dup}")
    # smoke set: the first ordinary structures of madcore_2k plus the references, so every batch mixes both
    smoke_rows = read_tsv(V2 / "madcore_5k.tsv")[:SMOKE_ORDINARY]
    index = index_xyz(V2 / "xyz" / "madcore_5k.xyz")
    source = (V2 / "xyz" / "madcore_5k.xyz").read_bytes()
    offsets = sorted(index.values()) + [len(source)]
    smoke_xyz = b"".join(source[index[r["configuration_id"]]:offsets[offsets.index(index[r["configuration_id"]]) + 1]] for r in smoke_rows)
    (XYZ / "smoke_mixed.xyz").write_bytes(smoke_xyz + ref_xyz)
    write_tsv(OUT / "smoke_mixed.tsv", smoke_rows + [{**r, "rank": len(smoke_rows) + i} for i, r in enumerate(references)], COLUMNS)
    record["sets"]["smoke_mixed"] = {"ordinary_structures": SMOKE_ORDINARY, "added_references": 102, "total_structures": SMOKE_ORDINARY + 102, "steps_per_epoch_batch_64_drop_last": (SMOKE_ORDINARY + 102) // 64, "xyz_sha256": sha256(XYZ / "smoke_mixed.xyz")}
    numbers = sorted(atomic_numbers[s] for s in species_all | {r["elements"] for r in references})
    assert numbers == list(range(1, 103))
    record["atomic_types"] = numbers
    record["checks"]["prefix_nesting"] = "2k subset of 5k subset of 10k subset of 20k by configuration_id order, both arms, verified"
    record["checks"]["ordinary_frames_identical_to_round2"] = True
    record["checks"]["eligibility_verification"] = str(S.parent / "scratch_scaling" / "reports" / "audit" / "final_mae_10k" / "DIAGNOSTIC.md") + " section 3 (pair rule, fmax <= 100 both functionals, zero violations, all round 2 sets)"
    record["validation"] = {"combined_xyz": str(V2 / "xyz" / "val_combined_1250.xyz"), "combined_xyz_sha256": sha256(V2 / "xyz" / "val_combined_1250.xyz"), "val_madcore_250_sha256": sha256(V2 / "val_madcore_250.tsv"),
                            "native_aggregation": v2_split["validation"]["native_aggregation"]}
    json.dump(record, open(OUT / "SETS.json", "w"), indent=1)
    with open(OUT / "CHECKSUMS.sha256", "w") as fh:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name != "CHECKSUMS.sha256":
                fh.write(f"{sha256(p)}  {p.relative_to(OUT)}\n")
    print("duplicates:", {k: v for k, v in record["duplicates"].items() if v["ordinary_nonperiodic_single_atoms_also_present_as_reference"]})
    print("holdout checks:", {k: v for k, v in record["checks"].items() if "single" in k or "colliding" in k})
    print("DONE")


def sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
