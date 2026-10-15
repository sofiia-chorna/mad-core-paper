import csv
import gzip
import hashlib
import io
import json
import shutil
import sys
import tarfile
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

PHACE = Path("${DEV_ROOT}/phace-atomic")
EXPERIMENTS = PHACE / "mad-core-valid" / "experiments"
sys.path.insert(0, str(EXPERIMENTS / "delta_lda_pbe_10k" / "scripts"))
import build_sets as B

STORE = Path("${STORE_ROOT}/mad-core")
RUN = EXPERIMENTS / "delta_lda_pbe_18k"
RUN_STORE = STORE / "experiments" / "delta_lda_pbe_18k"
RANDOM_TEST = STORE / "r2scan_random_test"
COLLABORATOR = Path("${COLLABORATOR_STORE}/mad-v2.0")
BOTR0_XYZ = COLLABORATOR / "madcore" / "BotR-0.xyz"
BOTR0_INDEX = EXPERIMENTS / "delta_lda_18774" / "manifests" / "index" / "botr0_r2scan.tsv"
MONOMERS = COLLABORATOR / "monomers" / "lowest_isolated_energies.xyz"
ROUND1_ORIGINALS = ("BotR_3d/final_r2scan.xyz", "BotR_isolated/final_r2scan.xyz")
SPACE_EVAL = Path("${DEV_ROOT}/paper/figures/mad1.5_gfre_lfre_compare/random_test/results_random_test/r2scan_eval.json")
RECORD_FILES = PHACE / "matcloud-upload" / "xyz-2pow24-v2" / "files"

HERE = Path(__file__).resolve().parent
RECORD_BEFORE = HERE / "out" / "record_before_addition"
TEMPLATES = HERE / "templates"
ECHOES = HERE / "out" / "control_echoes"
PREVIOUS = HERE / "out" / "previous_dft_r2scan.extxyz.gz"
OUT = PHACE / "matcloud-upload" / "xyz-2pow24-v2" / "r2scan_addition" / "files"
CHECKS = HERE / "out" / "addition_checks.json"
LABEL_SOURCES = HERE / "out" / "r2scan_label_sources.tsv"

SPLITS = {"train": 18000, "val": 1000, "test_madcore": 1881, "test_random": 973}
RUN_NAMES = ("delta_lda--omat-s--madcore_shared18000--s42", "delta_pbe--omat-s--madcore_shared18000--s42")
ATOMIZATION_TOL = 1e-6
METADATA_KEYS = ("dataset_id", "dataset_group", "subset", "fps_order", "madcore_cell")
FUNCTIONALS = ("pw-lda", "pbe", "r2scan")
PBC_BY_CONTROL = {"periodic_ttt": "TTT", "surfaces_ttf": "TTT", "isolated_fff": "FFF"}

# run settings that change speed, output files or input checks but not the result
INFRASTRUCTURE = ("use_gpu", "gpu_", "output", "elsi_restart", "use_local_index", "override_kgrid_checks")
FIXED_SMEARING = ["occupation_type gaussian 0.05", "charge_mix_param 0.02"]
ADJUST_SCF = ["adjust_scf always 2", "adjust_scf_param lowgap occupation_width 0.05", "adjust_scf_param largegap occupation_width 0.05"]

SOURCE_FILES = {
    "botr": B.BOTR_XYZ,
    "mad16_train": B.MAD16["train"],
    "mad16_val": B.MAD16["val"],
    "mad16_test": B.MAD16["test"],
    "botr0": BOTR0_XYZ,
    "r2scan_random_test": RANDOM_TEST / "results" / "test_1000_r2scan.xyz",
}

# subsets of the MAD authors' calculations that ran without periodicity
ISOLATED_SUBSETS = {"madcore_isolated", "flashmd_isolated"}

# label_source names of the previous staged file
RENAMED_SOURCES = {
    "this_work": "madcore_work",
    "mad2_madcore_round1": "mad2_prep",
    "mad2_madnext_flashmd": "mad2_prep",
    "mad16_record": "mad16_record",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def write_tsv(path, rows, columns):
    with open(path, "w") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")


def read_gz(path):
    return read(io.StringIO(gzip.open(path, "rt").read()), index=":", format="extxyz")


def label_source(source):
    if source in ("botr", "botr0"):
        return "mad2_prep"
    if source.startswith("mad16"):
        return "mad16_record"
    return "madcore_work"


# the settings lines of a control.in, without comments, blank lines and run infrastructure
def settings(text):
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split("#")[0].split())
        if line and not line.startswith(INFRASTRUCTURE):
            lines.append(line)
    return sorted(lines)


def variant(text, control):
    lines = settings(text)
    if control == "isolated_fff":
        return sorted(line for line in lines if not line.startswith(("k_grid_density", "compute_analytical_stress")))
    if control == "surfaces_ttf":
        return sorted(lines + ["use_dipole_correction .true."])
    return lines


def adjusted(lines, init_iter=None):
    lines = [line for line in lines if line not in FIXED_SMEARING] + ADJUST_SCF
    if init_iter is not None:
        lines = [f"sc_init_iter {init_iter}" if line.startswith("sc_init_iter") else line for line in lines]
    return sorted(lines)


def check_templates(checks):
    templates = {functional: (TEMPLATES / f"control_{functional}.in").read_text() for functional in FUNCTIONALS}
    result = {}

    # the LDA and PBE variants reproduce every template of the archive before the addition
    with tarfile.open(RECORD_BEFORE / "files" / "dft_control.tar.gz") as archive:
        for member in archive.getmembers():
            control, name = member.name.split("/")
            functional = name[len("control_"):-len(".in")]
            original = archive.extractfile(member).read().decode()
            assert variant(templates[functional], control) == settings(original), member.name
            result[member.name] = "reproduced"

    # the r2SCAN variants reproduce the inputs echoed by FHI-aims in each campaign
    expected = {
        "this_work_periodic": variant(templates["r2scan"], "periodic_ttt"),
        "this_work_isolated": variant(templates["r2scan"], "isolated_fff"),
        "mad2_round1_periodic": variant(templates["r2scan"], "periodic_ttt"),
        "mad2_round1_isolated": variant(templates["r2scan"], "isolated_fff"),
        "mad2_madnext_flashmd_periodic": adjusted(variant(templates["r2scan"], "periodic_ttt")),
        "mad2_madnext_flashmd_isolated": adjusted(variant(templates["r2scan"], "isolated_fff")),
        "mad16_record": adjusted(variant(templates["r2scan"], "periodic_ttt"), init_iter=99),
    }
    for name, lines in expected.items():
        echoed = settings((ECHOES / f"{name}.in").read_text())
        assert echoed == lines, (name, sorted(set(echoed) ^ set(lines)))
        result[f"r2scan {name}"] = "reproduced by the documented variant"

    checks["settings_archive"] = result


def tie_to_runs(checks):
    build_sums = dict(line.split()[::-1] for line in (RUN / "manifests" / "CHECKSUMS.sha256").read_text().splitlines() if line.strip())
    store_sums = dict(line.split()[::-1] for line in (RUN_STORE / "SHA256SUMS").read_text().splitlines() if line.strip())
    tie = {}

    for split in SPLITS:
        current = sha256(RUN / "manifests" / "xyz" / f"{split}.xyz")
        build = next(value for key, value in build_sums.items() if key.endswith(f"xyz/{split}.xyz"))
        store = store_sums[f"manifests/xyz/{split}.xyz"]
        assert current == build == store, split
        tie[split] = {"sha256": current, "equals_build_checksum": True, "equals_aa002_archive": True}

    for split in ("test_madcore", "test_random"):
        for run in RUN_NAMES:
            provenance = json.loads((RUN / "reports" / "preds" / f"main-{run}@{split}.provenance.json").read_text())
            assert provenance["test_xyz_sha256"] == tie[split]["sha256"], (run, split)
        tie[split]["equals_evaluation_provenance_of_both_runs"] = True

    for run in RUN_NAMES:
        log = (RUN / "runs" / "main" / run / "train.log").read_text()
        assert "Dataset containing 18000 structures" in log and "Dataset containing 1000 structures" in log, run
        assert "delta_lda_pbe_shared/manifests/xyz/train.xyz" in (RUN / "runs" / "main" / run / "options.yaml").read_text(), run
    tie["train_logs_report_18000_and_1000"] = True

    checks["files_used_by_final_runs"] = tie


def write_control_archive(path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for functional in FUNCTIONALS:
            data = (TEMPLATES / f"control_{functional}.in").read_bytes()
            member = tarfile.TarInfo(f"control_{functional}.in")
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))

    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(buffer.getvalue())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    tie_to_runs(checks)
    check_templates(checks)

    references = {int(atoms.numbers[0]): atoms.get_potential_energy() for atoms in read(MONOMERS, index=":")}
    used_references = {int(z): float(value) for z, value in json.loads(B.E0_R2SCAN.read_text()).items()}
    assert references == used_references
    checks["isolated_atom_references"] = {"elements": len(references), "identical_to_references_used_for_targets": True}

    metadata, lda_pbe_control = {}, {}
    for functional in ("lda", "pbe"):
        for frame in read_gz(RECORD_FILES / f"dft_{functional}.extxyz.gz"):
            cid = frame.info["configuration_id"]
            metadata.setdefault(cid, set()).add(tuple(frame.info[key] for key in METADATA_KEYS))
            lda_pbe_control.setdefault(cid, set()).add(frame.info["control"])

    indexes = {"botr": read_tsv(B.INDEX / "botr_r2scan.tsv"), "botr0": read_tsv(BOTR0_INDEX)}
    for split in ("train", "val", "test"):
        indexes[f"mad16_{split}"] = read_tsv(B.INDEX / f"mad16_r2scan_{split}.tsv")
    by_hash = {source: {entry["hash"]: entry for entry in table} for source, table in indexes.items()}

    # round 1 labels are read from our reordered copy, the retrieval id points at the MAD authors' own files
    round1_original = {}
    for name in ROUND1_ORIGINALS:
        for position, frame in enumerate(B.read_all(COLLABORATOR / name)):
            round1_original.setdefault(B.frame_hash(frame), (name, position, frame))

    random_frames = B.read_all(SOURCE_FILES["r2scan_random_test"])
    random_position = {frame["info"]["configuration_id"]: position for position, frame in enumerate(random_frames)}
    results = read_tsv(RANDOM_TEST / "results" / "results.tsv")
    sample_rank = {row["configuration_id"]: int(row["sample_rank"]) for row in results}

    frames, manifest = [], []
    ids = {split: [] for split in SPLITS}
    hashes = {split: set() for split in SPLITS}
    worst = Counter()
    filled = set()
    control_pairs = Counter()

    for split, expected in SPLITS.items():
        rows = read_tsv(RUN / "manifests" / f"{split}.tsv")
        training = read(RUN / "manifests" / "xyz" / f"{split}.xyz", index=":")
        assert len(rows) == len(training) == expected, split

        for row, consumed in zip(rows, training):
            cid = row["configuration_id"]
            assert consumed.info["configuration_id"] == cid and consumed.info["r2scan_source"] == row["r2scan_source"]
            ids[split].append(cid)
            hashes[split].add(row["hash"])

            source = row["r2scan_source"]
            if source == "r2scan_random_test":
                position = random_position[cid]
                frame = random_frames[position]
                entry = {"idx": position, "subset": "", "configuration_id": f"frame_{sample_rank[cid]:04d}"}
            else:
                entry = by_hash[source][row["hash"]]
                frame = B.read_at(SOURCE_FILES[source], int(entry["offset"]))
            assert B.frame_hash(frame) == row["hash"], cid

            # the target the runs consumed, and the energy it was derived from
            atomization = float(consumed.info["atomization_energy"])
            energy = float(frame["info"]["energy"])
            worst["energy_minus_references_vs_target_eV"] = max(worst["energy_minus_references_vs_target_eV"], abs(energy - sum(references[int(z)] for z in frame["numbers"]) - atomization))
            assert worst["energy_minus_references_vs_target_eV"] <= ATOMIZATION_TOL, cid

            forces = consumed.get_forces()
            worst["source_vs_training_forces_eV_per_A"] = max(worst["source_vs_training_forces_eV_per_A"], float(np.abs(frame["forces"] - forces).max()))
            worst["source_vs_training_positions_A"] = max(worst["source_vs_training_positions_A"], float(np.abs(frame["positions"] - consumed.positions).max()))

            # the r2SCAN calculations followed the same classification and dipole rule as LDA and PBE
            assert len(lda_pbe_control[cid]) == 1, cid
            control = next(iter(lda_pbe_control[cid]))
            assert PBC_BY_CONTROL[control] == row["pbc"], cid
            if source in ("botr", "botr0"):
                assert (entry["subset"] in ISOLATED_SUBSETS) == (control == "isolated_fff"), cid
            control_pairs[control] += 1

            assert len(metadata[cid]) == 1, cid
            dataset_id, dataset_group, subset, fps_order, madcore_cell = next(iter(metadata[cid]))

            # the readme tells the two mad2_prep campaigns apart by dataset_id
            if source in ("botr", "botr0"):
                assert (dataset_id == "mad_next") == (source == "botr0"), cid

            info = {
                "configuration_id": cid,
                "dataset_id": dataset_id,
                "dataset_group": dataset_group,
                "subset": subset,
                "functional": "r2scan",
                "fps_order": int(fps_order),
                "madcore_cell": int(madcore_cell),
                "control": control,
                "split": split,
                "label_source": label_source(source),
                "energy": energy,
            }
            # the MAD-1.6 record omits free_energy in some frames, its authors confirmed it equals energy
            if "free_energy" in frame["info"]:
                info["free_energy"] = float(frame["info"]["free_energy"])
            else:
                assert label_source(source) == "mad16_record", cid
                info["free_energy"] = energy
                filled.add(cid)
            info["corrected_energy"] = atomization

            atoms = Atoms(numbers=consumed.numbers, positions=consumed.positions, cell=consumed.cell, pbc=consumed.pbc)
            atoms.info = info
            atoms.arrays["forces"] = forces
            frames.append(atoms)

            source_file, source_index = SOURCE_FILES[source].name, entry["idx"]
            if source == "botr":
                source_file, source_index, original = round1_original[row["hash"]]
                assert float(original["info"]["energy"]) == energy and np.abs(original["forces"] - frame["forces"]).max() == 0.0, cid
                worst["round1_original_files_found"] += 1

            manifest.append({
                "configuration_id": cid,
                "split": split,
                "label_source": label_source(source),
                "source_file": source_file,
                "source_frame_index": source_index,
                "source_native_id": entry.get("configuration_id", "") or "",
            })

    checks["counts"] = {split: len(values) for split, values in ids.items()}
    checks["counts"]["total"] = len(frames)
    checks["labels_vs_sources"] = dict(worst)
    checks["control_from_lda_pbe_classification"] = dict(control_pairs)

    for split in SPLITS:
        assert len(set(ids[split])) == len(ids[split]), split
    overlaps = {f"{a}|{b}": len(set(ids[a]) & set(ids[b])) for a, b in combinations(SPLITS, 2)}
    hash_overlaps = {f"{a}|{b}": len(hashes[a] & hashes[b]) for a, b in combinations(SPLITS, 2)}
    assert not any(overlaps.values()) and not any(hash_overlaps.values())
    checks["disjoint"] = {"by_configuration_id": overlaps, "by_geometry_hash": hash_overlaps}

    converged = {row["configuration_id"] for row in results if row["status"] == "converged"}
    failed = [row for row in results if row["status"] != "converged"]
    missing = read_tsv(RANDOM_TEST / "results" / "missing_frames.tsv")
    space_ids = set(json.loads(SPACE_EVAL.read_text())["per_structure"]["configuration_id"])
    random_ids = set(ids["test_random"])
    assert random_ids == converged == space_ids == set(random_position)
    assert {row["configuration_id"] for row in failed} == {row["configuration_id"] for row in missing} and len(missing) == 27
    assert not ({row["configuration_id"] for row in missing} & set(cid for values in ids.values() for cid in values))
    checks["random_test"] = {
        "exported": len(random_ids),
        "equals_converged_rows_of_results_tsv": True,
        "equals_space_evaluation_ids": True,
        "equals_aa002_labels_file": True,
        "failed": len(missing),
        "failed_absent_from_export": True,
    }

    text = io.StringIO()
    write(text, frames, format="extxyz")
    with gzip.GzipFile(OUT / "dft_r2scan.extxyz.gz", "wb", compresslevel=9, mtime=0) as handle:
        handle.write(text.getvalue().encode())

    # the source file and frame of every label stay internal
    columns = ["configuration_id", "split", "label_source", "source_file", "source_frame_index", "source_native_id"]
    write_tsv(LABEL_SOURCES, manifest, columns)

    shutil.copyfile(MONOMERS, OUT / "dft_isolated_atoms_r2scan.xyz")
    write_control_archive(OUT / "dft_control.tar.gz")

    # read everything back the way a user would and compare every field
    back = read_gz(OUT / "dft_r2scan.extxyz.gz")
    assert len(back) == len(frames)
    deviation = Counter()
    for written, loaded in zip(frames, back):
        assert loaded.info["configuration_id"] == written.info["configuration_id"]
        for key in ("dataset_id", "dataset_group", "subset", "functional", "fps_order", "madcore_cell", "control", "split", "label_source"):
            assert loaded.info[key] == written.info[key], key
        results_back = {**loaded.info, **loaded.calc.results}
        for key in ("energy", "free_energy", "corrected_energy"):
            if key in written.info:
                deviation[key] = max(deviation[key], abs(float(results_back[key]) - written.info[key]))
            else:
                assert key not in results_back, key
        deviation["forces"] = max(deviation["forces"], float(np.abs(loaded.get_forces() - written.arrays["forces"]).max()))
        deviation["positions"] = max(deviation["positions"], float(np.abs(loaded.positions - written.positions).max()))
        deviation["cell"] = max(deviation["cell"], float(np.abs(loaded.cell.array - written.cell.array).max()))
        assert (loaded.numbers == written.numbers).all() and (loaded.pbc == written.pbc).all()
    checks["read_back"] = {"frames": len(back), "max_abs_difference": dict(deviation)}

    # every value of the previous staged file is unchanged, control is the variant and the missing free_energy is energy
    previous = read_gz(PREVIOUS)
    assert len(previous) == len(back)
    renamed = Counter()
    relabelled = Counter()
    filled_back = set()
    manifest_back = read_tsv(LABEL_SOURCES)
    for old, new, entry in zip(previous, back, manifest_back):
        assert old.info["configuration_id"] == new.info["configuration_id"] == entry["configuration_id"]
        assert RENAMED_SOURCES[old.info["label_source"]] == new.info["label_source"] == entry["label_source"]
        assert old.info["split"] == new.info["split"] == entry["split"]
        assert old.positions.tobytes() == new.positions.tobytes() and old.cell.array.tobytes() == new.cell.array.tobytes()
        assert (old.numbers == new.numbers).all() and (old.pbc == new.pbc).all()
        assert old.get_forces().tobytes() == new.get_forces().tobytes()

        old_values = {**old.info, **old.calc.results}
        new_values = {**new.info, **new.calc.results}
        added = set(new_values) - set(old_values)
        assert set(old_values) <= set(new_values) and added <= {"free_energy"}
        for key in set(old_values) - {"control", "label_source"}:
            assert np.array_equal(old_values[key], new_values[key]), key

        if added:
            assert old_values["label_source"] == "mad16_record"
            assert new_values["free_energy"] == new_values["energy"] == old_values["energy"]
            filled_back.add(new.info["configuration_id"])

        renamed[f"{old.info['control']} -> {new.info['control']}"] += 1
        relabelled[f"{old.info['label_source']} -> {new.info['label_source']}"] += 1

    assert filled_back == filled and len(filled) == 634
    checks["unchanged_against_previous_staged_file"] = {
        "frames": len(previous),
        "control_renamed": dict(renamed),
        "label_source_renamed": dict(relabelled),
        "free_energy_set_to_energy": len(filled_back),
    }

    assert len({row["configuration_id"] for row in manifest_back}) == len(manifest_back) == len(back)
    with tarfile.open(OUT / "dft_control.tar.gz") as archive:
        assert sorted(member.name for member in archive.getmembers()) == sorted(f"control_{functional}.in" for functional in FUNCTIONALS)

    checks["composition"] = {
        "by_label_source": dict(Counter(atoms.info["label_source"] for atoms in back)),
        "by_control": dict(Counter(atoms.info["control"] for atoms in back)),
        "by_label_source_and_control": dict(Counter(f"{atoms.info['label_source']}|{atoms.info['control']}" for atoms in back)),
        "without_free_energy": sum(1 for atoms in frames if "free_energy" not in atoms.info),
        "free_energy_set_to_energy": len(filled),
        "atoms": int(sum(len(atoms) for atoms in back)),
    }

    files = ("dft_r2scan.extxyz.gz", "dft_isolated_atoms_r2scan.xyz", "dft_control.tar.gz")
    checks["files"] = {name: {"bytes": (OUT / name).stat().st_size, "sha256": sha256(OUT / name)} for name in files}
    checks["internal_files"] = {LABEL_SOURCES.name: {"bytes": LABEL_SOURCES.stat().st_size, "sha256": sha256(LABEL_SOURCES)}}

    CHECKS.write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
