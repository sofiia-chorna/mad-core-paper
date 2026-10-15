# re-derives every value of the four SI tables and the stated numbers that come from these
# files, and compares them with the submitted text. inputs are the source files in data/
# (copied from the experiment reports, see provenance/extracted_files.tsv); nothing is
# recomputed from predictions here, that is level 2 (workflows/)
import argparse
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# SI Table training-set sizes, structures and ordinary atoms, as printed in si.tex
TRAINING_SET_SIZE = {
    ("MAD-CORE", 5000): 32973, ("MAD-CORE", 10000): 63653, ("MAD-CORE", 20000): 120468,
    ("RANDOM", 5000): 186211, ("RANDOM", 10000): 369718, ("RANDOM", 20000): 730646,
}
# SI Table finetune, balanced 4,000 test, energy meV/atom and force meV/A, as printed
FINETUNE = {
    "scratch": {"E_mae": 176.7, "E_rmse": 262.8, "F_mae": 583, "F_rmse": 1008, "E_p8": 1079.6, "F_p8": 5253, "E_sup": 3007.2, "F_sup": 22173},
    "finetune_aligned": {"E_mae": 90.0, "E_rmse": 143.7, "F_mae": 191, "F_rmse": 360, "E_p8": 585.9, "F_p8": 5717, "E_sup": 1464.7, "F_sup": 24924},
}
# SI Table delta-learning, test_madcore and test_random, as printed
DELTA = {
    ("test_madcore", "direct"): (143.2, 209.0, 382, 903, 629.6, 5905, 1483.3, 17858),
    ("test_madcore", "lda"): (28.5, 58.8, 62, 153, 479.5, 1308, 1227.8, 4281),
    ("test_madcore", "pbe"): (24.0, 56.7, 51, 133, 464.8, 1221, 1176.3, 3958),
    ("test_random", "direct"): (61.6, 78.0, 266, 391, 202.9, 1163, 431.5, 4311),
    ("test_random", "lda"): (20.3, 24.1, 41, 60, 80.6, 252, 174.0, 1028),
    ("test_random", "pbe"): (6.2, 12.9, 24, 36, 70.6, 220, 159.4, 889),
}
# SI Table accrete-levels: size, from MAD-NEXT, coverage. the margin and time columns of the
# printed table come from the deleted run log and have no source file
ACCRETE = {16: (1, 9.186), 256: (8, 3.833), 4096: (224, 1.939), 65536: (11785, 0.941),
           262144: (38355, 0.710), 524288: (62967, 0.617), 856012: (86910, 0.556)}
# Table I of the main text, DFT label counts of the MAD-CORE record
RECORD_DFT_COUNTS = {"lda": 79241, "pbe": 77383, "r2scan": 21854}


def close(value, printed, decimals):
    return round(float(value), decimals) == round(float(printed), decimals)


def check_training_set_sizes(data, failures):
    sets = json.load(open(data / "training_sets_SETS.json"))["sets"]
    for (arm, size), atoms in TRAINING_SET_SIZE.items():
        key = f"{'madcore' if arm == 'MAD-CORE' else 'random'}_{size // 1000}k"
        found = sets[key]["atoms_ordinary"]
        print(f"training-set size {arm} {size}: file {found}, printed {atoms}")
        if found != atoms or sets[key]["ordinary_structures"] != size or sets[key]["added_references"] != 102:
            failures.append(f"training set {key}")


def check_finetune(data, failures):
    rows = {}
    for line in (data / "finetune_metrics.tsv").read_text().splitlines()[1:]:
        fields = line.split("\t")
        if fields[2] == "balanced_4000":
            rows[fields[0]] = dict(zip(["E_mae", "E_rmse", "E_p8", "E_sup", "F_mae", "F_rmse", "F_p8", "F_sup"], map(float, fields[6:14])))
    for model, printed in FINETUNE.items():
        for key, value in printed.items():
            decimals = 1 if key.startswith("E") else 0
            ok = close(rows[model][key], value, decimals)
            print(f"finetune {model} {key}: file {rows[model][key]:.3f}, printed {value}{'' if ok else '  MISMATCH'}")
            if not ok:
                failures.append(f"finetune {model} {key}")


def check_delta(data, failures):
    # both metrics files hold results[test view][model row]{E, F}{mae, rmse, p8, sup}; the
    # corrected rows are in the delta file, the direct row in the direct file
    corrections = json.load(open(data / "delta_lda_pbe_18k_metrics.json"))["results"]
    direct = json.load(open(data / "direct_r2scan_18k_metrics.json"))["results"]
    rows = {"direct": (direct, "direct r2SCAN (standalone)"),
            "lda": (corrections, "LDA + learned delta_LDA"),
            "pbe": (corrections, "PBE + learned delta_PBE")}
    for (test, model), printed in DELTA.items():
        source, label = rows[model]
        row = source[test][label]
        values = (row["E"]["mae"], row["E"]["rmse"], row["F"]["mae"], row["F"]["rmse"], row["E"]["p8"], row["F"]["p8"], row["E"]["sup"], row["F"]["sup"])
        decimals = (1, 1, 0, 0, 1, 0, 1, 0)
        for name, value, want, d in zip(("E_mae", "E_rmse", "F_mae", "F_rmse", "E_p8", "F_p8", "E_sup", "F_sup"), values, printed, decimals):
            ok = close(value, want, d)
            print(f"delta {test} {model} {name}: file {float(value):.3f}, printed {want}{'' if ok else '  MISMATCH'}")
            if not ok:
                failures.append(f"delta {test} {model} {name}")


def check_accrete(data, failures):
    levels = json.load(open(data / "levels.json"))["levels"]
    by_size = {level["size"]: level for level in levels}
    for size, (anchor, coverage) in ACCRETE.items():
        level = by_size[size]
        ok = level["from_anchor"] == anchor and close(level["coverage_runtime_float32_construction_log"], coverage, 3)
        print(f"accrete level {size}: from anchor {level['from_anchor']} (printed {anchor}), "
              f"cov {level['coverage_runtime_float32_construction_log']} (printed {coverage}), "
              f"float64 separation of the released features {level['separation_release_features']:.4f}{'' if ok else '  MISMATCH'}")
        if not ok:
            failures.append(f"accrete level {size}")
    print("accrete table: the margin and t(s) columns have no source file (run log deleted), not checked")


def check_record(data, failures, record_dir=None):
    # the frame counts of the three DFT label files of the MAD-CORE record; counted from the
    # extxyz files themselves when --record-dir points at a copy of the record
    index = json.load(open(data / "madcore_record_index.json"))["files"]
    for functional, count in RECORD_DFT_COUNTS.items():
        name = f"dft_{functional}.extxyz.gz"
        entry = index.get(name, {})
        line = f"record {name}: {entry.get('bytes')} bytes in the record index, printed {count} structures"
        if record_dir is None:
            print(line + ", not counted (no --record-dir)")
            continue
        frames = 0
        with gzip.open(Path(record_dir) / name, "rt") as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                natoms = int(header)
                fh.readline()
                for _ in range(natoms):
                    fh.readline()
                frames += 1
        ok = frames == count
        print(line + f", counted {frames}{'' if ok else '  MISMATCH'}")
        if not ok:
            failures.append(f"record {name} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--record-dir", default=None, help="folder holding the dft_*.extxyz.gz files of the MAD-CORE record, to count their structures")
    args = parser.parse_args()
    data = Path(args.data)
    failures = []
    check_training_set_sizes(data, failures)
    check_finetune(data, failures)
    check_delta(data, failures)
    check_accrete(data, failures)
    check_record(data, failures, args.record_dir)
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))
    print("all table values agree with their source files")


if __name__ == "__main__":
    main()
