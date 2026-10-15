import csv
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("${STORE_ROOT}/mad-core/lda_pbe")
FINAL = ROOT / "final"
OUTPUTS = ROOT / "dft_outputs"
MEMBERS = Path("${SCRATCHPAD}/outputs_index/members.tsv")
OUT = FINAL / "dft_outputs_index.tsv"
SUMMARY = FINAL / "dft_outputs_index_summary.md"

ARMS = ("madcore_25k", "random_25k", "voronoi_25k", "random_test_6k")
FILES = ("aims.out", "rs_hamiltonian.h5", "rs_overlap.h5")
FUNCTIONAL_COL = {"pbe": "pbe", "pw-lda": "lda"}
MEMBER = re.compile(r"(?:\./)?(?:block_\d+/)?frame_(\d+)/(pbe|pw-lda)/(aims\.out|rs_hamiltonian\.h5|rs_overlap\.h5)$")
LOOSE_DIRS = ("from_kuma/frames", "rerun_hamiltonians_20260914", "clariden_handoff/preserved_lda_20260929", "clariden_handoff/pilot_20260923/preserved_lda")


# arm and kind of a record from where its tar or loose file sits
def arm_kind(rel):
    parts = rel.split("/")
    if parts[0] == "results":
        return parts[1], parts[2]
    if parts[0] in ("epfl_deliveries", "clariden_handoff", "scratch_rescue_20260925"):
        return "voronoi_25k", "periodic_ttt"
    if parts[0] == "from_kuma":
        return parts[2], parts[3]
    if parts[0] == "rerun_hamiltonians_20260914":
        return parts[1], parts[2]
    raise ValueError(rel)


def main():
    locations = defaultdict(list)
    for line in open(MEMBERS):
        tar, member = line.rstrip("\n").split("\t")
        m = MEMBER.search(member)
        if not m:
            continue
        rel = str(Path(tar).relative_to(OUTPUTS))
        arm, kind = arm_kind(rel)
        locations[(arm, kind, int(m.group(1)), m.group(2), m.group(3))].append(f"{rel}::{member}")

    loose = subprocess.run(["find"] + [str(OUTPUTS / d) for d in LOOSE_DIRS] + ["-type", "f", "(", "-name", "aims.out", "-o", "-name", "rs_hamiltonian.h5", "-o", "-name", "rs_overlap.h5", ")"],
                           capture_output=True, text=True, check=True).stdout.split()
    for path in loose:
        rel = str(Path(path).relative_to(OUTPUTS))
        m = MEMBER.search(rel)
        if not m:
            continue
        arm, kind = arm_kind(rel)
        locations[(arm, kind, int(m.group(1)), m.group(2), m.group(3))].append(rel)
    print(f"{len(locations)} (arm, kind, frame, functional, file) records located, {len(loose)} loose files")

    rows = []
    coverage = Counter()
    for arm in ARMS:
        with open(FINAL / arm / "manifest.tsv") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                kind = r["dft_bucket"]
                frame = int(r["frame_id"]) if r["frame_id"] != "" else None
                for functional, col in FUNCTIONAL_COL.items():
                    labelled = r[f"has_{col}"] == "1"
                    if not labelled and frame is None:
                        continue
                    locs = {f: locations.get((arm, kind, frame, functional, f), []) if frame is not None else [] for f in FILES}
                    if r["label_source"] == "madcore_25k" and arm == "voronoi_25k":
                        locs = {f: locations.get(("madcore_25k", "madcore_25k", frame, functional, f), []) for f in FILES}
                    rows.append({
                        "configuration_id": r["configuration_id"], "arm": arm, "label_source": r["label_source"], "kind": kind,
                        "frame_id": "" if frame is None else f"frame_{frame:05d}", "functional": functional, "labelled": int(labelled),
                        "aims_out": ";".join(locs["aims.out"]), "rs_hamiltonian": ";".join(locs["rs_hamiltonian.h5"]), "rs_overlap": ";".join(locs["rs_overlap.h5"]),
                    })
                    if labelled:
                        coverage[(arm, functional, "label")] += 1
                        coverage[(arm, functional, "with_hamiltonian")] += bool(locs["rs_hamiltonian.h5"])
                        coverage[(arm, functional, "with_aims_out")] += bool(locs["aims.out"])
                    else:
                        coverage[(arm, functional, "unlabelled_with_aims_out")] += bool(locs["aims.out"])

    with open(OUT, "w") as fh:
        cols = ["configuration_id", "arm", "label_source", "kind", "frame_id", "functional", "labelled", "aims_out", "rs_hamiltonian", "rs_overlap"]
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    lines = ["# dft_outputs_index.tsv, coverage summary", "",
             "One row per (structure, functional). `aims_out`, `rs_hamiltonian`, `rs_overlap` give every location of",
             "that file as `<tar relative to lda_pbe/dft_outputs>::<member>` or a loose path relative to `dft_outputs/`,",
             "`;`-separated when a record exists in more than one place (reruns, rescues). Empty = not archived.",
             "Built 2026-10-05 from the tar headers of every `*.tar` under `dft_outputs/` (density_matrices and the",
             "benchmark excluded) plus the loose frame directories of from_kuma, rerun_hamiltonians_20260914 and the",
             "preserved LDA sets. MAD-CORE labels inherited by the shared Voronoi structures are looked up under madcore_25k.", "",
             "| arm | functional | labelled | with aims.out | with rs_hamiltonian.h5 | unlabelled records with an aims.out |",
             "|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for functional in FUNCTIONAL_COL:
            c = coverage
            lines.append(f"| {arm} | {functional} | {c[(arm, functional, 'label')]} | {c[(arm, functional, 'with_aims_out')]} | "
                         f"{c[(arm, functional, 'with_hamiltonian')]} | {c[(arm, functional, 'unlabelled_with_aims_out')]} |")
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[9:]))
    print(f"wrote {OUT} ({len(rows)} rows) and {SUMMARY}")


if __name__ == "__main__":
    main()
