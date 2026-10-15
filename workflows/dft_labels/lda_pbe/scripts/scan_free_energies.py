import glob
import os
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

PIPELINE = Path("${DEV_ROOT}/lda_pbe")
ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "references" / "free_energy_scan"
OUTPUT = ROOT / "references" / "free_energies.tsv"
WORKERS = 8

MEMBER = re.compile(r"(?:^|/)(frame_\d{5})/(pw-lda|pbe)/aims\.out$")
NATOMS = re.compile(r"\|\s*Number of atoms\s*:\s*(\d+)")
TOTAL = re.compile(r"\|\s*Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation\s*:\s*(\S+)\s*eV")
CORRECTED = re.compile(r"\|\s*Total energy corrected\s*:\s*(\S+)\s*eV")
FREE = re.compile(r"\|\s*Electronic free energy\s*:\s*(\S+)\s*eV")

HEADER = "arm\tkind\tframe\tfunctional\tconverged\tnatoms\te_total\te_corrected\te_free\ttar\n"


def last(pattern, text):
    found = pattern.findall(text)
    return found[-1].replace("E", "e") if found else ""


def scan_tar(tar_path):
    relative = os.path.relpath(tar_path, PIPELINE / "results")
    arm, kind = relative.split("/")[:2]
    shard = SHARDS / (relative.replace("/", "__") + ".tsv")
    if shard.exists():
        return relative, -1

    rows = []
    problem = ""
    try:
        with tarfile.open(tar_path, "r|") as tar:
            for member in tar:
                match = MEMBER.search(member.name)
                if not match:
                    continue
                text = tar.extractfile(member).read().decode(errors="replace")
                converged = int("Have a nice day." in text)
                rows.append("\t".join([arm, kind, match.group(1), match.group(2), str(converged),
                                       last(NATOMS, text), last(TOTAL, text), last(CORRECTED, text),
                                       last(FREE, text), relative]) + "\n")
    except (tarfile.TarError, EOFError, OSError) as error:
        problem = f"{type(error).__name__}: {error}"
        with open(SHARDS / "damaged_tars.log", "a") as log:
            log.write(f"{relative}\t{len(rows)} rows read\t{problem}\n")

    partial = shard.with_suffix(".part")
    partial.write_text("".join(rows))
    partial.rename(shard)
    return relative, len(rows) if not problem else -len(rows) - 2


def main():
    SHARDS.mkdir(parents=True, exist_ok=True)
    tars = sorted(glob.glob(str(PIPELINE / "results" / "*" / "*" / "*_outputs.tar"))
                  + glob.glob(str(PIPELINE / "results" / "*" / "*" / "chunks" / "*_outputs.tar")))
    tars = [t for t in tars if os.path.getsize(t) > 0]
    print(f"{len(tars)} tars, {WORKERS} workers", flush=True)

    with Pool(WORKERS) as pool:
        for done, (relative, n) in enumerate(pool.imap_unordered(scan_tar, tars), 1):
            state = "cached" if n == -1 else (f"{n} aims.out" if n >= 0 else f"DAMAGED after {-n - 2} aims.out")
            print(f"[{done}/{len(tars)}] {relative}: {state}", flush=True)

    with open(OUTPUT, "w") as out:
        out.write(HEADER)
        for shard in sorted(SHARDS.glob("*.tsv")):
            out.write(shard.read_text())
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
