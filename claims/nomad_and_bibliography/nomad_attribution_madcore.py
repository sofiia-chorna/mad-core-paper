import csv
import gzip
import json
import os

import numpy as np

CLAIMS = "${DEV_ROOT}/paper/claims"
DIR = f"{CLAIMS}/out/nomad_provenance"
SUB = f"{DIR}/madcore"
MAD_CORE = "${STORE_ROOT}/mad-core/release/mad-core"
OUT = f"{CLAIMS}/out/nomad_attribution_madcore.json"


def main():
    # the NOMAD structures of MAD-CORE, cut from the MAD-COSMO tables written by nomad_attribution.py
    csv.field_size_limit(10**9)
    os.makedirs(SUB, exist_ok=True)
    names = [n.decode() for n in np.load(f"{MAD_CORE}/release_directory_names.npy")]
    directory = np.load(f"{MAD_CORE}/release_directory.npy")
    row_in_directory = np.load(f"{MAD_CORE}/release_row_in_directory.npy")
    configuration_id = np.load(f"{MAD_CORE}/configuration_id.npy", mmap_mode="r")
    core_rows = np.flatnonzero(directory == names.index("collection/nomad"))
    core_of = {int(row_in_directory[i]): int(i) for i in core_rows}

    variants, uploads, dois = set(), set(), set()
    written = mp = mismatched = 0
    lines = {}
    with gzip.open(f"{DIR}/nomad_records.tsv.gz", "rt") as f:
        header = next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            core = core_of.get(int(c[0]))
            if core is None:
                continue
            mismatched += configuration_id[core].decode() != c[1]
            lines[core] = c
    with gzip.open(f"{SUB}/nomad_records_madcore.tsv.gz", "wt") as f:
        f.write("mad_core_row\t" + header)
        for core in sorted(lines):
            c = lines[core]
            f.write(f"{core}\t" + "\t".join(c) + "\n")
            written += 1
            mp += c[4] == "Materials Project"
            variants.add(c[8])
            uploads.add(c[3])
            dois.update(d.upper() for d in c[6].split(";") if d)

    def subset(name, column, keep, upper=False):
        with open(f"{DIR}/{name}") as src, open(f"{SUB}/{name.replace('.tsv', '_madcore.tsv')}", "w") as dst:
            header = next(src)
            dst.write(header)
            index = header.rstrip("\n").split("\t").index(column)
            n = 0
            for line in src:
                value = line.split("\t")[index]
                if (value.upper() if upper else value) in keep:
                    dst.write(line)
                    n += 1
        return n

    report = {
        "mad_core_rows_from_nomad": len(core_rows),
        "records_written": written,
        "configuration_id_mismatches": mismatched,
        "materials_project_records": mp,
        "credit_variants": subset("nomad_credit_variants.tsv", "credit_variant", variants),
        "uploads": subset("nomad_uploads.tsv", "nomad_upload_id", uploads),
        "dataset_dois": subset("nomad_dois.tsv", "dataset_doi", dois, upper=True),
    }
    assert written == len(core_rows) and mismatched == 0
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
