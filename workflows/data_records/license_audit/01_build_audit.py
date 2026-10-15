import csv
import pathlib

HERE = pathlib.Path(__file__).parent
SHIPPED = HERE / "dataset_metadata_shipped.tsv"
OUT = HERE / "audit_skeleton.tsv"

SPDX_MAP = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-4": "CC-BY-4.0",
    "mit": "MIT",
    "cc0-1": "CC0-1.0",
    "cc0": "CC0-1.0",
    "cc0-1.0": "CC0-1.0",
    "apache-2": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "nist-pd": "NIST-PD",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-3": "CC-BY-3.0",
    "cc-by-3.0": "CC-BY-3.0",
}

ALLOWED = {"CC-BY-4.0", "CC-BY-3.0", "CC0-1.0", "MIT", "BSD-3-Clause", "Apache-2.0", "NIST-PD"}


def normalize(raw):
    key = raw.strip().lower()
    if key in SPDX_MAP:
        return SPDX_MAP[key]
    if key.startswith("gdb terms"):
        return "GDB-terms(permission-2026-08-12)"
    return "UNRECOGNIZED:" + raw.strip()


def family(row):
    if row["colabfit_ds_id"].strip() or row["dataset_id"].startswith("colabfit_"):
        return "colabfit_hf"
    if row["dataset_id"].startswith("mcloudarchive__"):
        return "mcloud_api"
    return "manual_standalone"


def main():
    with open(SHIPPED, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 655, len(rows)

    out_fields = [
        "dataset_id", "dataset_group", "colabfit_ds_id", "rows", "title",
        "tsv_license", "spdx", "allowed_list", "check_family",
        "authority", "authority_license", "authority_ref", "ds_id_confirmed",
        "match", "status", "notes",
    ]

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()

        for row in rows:
            spdx = normalize(row["license"])
            writer.writerow({
                "dataset_id": row["dataset_id"].strip(),
                "dataset_group": row["dataset_group"].strip(),
                "colabfit_ds_id": row["colabfit_ds_id"].strip(),
                "rows": row["rows"].strip(),
                "title": row["title"].strip(),
                "tsv_license": row["license"].strip(),
                "spdx": spdx,
                "allowed_list": "yes" if spdx in ALLOWED else "no",
                "check_family": family(row),
                "authority": "",
                "authority_license": "",
                "authority_ref": "",
                "ds_id_confirmed": "",
                "match": "",
                "status": "PENDING",
                "notes": "",
            })

    fam_counts = {}
    for row in rows:
        fam_counts[family(row)] = fam_counts.get(family(row), 0) + 1
    print("rows:", len(rows), "families:", fam_counts)


main()
