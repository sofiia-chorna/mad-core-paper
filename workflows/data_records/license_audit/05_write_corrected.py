import csv
import pathlib

HERE = pathlib.Path(__file__).parent

SPDX_MAP = {
    "cc-by-4.0": "CC-BY-4.0", "cc-by-4": "CC-BY-4.0",
    "mit": "MIT",
    "cc0-1": "CC0-1.0", "cc0": "CC0-1.0",
    "apache-2": "Apache-2.0",
    "bsd-3-clause": "BSD-3-Clause",
    "nist-pd": "NIST-PD",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-3": "CC-BY-3.0",
    "license addendum,cc-by-4.0": "CC-BY-4.0",
}

GDB13_LICENSE = ("GDB terms of use (gdb.unibe.ch): citation required; no patent use; "
                 "redistribution covered by written permission of J.-L. Reymond (email, 2026-08-12)")

NOMAD_FILLS = {
    "title": "NOMAD repository (aggregated public entries)",
    "upstream_url": "https://nomad-lab.eu/",
}

MAD_TIER_IDS = {
    "mad15": ["binary_random", "dimers", "mc2d", "mc3d", "mc3d_cluster", "mc3d_ext",
              "mc3d_random", "mc3d_random_ext", "mc3d_rattled", "mc3d_surface", "monomers",
              "shiftml_molcrys", "shiftml_molfrags", "trimers"],
    "mad_next": ["dimers", "flashmd_2d", "flashmd_3d", "flashmd_isolated", "madcat",
                 "newtrimers", "trimers"],
    "mad_sol": ["solvated_ions", "solvents_interface", "solvents_soup"],
}


def main():
    with open(HERE / "dataset_metadata_shipped.tsv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)

    row_total = 0
    license_rows = {}
    for row in rows:
        raw = row["license"].strip()
        key = raw.lower()

        if row["dataset_id"].strip() == "gdb13":
            row["license"] = GDB13_LICENSE
        elif key in SPDX_MAP:
            row["license"] = SPDX_MAP[key]
        else:
            raise SystemExit(f"unmapped license for {row['dataset_id']}: {raw}")

        if row["dataset_id"].strip() == "nomad":
            for col, val in NOMAD_FILLS.items():
                if not row[col].strip():
                    row[col] = val

        n = int(row["rows"])
        row_total += n
        tag = "GDB-permission" if row["dataset_id"].strip() == "gdb13" else row["license"]
        license_rows[tag] = license_rows.get(tag, [0, 0])
        license_rows[tag][0] += 1
        license_rows[tag][1] += n

    assert row_total == 1_693_646_343, row_total

    with open(HERE / "dataset_metadata_corrected.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)

    mad_fields = ["dataset_id", "dataset_group", "colabfit_ds_id", "rows", "source_kind",
                  "store_subfolder", "title", "doi", "upstream_url", "source_publication", "license"]
    with open(HERE / "mad_tier_metadata_proposed.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=mad_fields, delimiter="\t", lineterminator="\r\n")
        writer.writeheader()
        for group, subsets in MAD_TIER_IDS.items():
            for sub in subsets:
                writer.writerow({
                    "dataset_id": f"{group}:{sub}",
                    "dataset_group": group,
                    "colabfit_ds_id": "",
                    "rows": "",
                    "source_kind": "mad_tier",
                    "store_subfolder": "",
                    "title": f"MAD {group} subset {sub} (own data)",
                    "doi": "",
                    "upstream_url": "",
                    "source_publication": "",
                    "license": "CC-BY-4.0 (proposed, confirm)",
                })

    print(f"{'license':55s} {'datasets':>8s} {'rows':>15s} {'share':>7s}")
    for tag, (n_ds, n_rows) in sorted(license_rows.items(), key=lambda kv: -kv[1][1]):
        print(f"{tag[:55]:55s} {n_ds:8d} {n_rows:15,d} {100 * n_rows / row_total:6.2f}%")


main()
