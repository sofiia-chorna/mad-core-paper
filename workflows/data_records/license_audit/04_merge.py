import csv
import pathlib

HERE = pathlib.Path(__file__).parent

HF_NORM = {"cc-by-4.0": "CC-BY-4.0", "mit": "MIT", "cc0-1.0": "CC0-1.0",
           "bsd-3-clause": "BSD-3-Clause", "apache-2.0": "Apache-2.0", "cc-by-3.0": "CC-BY-3.0"}

MC_NORM = {"cc-by-4.0": "CC-BY-4.0", "cc-by-4.0;": "CC-BY-4.0",
           "cc-by-sa-4.0": "CC-BY-SA-4.0", "mit": "MIT"}

FAIRCHEM_OC20 = "https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md"
FAIRCHEM_OC20NEB = "https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20neb.md"

MANUAL = {
    "Alexandria_pbesol": ("CC-BY-4.0", "alexandria.icams.rub.de", "https://alexandria.icams.rub.de/about.html",
                          "site states: available under the terms of the Creative Commons Attribution 4.0 License"),
    "Alexandria_pressure": ("CC-BY-4.0", "alexandria.icams.rub.de", "https://alexandria.icams.rub.de/about.html",
                            "site states: available under the terms of the Creative Commons Attribution 4.0 License"),
    "OC20_S2EF_train_all": ("CC-BY-4.0", "fairchem docs", FAIRCHEM_OC20,
                            "OC20 dataset is licensed under a Creative Commons Attribution 4.0 License"),
    "oc20neb": ("CC-BY-4.0", "fairchem docs", FAIRCHEM_OC20NEB,
                "License: CC-BY-4.0 in the official OC20NEB dataset doc"),
    "OC25_train": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/OC25",
                   "The OC25 dataset is provided under a CC-BY-4.0 license (checkpoints: FAIR Chemistry License)"),
    "OC25_val": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/OC25",
                 "The OC25 dataset is provided under a CC-BY-4.0 license (checkpoints: FAIR Chemistry License)"),
    "ODAC_train": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/ODAC25",
                   "The ODAC25 dataset is provided under a CC-BY-4.0 License (checkpoints: FAIR Chemistry License)"),
    "ODAC_val": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/ODAC25",
                 "The ODAC25 dataset is provided under a CC-BY-4.0 License (checkpoints: FAIR Chemistry License)"),
    "OMol25_train": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/OMol25",
                     "The OMol25 dataset is provided under a CC-BY-4.0 License (checkpoints: FAIR Chemistry License)"),
    "OMol25_val": ("CC-BY-4.0", "Meta HF page", "https://huggingface.co/facebook/OMol25",
                   "The OMol25 dataset is provided under a CC-BY-4.0 License (checkpoints: FAIR Chemistry License)"),
    "nomad": ("CC-BY-4.0", "NOMAD terms of use", "https://nomad-lab.eu/nomad-lab/terms.html",
              "content uploaded to NOMAD is made available subject to the rules of the CC Attribution 4.0 License"),
    "oqmd__oqmd": ("CC-BY-4.0", "oqmd.org", "https://www.oqmd.org/",
                   "site states data is licensed under CC-BY 4.0 with citation"),
    "gdb13": ("GDB-terms(permission-2026-08-12)", "written permission", "https://gdb.unibe.ch/",
              "email J.-L. Reymond 2026-08-12: yes, go ahead (redistribution of generated 3D coordinates)"),
    "mcloud__mc2d": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/17gf6-84915",
                     "The Materials Cloud 2D database (MC2D): rights cc-by-4.0"),
    "mcloud__2dtopo": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/knrr2-v5b78",
                       "Relative abundance of Z2 topological order in exfoliable 2D insulators: rights cc-by-4.0"),
    "mcloud__autowannier": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/0ctng-gre46",
                            "Automated high-throughput Wannierisation: rights cc-by-4.0"),
    "mcloud__curated-cofs": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/1xf71-nf390",
                             "via discover how-to-cite DOI 10.24435/materialscloud:cz-kq: rights cc-by-4.0"),
    "mcloud__pyrene-mofs": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/fzz0j-y2q60",
                            "via discover how-to-cite DOI 10.24435/materialscloud:z5-ct: rights cc-by-4.0"),
    "mcloud__stoceriaitf": ("CC-BY-4.0", "MC archive record", "https://archive.materialscloud.org/records/vzx65-faa74",
                            "via DOI 10.24435/materialscloud:2019.0025/v1 (SrTiO3-CeO2 interfaces): rights cc-by-4.0"),
    "mcloud__tin-antimony-sulfoiodide": ("CC-BY-4.0", "MC archive record",
                                         "https://archive.materialscloud.org/records/hmdsb-k3h43",
                                         "Sn2SbS2I3 chalcohalide record: rights cc-by-4.0"),
}

JARVIS_NOTE = ("NIST-PD: US-government work, public domain in the US, attribution requested, "
               "may be subject to foreign copyright; HF mirror tags it 'unknown'")

OPOLY_NOTE = ("Meta states: The OPoly26 dataset is provided under a CC-BY-4.0 license "
              "(on huggingface.co/facebook/OMol25); ColabFit HF mirror tags 'other' "
              "because model checkpoints use the FAIR Chemistry License")

SA_NOTE = "CC-BY-SA-4.0 confirmed at source; on project do-not-use list; decision pending with Michele"


def main():
    with open(HERE / "audit_skeleton.tsv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    with open(HERE / "hf_results.tsv", newline="", encoding="utf-8") as fh:
        hf = {r["dataset_id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    with open(HERE / "mcloud_results.tsv", newline="", encoding="utf-8") as fh:
        mc = {r["dataset_id"]: r for r in csv.DictReader(fh, delimiter="\t")}

    for row in rows:
        did = row["dataset_id"]

        if did in MANUAL:
            lic, auth, url, note = MANUAL[did]
            row.update(authority=auth, authority_license=lic, authority_ref=url, notes=note)
            row["match"] = "yes" if lic == row["spdx"] else "no"
            row["status"] = "PERMISSION" if did == "gdb13" else "VERIFIED"

        elif row["check_family"] == "colabfit_hf":
            h = hf[did]
            hf_spdx = HF_NORM.get(h["hf_license"], "HF:" + h["hf_license"])
            row.update(authority="huggingface colabfit mirror",
                       authority_license=hf_spdx,
                       authority_ref="https://huggingface.co/datasets/colabfit/" + h["hf_name"])
            row["ds_id_confirmed"] = "yes" if row["colabfit_ds_id"] and row["colabfit_ds_id"] in h["hf_ds_ids"] else "no"

            if hf_spdx == row["spdx"]:
                row["match"] = "yes"
                row["status"] = "VERIFIED"
            elif row["spdx"] == "NIST-PD" and h["hf_license"] == "unknown":
                row["match"] = "hf-tag-limitation"
                row["status"] = "VERIFIED"
                row["authority"] = "NIST license terms"
                row["authority_ref"] = "https://www.nist.gov/open/license"
                row["authority_license"] = "NIST-PD"
                row["notes"] = JARVIS_NOTE
            elif did.startswith("colabfit_OPoly26"):
                row["match"] = "resolved-to-meta-statement"
                row["status"] = "VERIFIED"
                row["authority"] = "Meta HF page"
                row["authority_ref"] = "https://huggingface.co/facebook/OMol25"
                row["authority_license"] = "CC-BY-4.0"
                row["notes"] = OPOLY_NOTE
            else:
                row["match"] = "no"
                row["status"] = "MISMATCH"

        elif row["check_family"] == "mcloud_api":
            m = mc[did]
            mc_spdx = MC_NORM.get(m["mc_license"], "MC:" + m["mc_license"])
            row.update(authority="Materials Cloud archive API",
                       authority_license=mc_spdx,
                       authority_ref="https://archive.materialscloud.org/records/" + m["invenio_id"])

            if mc_spdx == "CC-BY-SA-4.0":
                row["match"] = "yes" if row["spdx"] == "CC-BY-SA-4.0" else "no"
                row["status"] = "PROBLEM"
                row["notes"] = SA_NOTE
            elif mc_spdx == row["spdx"]:
                row["match"] = "yes"
                row["status"] = "VERIFIED"
                if m["mc_license"] == "cc-by-4.0;":
                    row["notes"] = ("record carries a License addendum entry: metadata (except emails) is "
                                    "CC-BY-SA-4.0; the data itself is CC-BY-4.0")
                if did == "mcloudarchive__hyjhn-8em96":
                    row["notes"] = SA_NOTE
            else:
                row["match"] = "no"
                row["status"] = "MISMATCH-CORRECTED"
                row["notes"] = f"source record says {mc_spdx}, shipped TSV said {row['tsv_license']}"

    with open(HERE / "license_audit.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("statuses:", counts)

    for row in rows:
        if row["status"] not in ("VERIFIED", "PERMISSION"):
            print(f"  {row['status']:20s} {row['dataset_id']:45s} tsv={row['tsv_license']} auth={row['authority_license']}")

    dsid_no = [r["dataset_id"] for r in rows if r["ds_id_confirmed"] == "no" and r["check_family"] == "colabfit_hf"]
    print("colabfit rows without DS-id echo in HF description:", len(dsid_no))


main()
