import csv
import json
import pathlib
import time
import urllib.request
import urllib.error

HERE = pathlib.Path(__file__).parent
SKELETON = HERE / "audit_skeleton.tsv"
PROVENANCE = "${STORE_ROOT}/mad-core/audit/mcloud_archive_provenance.csv"
OUT = HERE / "mcloud_results.tsv"

API = "https://archive.materialscloud.org/api/records/{rid}"

DOI_SUFFIX_REDIRECTS = {
    "1z-pd": "zn7cy-7cs67",
    "ed-g2": "rr3za-8c528",
    "jk-9v": "4da2r-xag86",
    "m0-zg": "gnv15-27a69",
    "zt-z4": "z24e6-dv880",
}


def fetch(rid):
    req = urllib.request.Request(API.format(rid=rid), headers={"User-Agent": "license-audit"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, None
    except Exception:
        return -1, None


def main():
    with open(PROVENANCE, newline="", encoding="utf-8") as fh:
        provenance = {r["folder"]: r for r in csv.DictReader(fh)}

    with open(SKELETON, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["check_family"] == "mcloud_api"]

    results = []
    for i, row in enumerate(rows):
        prov = provenance.get(row["dataset_id"])
        fallback_rid = row["dataset_id"][len("mcloudarchive__"):]
        rid = prov["invenio_id"] if prov else fallback_rid
        rid = DOI_SUFFIX_REDIRECTS.get(rid, rid)
        status, payload = fetch(rid)

        licenses = ""
        doi = ""
        title = ""
        if payload:
            meta = payload.get("metadata", {})
            licenses = ";".join(r.get("id", "") for r in meta.get("rights", []))
            doi = payload.get("links", {}).get("doi", "")
            title = " ".join(meta.get("title", "").split())[:100]

        results.append({
            "dataset_id": row["dataset_id"],
            "invenio_id": rid,
            "http_status": status,
            "mc_license": licenses,
            "mc_doi": doi,
            "mc_title": title,
            "in_provenance_csv": "yes" if prov else "no",
        })

        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}")
        time.sleep(0.1)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    bad = [r for r in results if r["http_status"] != 200]
    print("total:", len(results), "non-200:", len(bad))
    for r in bad:
        print("  MISS", r["dataset_id"], r["http_status"])


main()
