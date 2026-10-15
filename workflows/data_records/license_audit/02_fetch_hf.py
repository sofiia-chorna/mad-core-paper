import csv
import json
import pathlib
import re
import time
import urllib.request
import urllib.error

HERE = pathlib.Path(__file__).parent
SKELETON = HERE / "audit_skeleton.tsv"
OUT = HERE / "hf_results.tsv"

API = "https://huggingface.co/api/datasets/colabfit/{name}"
DS_ID_RE = re.compile(r"DS_[a-z0-9]{12}_\d+")


def strip_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def fetch(name):
    req = urllib.request.Request(API.format(name=name), headers={"User-Agent": "license-audit"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as err:
        return err.code, None
    except Exception:
        return -1, None


def main():
    with open(SKELETON, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["check_family"] == "colabfit_hf"]

    results = []
    for i, row in enumerate(rows):
        name = row["title"] or strip_prefix(row["dataset_id"], "colabfit_")
        status, payload = fetch(name)

        if status != 200 and name != strip_prefix(row["dataset_id"], "colabfit_"):
            name = strip_prefix(row["dataset_id"], "colabfit_")
            status, payload = fetch(name)

        hf_license = ""
        ds_ids = ""
        if payload:
            hf_license = (payload.get("cardData") or {}).get("license", "")
            if not hf_license:
                tags = [t for t in payload.get("tags", []) if t.startswith("license:")]
                hf_license = ";".join(t.split(":", 1)[1] for t in tags)
            ds_ids = ";".join(sorted(set(DS_ID_RE.findall(payload.get("description", "")))))

        results.append({
            "dataset_id": row["dataset_id"],
            "colabfit_ds_id": row["colabfit_ds_id"],
            "hf_name": name,
            "http_status": status,
            "hf_license": hf_license,
            "hf_ds_ids": ds_ids,
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
