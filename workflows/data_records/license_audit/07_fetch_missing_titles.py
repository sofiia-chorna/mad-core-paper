import csv
import json
import re
import sys
import urllib.error
import urllib.request

HERE = "${DEV_ROOT}/phace-atomic/release/build/license_audit"
RECORD_CSV = ("${DEV_ROOT}/phace-atomic/matcloud-upload/"
              "xyz-2pow24-v2/files/dataset_metadata.csv")
CARD = "https://huggingface.co/datasets/colabfit/{name}/raw/main/README.md"
OUT = f"{HERE}/missing_titles.tsv"

FIELDS = ["dataset_id", "title", "source_publication", "original_data_link", "card_license",
          "configurations"]


def card_text(name):
    req = urllib.request.Request(CARD.format(name=name), headers={"User-Agent": "mad-core-metadata"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode(errors="replace")


def section(text, heading):
    pattern = re.compile(r"#+\s*" + re.escape(heading) + r"\s*\n(.*?)(?:\n#|\Z)", re.S)
    found = pattern.search(text)
    if not found:
        return ""
    return " ".join(found.group(1).split()).strip()


def pretty_name(text):
    front = re.match(r"---\n(.*?)\n---", text, re.S)
    if front:
        found = re.search(r"^pretty_name:\s*(.+)$", front.group(1), re.M)
        if found:
            return found.group(1).strip()
    return section(text, "Dataset  Name") or section(text, "Dataset Name")


def main():
    with open(RECORD_CSV) as fh:
        rows = list(csv.DictReader(fh))

    missing = [r["dataset_id"] for r in rows
               if not r["title"].strip() and r["dataset_id"].startswith("colabfit_")]
    print(f"colabfit rows without a title: {len(missing)}")

    out = []
    for dataset_id in missing:
        name = dataset_id[len("colabfit_"):]
        try:
            text = card_text(name)
        except urllib.error.HTTPError as exc:
            print(f"  {name}: HTTP {exc.code}")
            out.append({"dataset_id": dataset_id, "title": "", "source_publication": "",
                        "original_data_link": "", "card_license": "", "configurations": ""})
            continue

        entry = {
            "dataset_id": dataset_id,
            "title": pretty_name(text) or name,
            "source_publication": section(text, "Publication"),
            "original_data_link": section(text, "Original data link"),
            "card_license": section(text, "License"),
            "configurations": section(text, "Number of unique molecular configurations"),
        }
        out.append(entry)
        print(f"  {name}: {entry['title']} | pub {entry['source_publication'][:60]}")

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
