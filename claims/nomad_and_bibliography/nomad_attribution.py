import gzip
import importlib.util
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

CLAIMS = "${DEV_ROOT}/paper/claims"
PAPER = "${DEV_ROOT}/paper"
DIR = f"{CLAIMS}/out/nomad_provenance"
NOMAD = "${STORE_ROOT}/mad-core/release/collection/nomad"
OUT = f"{CLAIMS}/out/nomad_attribution.json"

ENTRY_URL = "https://nomad-lab.eu/prod/v1/gui/search/entries/entry/id/{}"
UPLOAD_URL = "https://nomad-lab.eu/prod/v1/gui/user/uploads/upload/id/{}"

# entry ids per released row, exactly as nomad_provenance.py derives them
spec = importlib.util.spec_from_file_location("nomad_provenance", f"{CLAIMS}/nomad_provenance.py")
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def clean(value):
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


SHORT_DOI = {"10/F2696T": "10.17172/NOMAD/PBVEIBCA390AP80E7N8DX5IFZSAHZ8Q3RU8ZIVXT6KPPIMS",
             "10/Z9C": "10.17172/NOMAD/2O4HTSZH38W7JVZEK9E1RD8CVUBA6541K6T8V3DLQFAMJMS"}


def bib_dois(path):
    keys = {d.upper(): k for k, d in re.findall(r"@misc\{(\d+),\n\s*doi = \{([^}]*)\}", open(path).read())}
    keys.update({short: keys[long] for short, long in SHORT_DOI.items() if long in keys})
    return keys


def main():
    ids = provenance.row_entry_ids()
    configuration_id = np.load(f"{NOMAD}/configuration_id.npy", mmap_mode="r")

    upload_of = {}
    with gzip.open(f"{DIR}/entry_metadata_cache.tsv.gz", "rt") as f:
        for line in f:
            entry, upload, _ = line.rstrip("\n").split("\t")
            upload_of[entry] = upload

    # credit variants (upload, authors, references, comment, external_db) to keep memory small
    variant_index, variant_keys = {}, []
    credit, uploads_meta = {}, {}
    display = {}
    entry_dois = defaultdict(set)
    for upload in sorted(set(upload_of.values())):
        record = json.load(gzip.open(f"{DIR}/credit_cache/{upload}.json.gz", "rt"))
        uploads_meta[upload] = record["upload"]
        for entry, c in record["entries"].items():
            key = (upload, tuple(c["authors"]), tuple(c["references"]), clean(c["comment"]), clean(c["external_db"]))
            if key not in variant_index:
                variant_index[key] = len(variant_keys)
                variant_keys.append(key)
            credit[entry] = (variant_index[key], clean(c["external_id"]))
            for d in c["dois"]:
                display.setdefault(d.upper(), d)
                entry_dois[entry].add(d.upper())

    for d, result in json.load(gzip.open(f"{CLAIMS}/out/nomad_doi_entries.json.gz", "rt")).items():
        display.setdefault(d.upper(), d)
        for entry in result["entries"]:
            entry_dois[entry].add(d.upper())

    cited = bib_dois(f"{PAPER}/biblio/nomad.bib")
    before = {d.upper(): k for d, k in json.load(open(f"{CLAIMS}/out/nomad_bib_keys_before_dedup.json")).items()}
    nested = json.load(open(f"{CLAIMS}/out/nomad_citations.json"))["redundant_dois"]
    extra = json.load(open(f"{DIR}/doi_metadata_extra.json"))

    released = set(ids)
    doi_entries = defaultdict(set)
    for entry in released:
        for d in entry_dois.get(entry, ()):
            doi_entries[d].add(entry)
    rows_per_entry = Counter(ids)

    # first pass rows per variant, so that the variants of each upload can be numbered by size
    missing_key = {}
    variant_rows = Counter()
    for entry in ids:
        if entry in credit:
            variant_rows[credit[entry][0]] += 1
        else:
            upload = upload_of.get(entry, "")
            if upload not in missing_key:
                missing_key[upload] = len(variant_keys)
                variant_keys.append((upload, (), (), "", ""))
            variant_rows[missing_key[upload]] += 1
    by_upload = defaultdict(list)
    for i, key in enumerate(variant_keys):
        if variant_rows[i]:
            by_upload[key[0]].append(i)
    numbered = {}
    for upload, members in by_upload.items():
        for n, i in enumerate(sorted(members, key=lambda i: -variant_rows[i]), 1):
            numbered[i] = f"{upload}:{n}"

    per_upload = defaultdict(lambda: {"rows": 0, "rows_doi": 0, "rows_cited": 0, "dois": set(), "keys": set(),
                                      "authors": Counter(), "references": Counter(), "external_db": Counter()})
    missing_credit = 0
    totals = Counter()
    with gzip.open(f"{DIR}/nomad_records.tsv.gz", "wt") as f:
        f.write("release_row\tconfiguration_id\tnomad_entry_id\tnomad_upload_id\texternal_db\texternal_id\t"
                "dataset_dois\tcited_as\tcredit_variant\tnomad_entry_url\n")
        for row, entry in enumerate(ids):
            upload = upload_of.get(entry, "")
            if entry in credit:
                v, external_id = credit[entry]
            else:
                missing_credit += 1
                v, external_id = missing_key[upload], ""
            _, authors, references, comment, external_db = variant_keys[v]
            dois = sorted(entry_dois.get(entry, set()))
            keys = sorted({cited[d] for d in dois if d in cited}, key=int)
            totals["rows"] += 1
            totals["rows_with_doi"] += bool(dois)
            totals["rows_cited"] += bool(keys)
            p = per_upload[upload]
            p["rows"] += 1
            p["rows_doi"] += bool(dois)
            p["rows_cited"] += bool(keys)
            p["dois"].update(dois)
            p["keys"].update(keys)
            p["authors"].update(authors)
            p["references"].update(references)
            if external_db:
                p["external_db"][external_db] += 1
            f.write("\t".join([str(row), configuration_id[row].decode(), entry, upload, external_db, external_id,
                               ";".join(display[d] for d in dois), ";".join(keys), numbered[v],
                               ENTRY_URL.format(entry)]) + "\n")

    with open(f"{DIR}/nomad_credit_variants.tsv", "w") as f:
        f.write("credit_variant\tnomad_upload_id\treleased_records\tauthors\treferences\tcomment\texternal_db\n")
        for i in sorted(numbered, key=lambda i: (variant_keys[i][0], int(numbered[i].rsplit(":", 1)[1]))):
            upload, authors, references, comment, external_db = variant_keys[i]
            f.write("\t".join([numbered[i], upload, str(variant_rows[i]), "; ".join(clean(a) for a in authors),
                               " | ".join(clean(r) for r in references), comment, external_db]) + "\n")

    with open(f"{DIR}/nomad_uploads.tsv", "w") as f:
        f.write("nomad_upload_id\tupload_name\tmain_author\tupload_create_time\tpublish_time\treleased_records\t"
                "records_with_dataset_doi\trecords_in_cited_dataset\tdataset_dois\tcited_as\tauthors\treferences\t"
                "external_db\tcredit_variants\tnomad_upload_url\n")
        for upload, p in sorted(per_upload.items(), key=lambda t: -t[1]["rows"]):
            meta = uploads_meta.get(upload, {})
            f.write("\t".join([upload, clean(meta.get("upload_name", "")), clean(meta.get("main_author", "")),
                               meta.get("upload_create_time", ""), meta.get("publish_time", ""), str(p["rows"]),
                               str(p["rows_doi"]), str(p["rows_cited"]), ";".join(display[d] for d in sorted(p["dois"])),
                               ";".join(sorted(p["keys"], key=int)),
                               "; ".join(clean(a) for a, _ in p["authors"].most_common()),
                               " | ".join(clean(r) for r, _ in p["references"].most_common()),
                               "; ".join(f"{k} ({v})" for k, v in p["external_db"].most_common()),
                               str(len(by_upload[upload])), UPLOAD_URL.format(upload)]) + "\n")

    with open(f"{DIR}/nomad_dois.tsv", "w") as f:
        f.write("dataset_doi\tstatus\tbib_key\tprevious_bib_key\treleased_entries\treleased_records\t"
                "contained_in_doi\tcontained_in_bib_key\tidentical_released_entries\ttitle\tauthors\tyear\tmetadata_source\n")
        status_count = Counter()
        for d in sorted(doi_entries, key=lambda d: (d not in cited, -len(doi_entries[d]))):
            entries = doi_entries[d]
            records = sum(rows_per_entry[e] for e in entries)
            holder = ""
            if d in cited:
                status = "cited"
            elif d in nested:
                status = "omitted, nested in a cited DOI"
                holder = nested[d]["contained_in"].upper()
            else:
                containing = [c for c in cited if entries <= doi_entries.get(c, set())]
                holder = max(containing, key=lambda c: len(doi_entries[c])) if containing else ""
                status = "not cited, new since the download" + (", nested in a cited DOI" if holder else "")
            status_count[status] += 1
            meta = extra.get(d, {})
            f.write("\t".join([display[d], status, cited.get(d, ""), before.get(d, meta.get("old_bib_key", "")),
                               str(len(entries)), str(records), display.get(holder, holder), cited.get(holder, ""),
                               str(bool(holder) and entries == doi_entries.get(holder, set())),
                               clean(meta.get("title", "")), clean(meta.get("authors", "")), meta.get("year", ""),
                               meta.get("source", "biblio/nomad.bib" if d in cited else "")]) + "\n")

    # reconciliation of the records outside any cited dataset
    without_cited_upload = [u for u, p in per_upload.items() if p["rows_cited"] == 0]
    rows_in_uploads_without_cited = sum(per_upload[u]["rows"] for u in without_cited_upload)
    rows_outside_cited_in_mixed_uploads = sum(p["rows"] - p["rows_cited"] for u, p in per_upload.items() if p["rows_cited"] > 0)
    mp = {u: p for u, p in per_upload.items() if "Materials Project" in p["external_db"]}
    report = {
        "released_rows": totals["rows"],
        "rows_with_any_dataset_doi": totals["rows_with_doi"],
        "rows_in_a_cited_dataset": totals["rows_cited"],
        "rows_without_any_dataset_doi": totals["rows"] - totals["rows_with_doi"],
        "rows_outside_cited_datasets": totals["rows"] - totals["rows_cited"],
        "reconciliation": {
            "rows_outside_cited_datasets": totals["rows"] - totals["rows_cited"],
            "in_uploads_without_any_cited_dataset": rows_in_uploads_without_cited,
            "in_uploads_that_also_hold_cited_records": rows_outside_cited_in_mixed_uploads,
            "check": rows_in_uploads_without_cited + rows_outside_cited_in_mixed_uploads == totals["rows"] - totals["rows_cited"],
            "uploads_without_any_cited_dataset": len(without_cited_upload),
            "uploads_with_cited_records": len(per_upload) - len(without_cited_upload),
        },
        "dois_with_released_records": len(doi_entries),
        "doi_status": dict(status_count),
        "uploads": len(per_upload),
        "credit_variants": len(numbered),
        "uploads_with_more_than_one_credit_variant": sum(1 for k in by_upload.values() if len(k) > 1),
        "rows_without_credit_record": missing_credit,
        "distinct_authors": len({a for p in per_upload.values() for a in p["authors"]}),
        "distinct_references": len({r for p in per_upload.values() for r in p["references"]}),
        "uploads_with_references": sum(1 for p in per_upload.values() if p["references"]),
        "rows_with_references": sum(variant_rows[i] for i in numbered if variant_keys[i][2]),
        "external_db": dict(sum((p["external_db"] for p in per_upload.values()), Counter())),
        "materials_project": {"uploads": len(mp), "rows": sum(p["external_db"]["Materials Project"] for p in mp.values())},
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
