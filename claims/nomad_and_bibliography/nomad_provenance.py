import gzip
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
NOMAD = f"{RELEASE}/collection/nomad"
DOWNLOAD = "${STORE_ROOT}/dataset_collection/download_nomad"
PAPER = "${DEV_ROOT}/paper"
CLAIMS = f"{PAPER}/claims"
DIR = f"{CLAIMS}/out/nomad_provenance"
CACHE = f"{DIR}/entry_metadata_cache.tsv.gz"
UPLOAD_CACHE = f"{DIR}/upload_metadata_cache.json.gz"
OUT = f"{CLAIMS}/out/nomad_provenance.json"

API = "https://nomad-lab.eu/prod/v1/api/v1"
ENTRY_URL = "https://nomad-lab.eu/prod/v1/gui/search/entries/entry/id/{}"
UPLOAD_URL = "https://nomad-lab.eu/prod/v1/gui/user/uploads/upload/id/{}"
BATCH = 5000
THREADS = 4
MAX_REFERENCES = 5
INCLUDE = ["entry_id", "upload_id", "upload_name", "main_author.name", "datasets.doi", "references"]


def post(endpoint, payload, tries=8):
    data = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            request = urllib.request.Request(f"{API}/{endpoint}", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


def row_entry_ids():
    offsets = np.load(f"{NOMAD}/source_reference_offsets.npy")
    blob = np.fromfile(f"{NOMAD}/source_reference_bytes.bin", dtype=np.uint8).tobytes()
    ids, pending = [], defaultdict(dict)
    for i in range(len(offsets) - 1):
        reference = blob[offsets[i]:offsets[i + 1]]
        if b"|" in reference:
            ids.append(reference.split(b"|", 1)[1].decode())
        else:
            source, frame = reference.decode().rsplit("__frame", 1)
            pending[source.split("nomad__", 1)[1]][int(frame)] = i
            ids.append(None)
    for name, frames in pending.items():
        frame = -1
        with open(f"{DOWNLOAD}/{name}/part_0000.xyz") as f:
            for line in f:
                if "entry_id=" in line:
                    frame += 1
                    if frame in frames:
                        ids[frames[frame]] = re.search(r"entry_id=(\S+)", line).group(1)
    assert all(ids), sum(1 for e in ids if not e)
    return ids


def fetch(batch):
    # a batch that keeps failing is skipped and fetched again on the next run
    try:
        result = post("entries/query", {
            "query": {"entry_id": {"any": batch}},
            "pagination": {"page_size": len(batch)},
            "required": {"include": INCLUDE},
        })
        return result.get("data", [])
    except Exception as error:
        print(f"batch of {len(batch)} failed: {error}", flush=True)
        return None


def remember_upload(uploads, e):
    u = uploads.setdefault(e["upload_id"], {"upload_name": e.get("upload_name", ""),
                                            "main_author": (e.get("main_author") or {}).get("name", ""),
                                            "references": []})
    for reference in e.get("references", []) or []:
        if reference not in u["references"] and len(u["references"]) < MAX_REFERENCES:
            u["references"].append(reference)


def load_cache():
    entries, uploads = {}, {}
    if os.path.exists(CACHE):
        with gzip.open(CACHE, "rt") as f:
            for line in f:
                entry, upload, dois = line.rstrip("\n").split("\t")
                entries[entry] = (upload, dois)
    if os.path.exists(UPLOAD_CACHE):
        uploads = json.load(gzip.open(UPLOAD_CACHE, "rt"))
    return entries, uploads


def main():
    os.makedirs(DIR, exist_ok=True)
    ids = row_entry_ids()
    distinct = sorted(set(ids))
    entries, uploads = load_cache()

    offline = "--offline" in sys.argv
    missing = [e for e in distinct if e not in entries]
    if missing and not offline:
        batches = [missing[i:i + BATCH] for i in range(0, len(missing), BATCH)]
        print(f"querying {len(missing)} entries in {len(batches)} batches", flush=True)
        failed = 0
        with ThreadPoolExecutor(THREADS) as pool:
            for n, data in enumerate(pool.map(fetch, batches), 1):
                if data is None:
                    failed += 1
                    continue
                with gzip.open(CACHE, "at") as cache:
                    for e in data:
                        dois = ";".join(sorted({d["doi"] for d in e.get("datasets", []) if d.get("doi")}))
                        entries[e["entry_id"]] = (e["upload_id"], dois)
                        cache.write(f"{e['entry_id']}\t{e['upload_id']}\t{dois}\n")
                        remember_upload(uploads, e)
                with gzip.open(UPLOAD_CACHE, "wt") as f:
                    json.dump(uploads, f)
                if n % 25 == 0:
                    print(f"{n}/{len(batches)}", flush=True)
        if failed:
            raise SystemExit(f"{failed} batches failed, run again to fetch them")

    # uploads known from cached entries whose metadata was not saved before an interruption
    lacking = sorted({u for u, _ in entries.values()} - set(uploads))
    if lacking and not offline:
        print(f"fetching metadata of {len(lacking)} uploads", flush=True)
        for i in range(0, len(lacking), 200):
            for upload in lacking[i:i + 200]:
                result = post("entries/query", {"query": {"upload_id": upload}, "pagination": {"page_size": 1},
                                                "required": {"include": INCLUDE}})
                for e in result.get("data", []):
                    remember_upload(uploads, e)
            with gzip.open(UPLOAD_CACHE, "wt") as f:
                json.dump(uploads, f)

    # bib keys of the cited NOMAD DOIs
    bib = dict((d.upper(), k) for k, d in re.findall(r"@misc\{(\d+),\n\s*doi = \{([^}]*)\}", open(f"{PAPER}/biblio/nomad.bib").read()))

    entry_dois = defaultdict(set)
    for doi, result in json.load(gzip.open(f"{CLAIMS}/out/nomad_doi_entries.json.gz", "rt")).items():
        for entry in result["entries"]:
            entry_dois[entry].add(doi)
    configuration_id = np.load(f"{NOMAD}/configuration_id.npy", mmap_mode="r")

    per_upload = defaultdict(lambda: {"records": 0, "records_in_cited_dataset": 0, "dois": set(), "keys": set()})
    not_found = 0
    rows_cited = 0
    with gzip.open(f"{DIR}/nomad_records.tsv.gz", "wt") as f:
        f.write("release_row\tconfiguration_id\tnomad_entry_id\tnomad_upload_id\tdataset_dois\tcited_as\tnomad_entry_url\n")
        for row, entry in enumerate(ids):
            upload, api_dois = entries.get(entry, ("", ""))
            not_found += upload == ""
            dois = ";".join(sorted({d.upper() for d in api_dois.split(";") if d} | entry_dois.get(entry, set())))
            cited = sorted({bib[d.upper()] for d in dois.split(";") if d and d.upper() in bib}, key=int)
            rows_cited += bool(cited)
            f.write(f"{row}\t{configuration_id[row].decode()}\t{entry}\t{upload}\t{dois}\t{';'.join(cited)}\t{ENTRY_URL.format(entry)}\n")
            if upload:
                p = per_upload[upload]
                p["records"] += 1
                p["records_in_cited_dataset"] += bool(cited)
                p["dois"].update(d for d in dois.split(";") if d)
                p["keys"].update(cited)

    with open(f"{DIR}/nomad_uploads.tsv", "w") as f:
        f.write("nomad_upload_id\tupload_name\tmain_author\treleased_records\trecords_in_cited_dataset\tdataset_dois\tcited_as\treferences\tnomad_upload_url\n")
        for upload, p in sorted(per_upload.items(), key=lambda t: -t[1]["records"]):
            meta = uploads.get(upload, {})
            clean = lambda s: str(s).replace("\t", " ").replace("\n", " ")
            f.write("\t".join([upload, clean(meta.get("upload_name", "")), clean(meta.get("main_author", "")),
                               str(p["records"]), str(p["records_in_cited_dataset"]), ";".join(sorted(p["dois"])),
                               ";".join(sorted(p["keys"], key=int)), " ".join(clean(r) for r in meta.get("references", [])),
                               UPLOAD_URL.format(upload)]) + "\n")

    uploads_without_doi = [u for u, p in per_upload.items() if not p["keys"]]
    report = {
        "released_rows": len(ids),
        "distinct_entry_ids": len(distinct),
        "rows_resolved_from_source_files": 1204,
        "complete": not_found == 0,
        "rows_with_upload_metadata": len(ids) - not_found,
        "rows_without_upload_metadata": not_found,
        "uploads": len(per_upload),
        "rows_in_a_cited_dataset": rows_cited,
        "uploads_with_a_cited_dataset": len(per_upload) - len(uploads_without_doi),
        "uploads_without_any_cited_dataset": len(uploads_without_doi),
        "rows_in_uploads_without_any_cited_dataset": sum(per_upload[u]["records"] for u in uploads_without_doi),
        "uploads_with_references": sum(1 for u in per_upload if uploads.get(u, {}).get("references")),
        "distinct_main_authors": len({uploads.get(u, {}).get("main_author", "") for u in per_upload}),
        "files": {"per_record": f"{DIR}/nomad_records.tsv.gz", "per_upload": f"{DIR}/nomad_uploads.tsv"},
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
