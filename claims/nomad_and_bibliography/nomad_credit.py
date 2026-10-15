import gzip
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

CLAIMS = "${DEV_ROOT}/paper/claims"
DIR = f"{CLAIMS}/out/nomad_provenance"
ENTRY_CACHE = f"{DIR}/entry_metadata_cache.tsv.gz"
CREDIT = f"{DIR}/credit_cache"
OUT = f"{CLAIMS}/out/nomad_credit.json"

API = "https://nomad-lab.eu/prod/v1/api/v1"
PAGE = 10000
THREADS = 4
ENTRY_FIELDS = ["entry_id", "authors.name", "references", "comment", "external_db", "external_id", "datasets.doi"]
UPLOAD_FIELDS = ["upload_name", "main_author.name", "upload_create_time", "publish_time"]


def post(payload, tries=8):
    data = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            request = urllib.request.Request(f"{API}/entries/query", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(20 * (attempt + 1))


def fetch_upload(task):
    # every entry of one upload, page by page, keeping the released ones, all credit fields uncapped
    upload, wanted = task
    path = f"{CREDIT}/{upload}.json.gz"
    if os.path.exists(path):
        return upload, "cached"
    try:
        entries, meta, after, total = {}, {}, None, None
        while True:
            pagination = {"page_size": PAGE, "order_by": "entry_id"}
            if after:
                pagination["page_after_value"] = after
            include = ENTRY_FIELDS + ([] if meta else UPLOAD_FIELDS)
            result = post({"query": {"upload_id": upload}, "pagination": pagination, "required": {"include": include}})
            total = result.get("pagination", {}).get("total", total)
            for e in result.get("data", []):
                if not meta:
                    meta = {"upload_name": e.get("upload_name", ""), "main_author": (e.get("main_author") or {}).get("name", ""),
                            "upload_create_time": e.get("upload_create_time", ""), "publish_time": e.get("publish_time", "")}
                if e["entry_id"] in wanted:
                    entries[e["entry_id"]] = {
                        "authors": [a.get("name", "") for a in e.get("authors", []) or []],
                        "references": e.get("references", []) or [],
                        "comment": e.get("comment", "") or "",
                        "external_db": e.get("external_db", "") or "",
                        "external_id": e.get("external_id", "") or "",
                        "dois": sorted({d["doi"] for d in e.get("datasets", []) or [] if d.get("doi")}),
                    }
            after = result.get("pagination", {}).get("next_page_after_value")
            if not after or not result.get("data"):
                break
        record = {"upload": meta, "entries_in_upload": total, "entries": entries,
                  "missing": sorted(set(wanted) - set(entries))}
        tmp = path + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(record, f)
        os.replace(tmp, path)
        return upload, "fetched"
    except Exception as error:
        return upload, f"failed: {error}"


def fetch_batch(batch):
    # many small uploads in one query, split by upload afterwards
    uploads = [u for u, _ in batch if not os.path.exists(f"{CREDIT}/{u}.json.gz")]
    if not uploads:
        return len(batch), "cached"
    wanted = {u: w for u, w in batch}
    try:
        entries, meta, totals, after = defaultdict(dict), {}, defaultdict(int), None
        while True:
            pagination = {"page_size": PAGE, "order_by": "entry_id"}
            if after:
                pagination["page_after_value"] = after
            result = post({"query": {"upload_id": {"any": uploads}}, "pagination": pagination,
                           "required": {"include": ENTRY_FIELDS + UPLOAD_FIELDS + ["upload_id"]}})
            for e in result.get("data", []):
                u = e["upload_id"]
                totals[u] += 1
                meta.setdefault(u, {"upload_name": e.get("upload_name", ""), "main_author": (e.get("main_author") or {}).get("name", ""),
                                    "upload_create_time": e.get("upload_create_time", ""), "publish_time": e.get("publish_time", "")})
                if e["entry_id"] in wanted.get(u, ()):
                    entries[u][e["entry_id"]] = {
                        "authors": [a.get("name", "") for a in e.get("authors", []) or []],
                        "references": e.get("references", []) or [],
                        "comment": e.get("comment", "") or "",
                        "external_db": e.get("external_db", "") or "",
                        "external_id": e.get("external_id", "") or "",
                        "dois": sorted({d["doi"] for d in e.get("datasets", []) or [] if d.get("doi")}),
                    }
            after = result.get("pagination", {}).get("next_page_after_value")
            if not after or not result.get("data"):
                break
        for u in uploads:
            record = {"upload": meta.get(u, {}), "entries_in_upload": totals[u], "entries": entries[u],
                      "missing": sorted(set(wanted[u]) - set(entries[u]))}
            path = f"{CREDIT}/{u}.json.gz"
            with gzip.open(path + ".tmp", "wt") as f:
                json.dump(record, f)
            os.replace(path + ".tmp", path)
        return len(uploads), "fetched"
    except Exception as error:
        return len(uploads), f"failed: {error}"


def main():
    os.makedirs(CREDIT, exist_ok=True)
    by_upload = defaultdict(set)
    with gzip.open(ENTRY_CACHE, "rt") as f:
        for line in f:
            entry, upload, _ = line.rstrip("\n").split("\t")
            by_upload[upload].add(entry)
    tasks = sorted(by_upload.items(), key=lambda t: -len(t[1]))
    # --small-first works from the smallest uploads up, so a second worker can run next to the first
    order = tasks[::-1] if "--small-first" in sys.argv else tasks
    todo = [t for t in order if not os.path.exists(f"{CREDIT}/{t[0]}.json.gz")]
    print(f"{len(tasks)} uploads, {len(todo)} to fetch", flush=True)

    failed = []
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else THREADS

    # --batched: uploads below SMALL released entries go in groups of at most 200 uploads and BATCH entries
    if "--batched" in sys.argv:
        SMALL, BATCH = 5000, 20000
        small = [t for t in todo if len(t[1]) < SMALL][::-1]
        batches, current, size = [], [], 0
        for t in small:
            if current and (len(current) == 200 or size + len(t[1]) > BATCH):
                batches.append(current)
                current, size = [], 0
            current.append(t)
            size += len(t[1])
        if current:
            batches.append(current)
        print(f"{len(small)} small uploads in {len(batches)} batches", flush=True)
        done = 0
        with ThreadPoolExecutor(threads) as pool:
            for n, (count, status) in enumerate(pool.map(fetch_batch, batches), 1):
                done += count
                if status.startswith("failed"):
                    print(f"batch {n} {status}", flush=True)
                if n % 10 == 0:
                    print(f"batches {n}/{len(batches)}, uploads {done}", flush=True)
        return

    with ThreadPoolExecutor(threads) as pool:
        for n, (upload, status) in enumerate(pool.map(fetch_upload, todo), 1):
            if status.startswith("failed"):
                failed.append(upload)
                print(f"{upload} {status}", flush=True)
            if n % 250 == 0:
                print(f"{n}/{len(todo)}", flush=True)

    entries = missing = 0
    for upload, wanted in tasks:
        path = f"{CREDIT}/{upload}.json.gz"
        if os.path.exists(path):
            record = json.load(gzip.open(path, "rt"))
            entries += len(record["entries"])
            missing += len(record["missing"])
    report = {"uploads": len(tasks), "uploads_failed": failed, "entries_with_credit": entries,
              "released_entries_not_found_in_their_upload": missing}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
