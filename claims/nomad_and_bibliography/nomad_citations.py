import gzip
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RELEASE = "${STORE_ROOT}/mad-core/release"
NOMAD = f"{RELEASE}/collection/nomad"
DOWNLOAD = "${STORE_ROOT}/dataset_collection/download_nomad"
PAPER = "${DEV_ROOT}/paper"
CLAIMS = f"{PAPER}/claims"
OUT = f"{CLAIMS}/out/nomad_citations.json"
CACHE = f"{CLAIMS}/out/nomad_doi_entries.json.gz"

# bib keys kept whatever their overlap: the descriptive title of each identical pair, and graphene and defective
# graphene, which would otherwise be credited only through the dataset titled "Test-22.08.2018-3" (key 112)
ALWAYS_KEEP = ["0", "1", "52", "96", "157", "175", "462"]

API = "https://nomad-lab.eu/prod/v1/api/v1"
PAGE = 10000
THREADS = 8


def post(endpoint, payload, tries=6):
    data = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            request = urllib.request.Request(f"{API}/{endpoint}", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except Exception as error:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def fetch_doi(doi):
    # every entry carrying this dataset DOI, with the dataset records of the first page for names and ids
    entries, after, datasets = [], None, {}
    while True:
        pagination = {"page_size": PAGE}
        if after:
            pagination["page_after_value"] = after
        include = ["entry_id"] if datasets else ["entry_id", "datasets"]
        result = post("entries/query", {"query": {"datasets.doi": doi}, "pagination": pagination, "required": {"include": include}})
        for entry in result.get("data", []):
            entries.append(entry["entry_id"])
            for dataset in entry.get("datasets", []):
                if dataset.get("doi") == doi:
                    datasets[dataset.get("dataset_id")] = {
                        "dataset_name": dataset.get("dataset_name"),
                        "dataset_create_time": dataset.get("dataset_create_time"),
                    }
        after = result.get("pagination", {}).get("next_page_after_value")
        if not after or not result.get("data"):
            break
    return doi, {"entries": entries, "datasets": datasets}


def released_entry_ids():
    offsets = np.load(f"{NOMAD}/source_reference_offsets.npy")
    blob = np.fromfile(f"{NOMAD}/source_reference_bytes.bin", dtype=np.uint8).tobytes()
    ids, frames = Counter(), []
    for i in range(len(offsets) - 1):
        reference = blob[offsets[i]:offsets[i + 1]]
        if b"|" in reference:
            ids[reference.split(b"|", 1)[1].decode()] += 1
        else:
            source, frame = reference.decode().rsplit("__frame", 1)
            frames.append((source.split("nomad__", 1)[1], int(frame)))

    # rows whose reference carries no entry id point to a frame of a filtered source file
    by_file = defaultdict(set)
    for name, frame in frames:
        by_file[name].add(frame)
    for name, wanted in by_file.items():
        found = 0
        with open(f"{DOWNLOAD}/{name}/part_0000.xyz") as f:
            frame = -1
            for line in f:
                if "entry_id=" in line:
                    frame += 1
                    if frame in wanted:
                        ids[re.search(r"entry_id=(\S+)", line).group(1)] += 1
                        found += 1
        assert found == len(wanted), (name, found, len(wanted))
    return ids, len(frames)


def bib_entries(path):
    text = open(path).read()
    out = []
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        # NOMAD datasets have numeric keys; the original sources kept in the same file do not
        if not key.strip().isdigit():
            continue
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", body, re.M))
        out.append({"key": key, "doi": fields.get("doi", "").upper(), "title": fields.get("title", ""), "author": fields.get("author", "")})
    return out


def main():
    candidates = [line.strip().upper() for line in open(f"{DOWNLOAD}/doi.txt") if line.strip()]
    bib = bib_entries(f"{PAPER}/biblio/nomad.bib")
    bib_dois = {e["doi"] for e in bib}
    all_dois = sorted(set(candidates) | bib_dois)

    cache = json.load(gzip.open(CACHE, "rt")) if os.path.exists(CACHE) else {}
    missing = [d for d in all_dois if d not in cache]
    if missing:
        print(f"querying {len(missing)} DOIs", flush=True)
        with ThreadPoolExecutor(THREADS) as pool:
            for n, (doi, result) in enumerate(pool.map(fetch_doi, missing), 1):
                cache[doi] = result
                if n % 50 == 0:
                    print(f"{n}/{len(missing)}", flush=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with gzip.open(CACHE, "wt") as f:
            json.dump(cache, f)

    released, resolved_from_files = released_entry_ids()
    rows = sum(released.values())
    assert rows == int(np.load(f"{NOMAD}/ns.npy")[0]), rows

    per_doi = {}
    for doi in all_dois:
        hit = [e for e in set(cache[doi]["entries"]) if e in released]
        per_doi[doi] = {
            "entries_now": len(set(cache[doi]["entries"])),
            "released_entries": len(hit),
            "released_rows": sum(released[e] for e in hit),
            "datasets": cache[doi]["datasets"],
            "in_bib": doi in bib_dois,
            "in_download_list": doi in candidates,
        }
    released_sets = {d: set(cache[d]["entries"]) & set(released) for d in all_dois}

    # DOIs whose released entries all belong to another DOI as well
    covered_by = {}
    live = [d for d in all_dois if released_sets[d]]
    for d in live:
        others = [o for o in live if o != d and released_sets[d] <= released_sets[o]]
        if others:
            covered_by[d] = sorted(others, key=lambda o: -len(released_sets[o]))
    identical = sorted({tuple(sorted([d, o])) for d, os_ in covered_by.items() for o in os_ if released_sets[d] == released_sets[o]})
    names = defaultdict(list)
    for d in live:
        for info in per_doi[d]["datasets"].values():
            names[info["dataset_name"]].append(d)

    # minimal citation set: largest DOIs first, a DOI is kept unless one kept DOI already holds all its released entries
    keys = {e["doi"]: e["key"] for e in bib}
    if os.path.exists(f"{CLAIMS}/out/nomad_bib_keys_before_dedup.json"):
        keys.update(json.load(open(f"{CLAIMS}/out/nomad_bib_keys_before_dedup.json")))
    by_key = {k: d for d, k in keys.items()}
    kept = [by_key[k] for k in ALWAYS_KEEP]
    dropped = {}
    for d in sorted(live, key=lambda d: (-len(released_sets[d]), keys.get(d, d))):
        if d in kept:
            continue
        holder = next((k for k in kept if released_sets[d] <= released_sets[k]), None)
        if holder is None:
            kept.append(d)
        else:
            dropped[d] = holder
    union_all = set().union(*(released_sets[d] for d in live))
    union_kept = set().union(*(released_sets[d] for d in kept))
    assert union_all == union_kept

    entries_with_doi = set().union(*released_sets.values()) if released_sets else set()
    by_title = defaultdict(list)
    for e in bib:
        by_title[(e["title"], e["author"])].append(e["key"])

    report = {
        "released_rows": rows,
        "released_entry_ids": len(released),
        "rows_resolved_from_source_files": resolved_from_files,
        "doi_candidates_from_download": len(set(candidates)),
        "bib_entries": len(bib),
        "dois_with_released_entries": len(live),
        "rows_in_some_cited_dataset": sum(released[e] for e in entries_with_doi),
        "cited_without_released_entries": sorted(d for d in bib_dois if not released_sets[d]),
        "uncited_with_released_entries": sorted(d for d in live if d not in bib_dois),
        "covered_by_other_doi": {d: {"released_entries": len(released_sets[d]), "also_in": covered_by[d][:3],
                                     "bib_keys": [e["key"] for e in bib if e["doi"] == d]} for d in sorted(covered_by)},
        "identical_released_sets": identical,
        "dataset_names_with_several_dois": {n: ds for n, ds in names.items() if len(set(ds)) > 1},
        "bib_same_title_and_authors": [k for k in by_title.values() if len(k) > 1],
        "minimal_citation_set": len(kept),
        "bib_equals_minimal_set": bib_dois == set(kept),
        "redundant_dois": {d: {"bib_key": keys.get(d), "released_entries": len(released_sets[d]),
                               "contained_in": holder, "contained_in_bib_key": keys.get(holder),
                               "identical": released_sets[d] == released_sets[holder]} for d, holder in sorted(dropped.items())},
        "per_doi": per_doi,
    }
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    summary = {k: (v if not isinstance(v, (dict, list)) or len(v) < 40 else f"{len(v)} items") for k, v in report.items() if k != "per_doi"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
