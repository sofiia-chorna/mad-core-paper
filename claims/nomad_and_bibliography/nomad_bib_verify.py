import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAPER = "${DEV_ROOT}/paper"
CLAIMS = f"{PAPER}/claims"
OUT = f"{CLAIMS}/out/nomad_bib_verify.json"
API = "https://nomad-lab.eu/prod/v1/api/v1"
SHORT = {"10.17172/NOMAD/PBVEIBCA390AP80E7N8DX5IFZSAHZ8Q3RU8ZIVXT6KPPIMS": "10/f2696t",
         "10.17172/NOMAD/2O4HTSZH38W7JVZEK9E1RD8CVUBA6541K6T8V3DLQFAMJMS": "10/z9c"}


def get(url, accept="application/json", data=None, tries=5):
    for attempt in range(tries):
        try:
            headers = {"Accept": accept}
            if data is not None:
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {"error": "404"}
            if attempt == tries - 1:
                return {"error": str(error)}
        except Exception as error:
            if attempt == tries - 1:
                return {"error": str(error)}
        time.sleep(5 * (attempt + 1))


def norm(text):
    text = text.replace("\\&", "&").replace("\\textasciicircum{}", "^").replace("_", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def name_tokens(name):
    # "Family, Given" or "Given Family" -> set of normalised words, suffixes dropped
    words = re.split(r"[\s,]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())
    return {re.sub(r"[^a-z]", "", w) for w in words if re.sub(r"[^a-z]", "", w) not in ("", "jr", "sr")}


def bib_entries():
    text = open(f"{PAPER}/biblio/nomad.bib").read()
    out = []
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", body, re.M))
        out.append({"key": key.strip(), **fields})
    return out


def released_by_doi():
    with open(f"{CLAIMS}/out/nomad_provenance/nomad_dois.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {r["dataset_doi"].upper(): r for r in rows}


def check_dataset(entry, released):
    doi = entry["doi"]
    stored = SHORT.get(doi.upper(), released.get(doi.upper(), {}).get("dataset_doi", doi))
    result = {"key": entry["key"], "doi": doi, "checks": {}, "detail": {}}
    c, d = result["checks"], result["detail"]

    resolved = get(f"https://doi.org/{urllib.parse.quote(doi, safe='/')}", accept="application/vnd.citationstyles.csl+json")
    c["doi_resolves"] = "error" not in resolved
    d["datacite_title"] = resolved.get("title", "")

    query = get(f"{API}/entries/query", data={
        "query": {"datasets.doi": stored}, "pagination": {"page_size": 1},
        "required": {"include": ["datasets.doi", "datasets.dataset_name", "datasets.dataset_id"]},
        "aggregations": {"au": {"terms": {"quantity": "authors.name", "size": 500}}}})
    entries_total = (query.get("pagination") or {}).get("total", 0)
    c["nomad_dataset_has_entries"] = entries_total > 0
    d["nomad_entries"] = entries_total
    names = []
    if query.get("data"):
        names = [x.get("dataset_name", "") for x in query["data"][0].get("datasets", []) if (x.get("doi") or "").upper() == stored.upper()]
    d["nomad_dataset_name"] = names[0] if names else ""
    title = entry.get("title", "").replace("NOMAD dataset: ", "", 1)
    c["title_matches_nomad_name"] = bool(names) and norm(title) == norm(names[0])

    nomad_authors = [b["value"] for b in (query.get("aggregations", {}).get("au", {}).get("terms", {}).get("data", []))]
    d["nomad_authors"] = nomad_authors
    bib_authors = [a.strip().strip("{}") for a in entry.get("author", "").split(" and ") if a.strip()]
    unmatched = []
    for a in bib_authors:
        tokens = name_tokens(a)
        if not any(tokens and tokens <= name_tokens(n) for n in nomad_authors):
            unmatched.append(a)
    c["all_bib_authors_in_nomad"] = not unmatched
    d["bib_authors_not_in_nomad"] = unmatched

    r = released.get(doi.upper()) or released.get(stored.upper())
    c["has_released_records"] = bool(r) and int(r["released_records"]) > 0
    d["released_records"] = int(r["released_records"]) if r else 0
    return result


def check_source(key, doi, expected):
    meta = get(f"https://doi.org/{doi}", accept="application/vnd.citationstyles.csl+json")
    families = [a.get("family", a.get("literal", "")) for a in meta.get("author", [])]
    return {"key": key, "doi": doi, "resolves": "error" not in meta, "title": meta.get("title", ""),
            "title_matches": norm(meta.get("title", "")).startswith(norm(expected["title"])[:40]),
            "authors": families, "authors_match": [norm(f) for f in families][:len(expected["authors"])] == [norm(f) for f in expected["authors"]],
            "year": ((meta.get("issued") or {}).get("date-parts") or [[None]])[0][0]}


def main():
    entries = bib_entries()
    released = released_by_doi()
    datasets = [e for e in entries if e["key"].isdigit()]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda e: check_dataset(e, released), datasets))

    sources = {e["key"]: e for e in entries if not e["key"].isdigit()}
    expected = {
        "Johnson2022CCCBDB": {"title": "Computational Chemistry Comparison and Benchmark Database", "authors": ["Johnson"]},
        "Gupta2021": {"title": "Revving up 13C NMR shielding predictions", "authors": ["Gupta", "Chakraborty", "Ramakrishnan"]},
        "Grisafi2022": {"title": "Electronic-structure properties from atom-centered predictions", "authors": ["Grisafi", "Lewis", "Rossi", "Ceriotti"]},
    }
    source_results = [check_source(k, sources[k]["doi"], expected[k]) for k in expected if k in sources]

    failed = [r for r in results if not all(r["checks"].values())]
    summary = {"dataset_entries": len(results),
               "passed_all_checks": len(results) - len(failed),
               "failures_by_check": {k: sum(1 for r in results if not r["checks"][k]) for k in results[0]["checks"]},
               "source_entries": source_results}
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "failed": failed, "all": results}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    for r in failed:
        bad = [k for k, v in r["checks"].items() if not v]
        print(f"{r['key']:>4} {r['doi'][:42]:42} {bad} | bib-not-in-nomad {r['detail']['bib_authors_not_in_nomad']} | "
              f"nomad name {r['detail']['nomad_dataset_name'][:60]!r} | nomad authors {r['detail']['nomad_authors'][:6]}")


if __name__ == "__main__":
    main()
