import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

PAPER = "${DEV_ROOT}/paper"
FILES = ["colabfit.bib", "optimade.bib", "extra_datasets.bib"]
OUT = f"{PAPER}/claims/out/bib_verify.json"


def fetch(url, accept="application/json", tries=5, raw=False):
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "mad-core-bib-check (mailto:none)"})
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                return {"status": response.status, "final_url": response.geturl(), "body": body if raw else json.loads(body)}
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                return {"error": f"HTTP {error.code}"}
            if attempt == tries - 1:
                return {"error": f"HTTP {error.code}"}
        except Exception as error:
            if attempt == tries - 1:
                return {"error": str(error)}
        time.sleep(4 * (attempt + 1))


def norm(text):
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text or "").replace("\\_", "_").replace("_", " ").replace("{", "").replace("}", "")
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def families_from_bib(author):
    out = []
    for name in re.split(r"\s+and\s+", author or ""):
        name = name.strip().strip("{}")
        if not name:
            continue
        family = name.split(",")[0] if "," in name else name.split()[-1]
        out.append(norm(family))
    return out


def parse_fields(body):
    """name = {value} or "value" or bare value, values may span lines and nest braces"""
    fields, i, n = {}, 0, len(body)
    while i < n:
        m = re.compile(r"\s*,?\s*(\w[\w-]*)\s*=\s*").match(body, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if i < n and body[i] in "{\"":
            close = "}" if body[i] == "{" else "\""
            depth, j = 0, i
            while j < n:
                c = body[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                if (close == "}" and depth == 0 and c == "}") or (close == "\"" and c == "\"" and j > i and depth == 0):
                    break
                j += 1
            value, i = body[i + 1:j], j + 1
        else:
            m = re.compile(r"[^,\n]*").match(body, i)
            value, i = m.group(0), m.end()
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


def clean_doi(doi):
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", (doi or "").strip(), flags=re.I)


def parse(path):
    text = open(path).read()
    entries = []
    for comment, kind, key, body in re.findall(r"((?:%[^\n]*\n)*)@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = parse_fields(body)
        if "doi" in fields:
            fields["doi"] = clean_doi(fields["doi"])
        # an eprint is an arXiv id only when it looks like one, some entries keep a pdf link there
        if "eprint" in fields and not re.fullmatch(r"(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?", fields["eprint"]):
            fields["eprint_not_arxiv"] = fields.pop("eprint")
        meta = dict(re.findall(r"%\s*([\w-]+):\s*(\S+)", comment))
        entries.append({"file": path.rsplit("/", 1)[1], "key": key, "type": kind.lower(), "comment": meta, **fields})
    return entries


def check(e):
    r = {"file": e["file"], "key": e["key"], "checks": {}, "detail": {}}
    c, d = r["checks"], r["detail"]
    doi = e.get("doi", "").strip()
    if doi:
        datacite = (doi.startswith(("10.60732", "10.48550", "10.24435", "10.5281", "10.6084"))
                    or doi.upper().startswith("10.17172") or "zenodo" in doi.lower())
        if not datacite:
            m = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}")
            datacite = m.get("error") == "HTTP 404"
        if datacite:
            m = fetch(f"https://api.datacite.org/dois/{urllib.parse.quote(doi, safe='')}")
            a = (m.get("body") or {}).get("data", {}).get("attributes", {}) if "error" not in m else {}
            title = (a.get("titles") or [{}])[0].get("title", "")
            fam = [norm(x.get("familyName") or x.get("name", "").split(",")[0]) for x in a.get("creators", [])]
            d["registered_names"] = [x.get("name", "") or f"{x.get('familyName','')}, {x.get('givenName','')}" for x in a.get("creators", [])]
            year = str(a.get("publicationYear") or "")
            d["registered_url"] = a.get("url", "")
            d["rights"] = [x.get("rightsIdentifier") or x.get("rights") for x in a.get("rightsList", [])]
            source = "DataCite"
        else:
            a = (m.get("body") or {}).get("message", {}) if "error" not in m else {}
            title = (a.get("title") or [""])[0]
            fam = [norm(x.get("family", x.get("name", ""))) for x in a.get("author", [])]
            d["registered_names"] = [f"{x.get('family', x.get('name',''))}, {x.get('given','')}" for x in a.get("author", [])]
            parts = (a.get("issued") or {}).get("date-parts") or [[None]]
            year = str(parts[0][0] or "")
            d["journal"] = (a.get("container-title") or [""])[0]
            source = "Crossref"
        c["doi_resolves"] = bool(a)
        d["source"], d["registered_title"], d["registered_year"] = source, title, year
        d["error"] = m.get("error", "")
        if "doi.org/" in e.get("url", ""):
            c["url_matches_doi"] = clean_doi(e["url"]).lower() == doi.lower()
            d["bib_url"] = e["url"]
    elif e.get("eprint"):
        m = fetch(f"https://export.arxiv.org/api/query?id_list={e['eprint']}", accept="application/atom+xml", raw=True)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        title, fam, year = "", [], ""
        if "error" not in m:
            root = ET.fromstring(m["body"])
            entry = root.find("a:entry", ns)
            if entry is not None and entry.find("a:title", ns) is not None:
                title = " ".join(entry.find("a:title", ns).text.split())
                fam = [norm(x.find("a:name", ns).text.split()[-1]) for x in entry.findall("a:author", ns)]
                year = entry.find("a:published", ns).text[:4]
        c["doi_resolves"] = bool(title)
        d["source"], d["registered_title"], d["registered_year"] = "arXiv", title, year
    else:
        m = fetch(e.get("url", ""), accept="text/html", raw=True) if e.get("url") else {"error": "no doi, eprint or url"}
        c["url_resolves"] = "error" not in m
        d["source"], d["error"] = "url", m.get("error", "")
        return r

    if c["doi_resolves"]:
        c["title_matches"] = norm(title) == norm(e.get("title", "")) or (len(norm(title)) > 30 and norm(e.get("title", "")).startswith(norm(title)[:30]))
        bib_fam = families_from_bib(e.get("author", ""))
        def tokens(n):
            n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower()
            return {t for t in re.split(r"[^a-z]+", n) if len(t) > 1 and t not in ("jr", "sr", "iii")}
        bib_tokens = [tokens(a) for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()]
        reg = d.get("registered_names", [])
        missing = [n for n in reg if n.strip(", ") and not any(tokens(n) and (tokens(n) <= b or b <= tokens(n) or len(tokens(n) & b) >= 2 or (len(tokens(n)) == 1 and tokens(n) & b)) for b in bib_tokens)]
        c["authors_match"] = not missing and (not reg or abs(len(reg) - len(bib_tokens)) <= max(1, len(reg) // 10))
        d["bib_authors"], d["registered_authors"], d["registered_not_in_bib"] = bib_fam[:8], fam[:8], missing[:6]
        d["n_bib_authors"], d["n_registered_authors"] = len(bib_tokens), len(reg)
        c["year_matches"] = not year or year == e.get("year", "")
        d["bib_title"], d["bib_year"] = e.get("title", ""), e.get("year", "")
        if e["file"] == "colabfit.bib":
            ds = e["comment"].get("colabfit-id", "").split("_0")[0]
            c["colabfit_id_in_landing_url"] = bool(ds) and ds.lower() in (d.get("registered_url", "")).lower()
            d["bib_licence"] = (re.search(r"License:\s*(.+?)\.?\s*$", e.get("note", "")) or [None, ""])[1]
    if e["file"] == "colabfit.bib":
        ds = e["comment"].get("colabfit-id", "")
        page = fetch(f"https://materials.colabfit.org/id/{ds}", accept="text/html", raw=True)
        html = page.get("body", b"").decode("utf-8", "replace") if "error" not in page else ""
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
        shown_lic = (re.search(r"[Ll]icen[cs]e\s*:?\s*([A-Za-z0-9.\-]+)", text) or [None, ""])[1]
        d["colabfit_page_licence"], d["colabfit_page_dois"] = shown_lic, sorted(set(re.findall(r"10\.60732/[0-9a-z]+", html)))
        c["colabfit_page_exists"] = bool(html)
        c["doi_shown_on_colabfit_page"] = doi in d["colabfit_page_dois"]
        bib_lic = (re.search(r"License:\s*(.+?)\.?\s*$", e.get("note", "")) or [None, ""])[1]
        c["licence_matches_colabfit_page"] = norm(bib_lic) == norm(shown_lic)
        if not c.get("doi_resolves", True):
            # an unregistered ColabFit DOI is reported separately, the remaining checks need DataCite metadata
            c["doi_resolves"] = False
    return r


def main():
    global FILES, OUT
    if len(sys.argv) > 1:
        FILES = sys.argv[1:]
        OUT = f"{PAPER}/claims/out/bib_verify_{'_'.join(f.replace('.bib', '') for f in FILES)}.json"
    entries = [e for f in FILES for e in parse(f"{PAPER}/biblio/{f}")]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(check, entries))
    failed = [r for r in results if not all(r["checks"].values())]
    summary = {f: {"entries": sum(1 for r in results if r["file"] == f),
                   "passed": sum(1 for r in results if r["file"] == f and all(r["checks"].values()))} for f in FILES}
    counts = {}
    for r in failed:
        for k, v in r["checks"].items():
            if not v:
                counts[k] = counts.get(k, 0) + 1
    with open(OUT, "w") as f:
        json.dump({"summary": summary, "failures_by_check": counts, "failed": failed, "all": results}, f, indent=1, ensure_ascii=False)
    print(json.dumps({"summary": summary, "failures_by_check": counts}, indent=1))
    for r in failed:
        bad = [k for k, v in r["checks"].items() if not v]
        dd = r["detail"]
        print(f"{r['file'][:10]} {r['key']:>22} {bad} | bib: {dd.get('bib_title','')[:50]!r} {dd.get('bib_year','')} {dd.get('bib_authors','')[:3]} "
              f"| reg: {dd.get('registered_title','')[:50]!r} {dd.get('registered_year','')} {dd.get('registered_authors','')[:3]} "
              f"{dd.get('registered_url','')[:60]} {dd.get('rights','')} {dd.get('bib_licence','')} {dd.get('error','')}")


if __name__ == "__main__":
    main()
