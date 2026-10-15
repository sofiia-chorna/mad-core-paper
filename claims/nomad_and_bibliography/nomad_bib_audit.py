import json
import re
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAPER = "${DEV_ROOT}/paper"
OUT = f"{PAPER}/claims/out/nomad_bib_audit.json"
URL_BY_DOI = "https://nomad-lab.eu/prod/v1/gui/dataset/doi/{}"


def entries():
    text = open(f"{PAPER}/biblio/nomad.bib").read()
    out = []
    for key, body in re.findall(r"@misc\{(\d+),(.*?)\n\}", text, re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*)\}\s*,?\s*$", body, re.M))
        out.append({"key": key, **fields})
    return out


def datacite(doi):
    try:
        request = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": "application/vnd.citationstyles.csl+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return doi, json.load(response)
    except Exception as error:
        return doi, {"error": str(error)}


def norm(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("nomad dataset", ""))


def main():
    bib = entries()
    with ThreadPoolExecutor(8) as pool:
        meta = dict(pool.map(datacite, [e["doi"] for e in bib]))

    issues = []

    def flag(e, kind, detail):
        issues.append({"key": e["key"], "doi": e["doi"], "kind": kind, "detail": detail})

    for e in bib:
        doi, url, title, author, year = e.get("doi", ""), e.get("url", ""), e.get("title", ""), e.get("author", ""), e.get("year", "")
        # format checks
        if not doi.upper().startswith("10.17172/NOMAD"):
            flag(e, "doi not a NOMAD DOI", doi)
        if re.fullmatch(r"10\.17172/NOMAD/[0-9A-Z]{30,}", doi.upper()):
            flag(e, "legacy hash-style DOI", doi)
        if url != URL_BY_DOI.format(doi) and "/gui/dataset/id/" not in url:
            flag(e, "url does not match doi", url)
        if not title.startswith("NOMAD dataset: "):
            flag(e, "title without the NOMAD dataset prefix", title)
        stripped = re.sub(r"\\[_%&#$]", "", title)
        if re.search(r"[_%&#]|(?<!\\)\$|\^|~", stripped):
            flag(e, "LaTeX special character in title", title)
        if title.count("{") != title.count("}"):
            flag(e, "unbalanced braces in title", title)
        if not re.fullmatch(r"(19|20)\d\d", year or ""):
            flag(e, "odd year", year)
        names = [a.strip() for a in author.split(" and ")] if author else []
        if not names:
            flag(e, "no author", "")
        for name in names:
            if re.search(r"admin|university|institute|universit|group|lab\b|team|project|center|centre", name, re.I):
                flag(e, "organisation or account as author", name)
            if "," not in name and len(name.split()) < 2:
                flag(e, "single-word author", name)
            if name == name.lower() or name == name.upper():
                flag(e, "author not capitalised normally", name)
        if len(set(names)) != len(names):
            flag(e, "repeated author", author)

        # comparison with the registered DataCite metadata
        m = meta.get(doi, {})
        if "error" in m:
            flag(e, "DOI does not resolve", m["error"])
            continue
        registered_title = (m.get("title", "") or "").replace("&amp;", "&")
        generic = re.search(r"NoMaD Repository Entry|^NOMAD dataset:\s*(temp\d*)?$", registered_title, re.I)
        if not generic and norm(registered_title) != norm(title.replace("\\&", "&").replace("\\textasciicircum{}", "^")):
            flag(e, "title differs from DataCite", f"bib: {title} | DataCite: {registered_title}")
        registered_year = str(((m.get("issued") or {}).get("date-parts") or [[""]])[0][0])
        if registered_year and registered_year != year:
            flag(e, "year differs from DataCite", f"bib {year} DataCite {registered_year}")
        # DataCite often registers only the first creator, or a placeholder; every registered person must appear in the bib,
        # and a literal "Given Family" name must not have been split the other way round
        placeholders = {"zastrow", "nomadrepository", "scheidgen"}
        bib_families = {norm(n.split(",")[0]) for n in names}
        bib_full = {norm(n.split(",")[1] + n.split(",")[0]) if "," in n else norm(n) for n in names}
        missing = []
        for a in m.get("author", []):
            if "family" in a:
                if norm(a["family"]) not in placeholders and norm(a["family"]) not in bib_families:
                    missing.append(a["family"])
            elif norm(a.get("literal", "")) not in placeholders and norm(a.get("literal", "")) not in bib_full:
                missing.append(a.get("literal", ""))
        if missing:
            flag(e, "authors differ from DataCite", f"bib: {author} | DataCite: {[a.get('family') or a.get('literal') for a in m.get('author', [])]}")

    kinds = {}
    for i in issues:
        kinds.setdefault(i["kind"], []).append(i)
    report = {"entries": len(bib), "resolved": sum(1 for d in meta.values() if "error" not in d),
              "issues_by_kind": {k: len(v) for k, v in kinds.items()}, "issues": issues}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("entries", "resolved", "issues_by_kind")}, indent=1))
    for kind, items in kinds.items():
        print(f"\n== {kind} ({len(items)})")
        for i in items[:12]:
            print(f"  {i['key']:>4} {i['doi'][:40]:40} {i['detail'][:150]}")


if __name__ == "__main__":
    main()
