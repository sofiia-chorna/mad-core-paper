#!/usr/bin/env python3
"""
Probe OPTIMADE providers for license metadata.

For each provider (and each of its sub-databases), hits:
  - <base_url>/v1/info                  -> attributes.license,
                                           available_licenses,
                                           available_licenses_for_entries
  - <base_url>/v1/structures?page_limit=1
                                        -> meta.license (sometimes present
                                           even when /info isn't updated)

Provider list is auto-discovered from the central registry, OR taken from
filenames in an optimade_data/ directory (matching the format produced by
the bulk downloader: <provider>__<database>.jsonl).

Usage
-----
    pip install requests
    python optimade_license_probe.py                       # probe all registry providers
    python optimade_license_probe.py --dir optimade_data/  # probe only providers we have data for
    python optimade_license_probe.py --json out.json       # also dump raw JSON results

Output is a table to stdout. SPDX-identifier columns are most useful for
deciding redistribution compatibility (e.g. CC-BY-4.0 is fine to redistribute
under the same; CC-BY-NC-* prevents commercial use; proprietary databases
will show 'unknown' or be missing entirely).
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

PROVIDERS_URL = "https://providers.optimade.org/v1/links"
TIMEOUT = 30


def get_json(url: str) -> Optional[dict]:
    try:
        r = requests.get(
            url, timeout=TIMEOUT,
            headers={"Accept": "application/vnd.api+json"},
        )
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None


def discover_providers() -> list[dict]:
    data = get_json(PROVIDERS_URL)
    if not data:
        sys.exit(f"Cannot reach providers registry at {PROVIDERS_URL}")
    return data.get("data", [])


def discover_sub_databases(provider: dict) -> list[dict]:
    """Return list of {provider_id, db_id, base_url} dicts."""
    pid = provider["id"]
    pattrs = provider.get("attributes", {})
    base_url = (pattrs.get("base_url") or "").rstrip("/")
    if not base_url:
        return []

    out = []
    links = get_json(f"{base_url}/v1/links")
    if links and "data" in links:
        for entry in links["data"]:
            la = entry.get("attributes", {})
            if la.get("link_type") != "child":
                continue
            db_base = (la.get("base_url") or "").rstrip("/")
            if db_base:
                out.append({
                    "provider_id": pid,
                    "db_id": entry.get("id", "unknown"),
                    "base_url": db_base,
                })
    if not out:
        # Provider serves structures directly (no sub-databases)
        out.append({
            "provider_id": pid,
            "db_id": pid,
            "base_url": base_url,
        })
    return out


def extract_license_info(base_url: str) -> dict:
    """Probe one base_url for license fields. Returns a dict of findings."""
    result: dict = {
        "base_url": base_url,
        "info_license": None,
        "info_available_licenses": None,
        "info_available_licenses_for_entries": None,
        "meta_license": None,
        "info_reachable": False,
        "structures_reachable": False,
    }

    info = get_json(f"{base_url}/v1/info")
    if info:
        result["info_reachable"] = True
        attrs = (info.get("data", {}) or {}).get("attributes", {}) or {}
        # `license` may be either a string URL or a JSON API link object {href, ...}
        lic = attrs.get("license")
        if isinstance(lic, dict):
            lic = lic.get("href")
        result["info_license"] = lic
        result["info_available_licenses"] = attrs.get("available_licenses")
        result["info_available_licenses_for_entries"] = (
            attrs.get("available_licenses_for_entries")
        )

    structs = get_json(f"{base_url}/v1/structures?page_limit=1")
    if structs:
        result["structures_reachable"] = True
        meta = structs.get("meta", {}) or {}
        result["meta_license"] = meta.get("license")

    return result


def parse_optimade_dir(path: Path) -> set[str]:
    """Return set of provider_ids based on jsonl filenames."""
    pids = set()
    for f in path.glob("*.jsonl"):
        m = re.match(r"^([^_]+)__", f.name)
        if m:
            pids.add(m.group(1))
    return pids


def fmt_license(result: dict) -> str:
    """One-line license summary for table display."""
    spdx = result.get("info_available_licenses_for_entries")
    if spdx:
        return ", ".join(spdx) + " (entries)"
    spdx = result.get("info_available_licenses")
    if spdx:
        return ", ".join(spdx) + " (db)"
    if result.get("info_license"):
        return f"URL: {result['info_license'][:60]}"
    if result.get("meta_license"):
        return f"meta: {result['meta_license'][:60]}"
    if not result.get("info_reachable") and not result.get("structures_reachable"):
        return "(unreachable)"
    return "(no license metadata)"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=None,
                   help="Restrict to provider IDs found as filenames in this "
                        "directory (matches optimade_data/ layout)")
    p.add_argument("--json", type=Path, default=None,
                   help="Also write full raw results to this JSON file")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    print("Discovering providers...")
    providers = discover_providers()
    print(f"Found {len(providers)} providers in registry")

    restrict = None
    if args.dir:
        if not args.dir.is_dir():
            sys.exit(f"--dir must be a directory: {args.dir}")
        restrict = parse_optimade_dir(args.dir)
        print(f"Restricting to {len(restrict)} provider IDs from {args.dir}: "
              f"{', '.join(sorted(restrict))}")

    # Expand to (provider, sub-database) pairs
    targets: list[dict] = []
    for prov in providers:
        if restrict is not None and prov["id"] not in restrict:
            continue
        for db in discover_sub_databases(prov):
            targets.append(db)

    print(f"Probing {len(targets)} (provider, database) endpoints...\n")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(extract_license_info, t["base_url"]):
                t for t in targets
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                lic = fut.result()
            except Exception as exc:
                lic = {"base_url": t["base_url"], "error": str(exc)}
            lic.update({"provider_id": t["provider_id"], "db_id": t["db_id"]})
            results.append(lic)

    # Sort for stable table output
    results.sort(key=lambda r: (r.get("provider_id", ""), r.get("db_id", "")))

    # Pretty table
    pid_w = max((len(r.get("provider_id", "")) for r in results), default=8)
    db_w = max((len(r.get("db_id", "")) for r in results), default=8)
    pid_w = max(pid_w, 8)
    db_w = max(db_w, 8)

    print(f"{'PROVIDER':<{pid_w}}  {'DATABASE':<{db_w}}  LICENSE")
    print("-" * (pid_w + db_w + 60))
    for r in results:
        print(
            f"{r.get('provider_id', ''):<{pid_w}}  "
            f"{r.get('db_id', ''):<{db_w}}  "
            f"{fmt_license(r)}"
        )

    # Quick summary by category
    n_total = len(results)
    n_spdx_entries = sum(
        1 for r in results if r.get("info_available_licenses_for_entries")
    )
    n_spdx_db = sum(
        1 for r in results
        if r.get("info_available_licenses")
        and not r.get("info_available_licenses_for_entries")
    )
    n_url_only = sum(
        1 for r in results
        if r.get("info_license")
        and not r.get("info_available_licenses")
        and not r.get("info_available_licenses_for_entries")
    )
    n_unreachable = sum(
        1 for r in results
        if not r.get("info_reachable") and not r.get("structures_reachable")
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total endpoints probed:                 {n_total}")
    print(f"  with SPDX entries license:            {n_spdx_entries}")
    print(f"  with SPDX database license:           {n_spdx_db}")
    print(f"  with license URL only:                {n_url_only}")
    print(f"  with no license metadata:             "
          f"{n_total - n_spdx_entries - n_spdx_db - n_url_only - n_unreachable}")
    print(f"  unreachable:                          {n_unreachable}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
