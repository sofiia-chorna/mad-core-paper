#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm import tqdm

# config

PROVIDERS_URL = "https://providers.optimade.org/v1/links"

# provider IDs to skip entirely
EXCLUDE_PROVIDER_IDS = {
    "aflow",
    "mp",
    "mpds",
    "gnome",
    "exmpl",
    "optimade",
    "matmatch",
    "nmd",
}

# sub-database URL / ID patterns to skip (matched against base_url AND db_id)
EXCLUDE_DB_PATTERNS = [
    re.compile(r"alexandria", re.IGNORECASE),
    re.compile(r"mc3d", re.IGNORECASE),
    re.compile(r"materials[-_]?project", re.IGNORECASE),
    re.compile(r"/aflow", re.IGNORECASE),
    re.compile(r"mpds", re.IGNORECASE),
    re.compile(r"gnome", re.IGNORECASE),
    re.compile(r"google[-_]?deepmind", re.IGNORECASE),
]

# only request these OPTIMADE-standard structural properties
STRUCTURAL_FIELDS = [
    "lattice_vectors",
    "cartesian_site_positions",
    "species",
    "species_at_sites",
    "dimension_types",
    "nperiodic_dimensions",
    "nsites",
    "chemical_formula_reduced",
    "chemical_formula_descriptive",
    "elements",
    "nelements",
    "structure_features",
]

PAGE_LIMIT = 100  # structures per page (many servers cap at 100)
MAX_CONCURRENT_TOTAL = 8  # total concurrent HTTP requests across all DBs
REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff base (seconds)

OUTPUT_DIR = Path("optimade_data")
STATE_FILE = OUTPUT_DIR / "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimade_dl")


# state management for restart/resume

def db_key(provider_id: str, db_id: str) -> str:
    return f"{provider_id}::{db_id}"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Corrupted state file, starting fresh: %s", exc)
    return {}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(STATE_FILE)


# data classes

@dataclass
class DatabaseInfo:
    provider_id: str
    provider_name: str
    db_id: str
    db_name: str
    base_url: str
    n_structures: Optional[int] = None


@dataclass
class DownloadStats:
    db: DatabaseInfo
    n_downloaded: int = 0
    n_skipped: int = 0
    n_errors: int = 0
    resumed: bool = False
    elapsed: float = 0.0


# helpers

def should_exclude(url: str, db_id: str = "") -> bool:
    text = f"{url} {db_id}"
    return any(p.search(text) for p in EXCLUDE_DB_PATTERNS)


def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def has_structural_data(attrs: dict) -> bool:
    lv = attrs.get("lattice_vectors")
    pos = attrs.get("cartesian_site_positions")
    if not lv and not pos:
        return False
    return True


def extract_next_url(page_data: dict) -> Optional[str]:
    links = page_data.get("links", {})
    nxt = links.get("next")
    if isinstance(nxt, str) and nxt:
        return nxt
    if isinstance(nxt, dict) and nxt.get("href"):
        return nxt["href"]
    return None


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = MAX_RETRIES,
) -> Optional[dict]:
    for attempt in range(retries):
        try:
            async with semaphore:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as resp:
                    if resp.status == 429:
                        wait = RETRY_BACKOFF ** (attempt + 1)
                        log.warning("Rate-limited on %s, waiting %.1fs", url, wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 400:
                        text = await resp.text()
                        log.warning(
                            "HTTP %d from %s: %.200s", resp.status, url, text
                        )
                        return None
                    return await resp.json(content_type=None)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            wait = RETRY_BACKOFF ** (attempt + 1)
            log.warning(
                "Attempt %d/%d failed for %s (%s), retrying in %.1fs",
                attempt + 1,
                retries,
                url,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    log.error("All retries exhausted for %s", url)
    return None


# provider / database discovery

async def discover_providers(session, sem) -> list[dict]:
    data = await fetch_json(session, PROVIDERS_URL, sem)
    if data is None:
        raise RuntimeError("Cannot fetch providers list from " + PROVIDERS_URL)
    return data.get("data", [])


async def discover_sub_databases(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    provider: dict,
) -> list[DatabaseInfo]:
    pid = provider["id"]
    attrs = provider.get("attributes", {})
    pname = attrs.get("name", pid)
    base_url = (attrs.get("base_url") or "").rstrip("/")

    if not base_url:
        log.info("Provider %s has no base_url, skipping", pid)
        return []

    # the index meta-database lists sub-databases at /v1/links
    links_url = f"{base_url}/v1/links"
    data = await fetch_json(session, links_url, sem)

    databases = []
    if data is None:
        # maybe this provider serves structures directly (no sub-databases)
        test_url = f"{base_url}/v1/structures?page_limit=1"
        test = await fetch_json(session, test_url, sem)
        if test and "data" in test:
            db = DatabaseInfo(
                provider_id=pid,
                provider_name=pname,
                db_id=pid,
                db_name=pname,
                base_url=base_url,
            )
            meta = test.get("meta", {})
            db.n_structures = meta.get("data_available")
            databases.append(db)
        return databases

    for entry in data.get("data", []):
        link_attrs = entry.get("attributes", {})
        link_type = link_attrs.get("link_type", "")
        db_base = (link_attrs.get("base_url") or "").rstrip("/")

        if link_type != "child" or not db_base:
            continue

        entry_id = entry.get("id", "unknown")
        entry_name = link_attrs.get("name", entry_id)

        if should_exclude(db_base, entry_id):
            log.info("Excluding sub-database %s/%s by pattern", pid, entry_id)
            continue

        databases.append(
            DatabaseInfo(
                provider_id=pid,
                provider_name=pname,
                db_id=entry_id,
                db_name=entry_name,
                base_url=db_base,
            )
        )

    return databases


async def discover_all_databases(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
) -> list[DatabaseInfo]:
    providers = await discover_providers(session, sem)
    log.info("Found %d providers in the OPTIMADE registry", len(providers))

    tasks = []
    for p in providers:
        pid = p["id"]
        attrs = p.get("attributes", {})
        base_url = attrs.get("base_url") or ""

        if pid in EXCLUDE_PROVIDER_IDS:
            log.info("Excluding provider: %s (by ID)", pid)
            continue

        if should_exclude(base_url, pid):
            log.info("Excluding provider: %s (by URL pattern)", pid)
            continue

        tasks.append(discover_sub_databases(session, sem, p))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    databases = []
    for r in results:
        if isinstance(r, Exception):
            log.warning("Error discovering sub-databases: %s", r)
            continue
        databases.extend(r)

    return databases


# structures download (with resume support)

def build_structures_url(base_url: str, page_offset: int = 0) -> str:
    fields = ",".join(STRUCTURAL_FIELDS)
    url = (
        f"{base_url}/v1/structures"
        f"?response_fields={fields}"
        f"&page_limit={PAGE_LIMIT}"
    )
    if page_offset > 0:
        url += f"&page_offset={page_offset}"
    return url


async def download_database(
    session: aiohttp.ClientSession,
    global_sem: asyncio.Semaphore,
    db: DatabaseInfo,
    state: dict,
    pbar: tqdm,
) -> DownloadStats:
    stats = DownloadStats(db=db)
    t0 = time.monotonic()
    key = db_key(db.provider_id, db.db_id)
    outfile = OUTPUT_DIR / f"{safe_filename(db.provider_id)}__{safe_filename(db.db_id)}.jsonl"

    db_state = state.get(key, {})

    # already completed in a previous run
    if db_state.get("status") == "done":
        stats.n_downloaded = db_state.get("n_downloaded", 0)
        stats.n_skipped = db_state.get("n_skipped", 0)
        stats.resumed = True
        stats.elapsed = 0.0
        pbar.update(stats.n_downloaded)
        log.info(
            "DB %s/%s: already done (%d structures), skipping",
            db.provider_id, db.db_id, stats.n_downloaded,
        )
        return stats

    # resume from where we left off
    resume_url = db_state.get("next_url")
    previously_downloaded = db_state.get("n_downloaded", 0)
    previously_skipped = db_state.get("n_skipped", 0)

    if resume_url and previously_downloaded > 0:
        stats.resumed = True
        stats.n_downloaded = previously_downloaded
        stats.n_skipped = previously_skipped
        pbar.update(previously_downloaded)
        log.info(
            "DB %s/%s: resuming from %d structures (next: %.80s...)",
            db.provider_id, db.db_id, previously_downloaded, resume_url,
        )
        first_url = resume_url
        file_mode = "a"  # append to existing JSONL
    else:
        first_url = build_structures_url(db.base_url)
        file_mode = "w"

    # first page (or resume page)
    page_data = await fetch_json(session, first_url, global_sem)
    if page_data is None:
        log.warning("Cannot reach %s, saving state for retry", db.base_url)
        # preserve existing in-progress state so next run retries
        if resume_url:
            state[key] = {
                "status": "in_progress",
                "n_downloaded": stats.n_downloaded,
                "n_skipped": stats.n_skipped,
                "next_url": resume_url,
                "base_url": db.base_url,
            }
            save_state(state)
        stats.n_errors += 1
        stats.elapsed = time.monotonic() - t0
        return stats

    meta = page_data.get("meta", {})
    n_available = meta.get("data_available", 0)
    db.n_structures = n_available
    if not stats.resumed:
        log.info(
            "DB %s/%s: %s structures available",
            db.provider_id, db.db_id, n_available or "unknown",
        )

    if n_available == 0 and not page_data.get("data"):
        state[key] = {
            "status": "done",
            "n_downloaded": 0,
            "n_skipped": 0,
            "next_url": None,
            "base_url": db.base_url,
        }
        save_state(state)
        stats.elapsed = time.monotonic() - t0
        return stats

    # paginate and write
    page_num = 0
    save_interval = 5  # persist state every N pages

    with open(outfile, file_mode) as fh:
        while True:
            entries = page_data.get("data", [])
            if not entries:
                break

            for entry in entries:
                attrs = entry.get("attributes", {})

                # skip hollow entries (no actual structural data)
                if not has_structural_data(attrs):
                    stats.n_skipped += 1
                    continue

                attrs["_optimade_id"] = entry.get("id")
                attrs["_provider"] = db.provider_id
                attrs["_database"] = db.db_id
                fh.write(json.dumps(attrs, separators=(",", ":")) + "\n")
                stats.n_downloaded += 1
                pbar.update(1)

            # get next page URL
            next_url = extract_next_url(page_data)

            # periodically save state for resumability
            page_num += 1
            if page_num % save_interval == 0:
                fh.flush()
                state[key] = {
                    "status": "in_progress",
                    "n_downloaded": stats.n_downloaded,
                    "n_skipped": stats.n_skipped,
                    "next_url": next_url,
                    "base_url": db.base_url,
                }
                save_state(state)
                log.debug(
                    "DB %s/%s: checkpoint at %d structures",
                    db.provider_id, db.db_id, stats.n_downloaded,
                )

            if not next_url:
                break

            page_data = await fetch_json(session, next_url, global_sem)
            if page_data is None:
                stats.n_errors += 1
                log.warning(
                    "Pagination error at page %d for %s/%s, will retry on restart",
                    page_num, db.provider_id, db.db_id,
                )
                # save state pointing at the failed page so next run retries it
                fh.flush()
                state[key] = {
                    "status": "in_progress",
                    "n_downloaded": stats.n_downloaded,
                    "n_skipped": stats.n_skipped,
                    "next_url": next_url,
                    "base_url": db.base_url,
                }
                save_state(state)
                break

    # mark as done only if we exhausted pagination without errors
    if stats.n_errors == 0:
        state[key] = {
            "status": "done",
            "n_downloaded": stats.n_downloaded,
            "n_skipped": stats.n_skipped,
            "next_url": None,
            "base_url": db.base_url,
        }
        save_state(state)

    stats.elapsed = time.monotonic() - t0
    return stats


# main

async def main():
    parser = argparse.ArgumentParser(description="OPTIMADE bulk structure downloader")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore previous state and start from scratch",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.fresh and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("Removed previous state file (--fresh)")

    state = load_state()
    if state:
        n_done = sum(1 for v in state.values() if v.get("status") == "done")
        n_prog = sum(1 for v in state.values() if v.get("status") == "in_progress")
        log.info(
            "Loaded state: %d done, %d in-progress (will resume)",
            n_done, n_prog,
        )

    global_sem = asyncio.Semaphore(MAX_CONCURRENT_TOTAL)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_TOTAL * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Discover databases
        log.info("Discovering OPTIMADE providers and sub-databases...")
        databases = await discover_all_databases(session, global_sem)
        log.info("Found %d databases to download from", len(databases))

        for db in databases:
            key = db_key(db.provider_id, db.db_id)
            status = state.get(key, {}).get("status", "new")
            log.info(
                "  -> %s / %s : %s [%s]",
                db.provider_id, db.db_id, db.base_url, status,
            )

        if not databases:
            log.warning("No databases found. Exiting.")
            return

        # 2. Download structures concurrently
        pbar = tqdm(desc="Structures downloaded", unit="struct")

        tasks = [
            download_database(session, global_sem, db, state, pbar)
            for db in databases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()

        # 3. Summary
        print("\n" + "=" * 80)
        print("DOWNLOAD SUMMARY")
        print("=" * 80)
        total_dl = 0
        total_skip = 0
        for r in results:
            if isinstance(r, Exception):
                log.error("Task failed: %s", r)
                continue
            total_dl += r.n_downloaded
            total_skip += r.n_skipped
            if r.resumed and r.n_errors == 0 and r.elapsed < 0.1:
                tag = "cached"
            elif r.resumed:
                tag = "resumed"
            elif r.n_errors > 0:
                tag = f"{r.n_errors} errors"
            else:
                tag = "OK"
            skip_str = f" ({r.n_skipped} hollow)" if r.n_skipped else ""
            print(
                f"  {r.db.provider_id:>16s} / {r.db.db_id:<24s} "
                f"{r.n_downloaded:>8d} structures{skip_str:<16s} "
                f"({r.elapsed:>6.1f}s)  [{tag}]"
            )
        print(
            f"\n  TOTAL: {total_dl} structures downloaded, "
            f"{total_skip} hollow entries skipped"
        )
        print(f"  Output directory: {OUTPUT_DIR.resolve()}")
        print(f"  State file: {STATE_FILE.resolve()}")

        n_incomplete = sum(
            1 for v in state.values() if v.get("status") == "in_progress"
        )
        if n_incomplete:
            print(f"\n  {n_incomplete} database(s) incomplete. Re-run to resume.")


if __name__ == "__main__":
    asyncio.run(main())
