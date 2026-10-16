#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from tqdm import tqdm

# configuration

NOMAD_API_BASE = "https://nomad-lab.eu/prod/v1/api/v1"

# what we request from each entry's archive
ARCHIVE_REQUIRED = {
    "run": {
        "system[-1]": {
            "atoms": {
                "positions": "*",
                "lattice_vectors": "*",
                "labels": "*",
                "periodic": "*",
                "species": "*",
            }
        }
    }
}

# metadata fields we always request alongside the archive
METADATA_INCLUDE = [
    "entry_id",
    "upload_id",
    "results.material.chemical_formula_hill",
    "results.material.chemical_formula_reduced",
    "results.material.material_id",
    "results.material.structural_type",
    "results.method.simulation.program_name",
]

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# checkpoint management

@dataclass
class Checkpoint:
    total_downloaded: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    last_page_after_value: Optional[str] = None
    shard_index: int = 0
    query_total: Optional[int] = None
    completed: bool = False

    def save(self, path: Path):
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()


# API interaction

class NomadClient:

    def __init__(
        self,
        base_url: str = NOMAD_API_BASE,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _post_with_retry(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code in (429, 503):
                    wait = self.backoff_base ** attempt
                    log.warning(
                        f"Rate limited ({resp.status_code}), "
                        f"waiting {wait:.0f}s (attempt {attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = self.backoff_base ** attempt
                    log.warning(
                        f"Server error {resp.status_code}, "
                        f"retrying in {wait:.0f}s (attempt {attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                else:
                    log.error(f"HTTP {resp.status_code}: {resp.text[:500]}")
                    resp.raise_for_status()
            except requests.exceptions.Timeout:
                wait = self.backoff_base ** attempt
                log.warning(
                    f"Request timeout, retrying in {wait:.0f}s "
                    f"(attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)
            except requests.exceptions.ConnectionError as e:
                wait = self.backoff_base ** attempt
                log.warning(
                    f"Connection error: {e}, retrying in {wait:.0f}s "
                    f"(attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {url}")

    def count_entries(self, query: dict) -> int:
        payload = {
            "query": query,
            "pagination": {"page_size": 0},
            "required": {"include": ["entry_id"]},
        }
        result = self._post_with_retry("entries/query", payload)
        return result["pagination"]["total"]

    def query_archives_page(
        self,
        query: dict,
        required: dict,
        page_size: int = 100,
        page_after_value: Optional[str] = None,
    ) -> dict:
        pagination: dict[str, Any] = {
            "page_size": page_size,
            "order_by": "entry_id",
            "order": "asc",
        }
        if page_after_value is not None:
            pagination["page_after_value"] = page_after_value

        payload = {
            "query": query,
            "required": required,
            "pagination": pagination,
        }
        return self._post_with_retry("entries/archive/query", payload)


# structure extraction → extxyz lines

def extract_structure_extxyz(entry: dict) -> Optional[str]:
    try:
        archive = entry.get("archive", {})
        runs = archive.get("run", [])
        if not runs:
            return None

        # get the last run's last system (we requested system -1 )
        systems = runs[-1].get("system", []) if runs else []
        if not systems:
            return None

        system = systems[-1] if isinstance(systems, list) else systems
        atoms = system.get("atoms", {})
        if not atoms:
            return None

        # positions NOMAD stores in SI (metres), convert to Angstrom
        positions = atoms.get("positions")
        if positions is None:
            return None
        positions = np.array(positions, dtype=np.float64) * 1e10  # m → Å

        n_atoms = len(positions)
        if n_atoms == 0:
            return None

        # labels (element symbols)
        labels = atoms.get("labels")
        if labels is None:
            # fall back to species
            species = atoms.get("species")
            if species is not None:
                labels = [sp.get("name", "X") if isinstance(sp, dict) else str(sp)
                          for sp in species]
            else:
                return None

        if len(labels) != n_atoms:
            return None

        # lattice vectors NOMAD stores in SI (metres) → Angstrom
        lattice = atoms.get("lattice_vectors")
        pbc = atoms.get("periodic")

        # build the comment/properties line
        entry_id = entry.get("entry_id", "unknown")

        # get metadata from the entry-level fields
        formula = ""
        material_id = ""
        program = ""
        structural_type = ""
        if "results" in entry:
            mat = entry["results"].get("material", {})
            formula = mat.get("chemical_formula_hill", "") or ""
            material_id = mat.get("material_id", "") or ""
            structural_type = mat.get("structural_type", "") or ""
            method = entry["results"].get("method", {})
            sim = method.get("simulation", {})
            program = sim.get("program_name", "") or ""

        lines = io.StringIO()

        # line 1 atom count
        lines.write(f"{n_atoms}\n")

        # line 2 comment line with properties
        comment_parts = []

        if lattice is not None:
            lat = np.array(lattice, dtype=np.float64) * 1e10  # m → Å
            lat_str = " ".join(f"{x:.10f}" for x in lat.flatten())
            comment_parts.append(f'Lattice="{lat_str}"')

        if pbc is not None:
            pbc_str = " ".join("T" if p else "F" for p in pbc)
            comment_parts.append(f'pbc="{pbc_str}"')
        elif lattice is not None:
            # if lattice is given but PBC is not, assume periodic
            comment_parts.append('pbc="T T T"')

        comment_parts.append('Properties=species:S:1:pos:R:3')
        comment_parts.append(f'entry_id={entry_id}')
        if formula:
            comment_parts.append(f'formula={formula}')
        if material_id:
            comment_parts.append(f'material_id={material_id}')
        if program:
            comment_parts.append(f'program={program}')
        if structural_type:
            comment_parts.append(f'structural_type={structural_type}')

        lines.write(" ".join(comment_parts) + "\n")

        # atom lines
        for i in range(n_atoms):
            symbol = labels[i]
            x, y, z = positions[i]
            lines.write(f"{symbol:>4s} {x:20.10f} {y:20.10f} {z:20.10f}\n")

        return lines.getvalue()

    except Exception as e:
        log.debug(f"Failed to extract structure from entry: {e}")
        return None


# shard writer

class ShardWriter:

    def __init__(self, output_dir: Path, shard_size: int, start_shard: int = 0):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.current_shard = start_shard
        self.current_count = 0
        self._fh: Optional[gzip.GzipFile] = None

    def _open_shard(self):
        if self._fh is not None:
            self._fh.close()
        path = self.output_dir / f"shard_{self.current_shard:06d}.xyz.gz"
        log.info(f"Opening shard: {path}")
        self._fh = gzip.open(path, "at", encoding="utf-8")
        self.current_count = 0

    def write(self, xyz_string: str):
        if self._fh is None:
            self._open_shard()
        self._fh.write(xyz_string)
        self.current_count += 1
        if self.current_count >= self.shard_size:
            self.current_shard += 1
            self._open_shard()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# query builder

# known large external databases imported into NOMAD
KNOWN_EXTERNAL_DBS = [
    "AFLOW",
    "The Materials Project",
    "OQMD",
]

# known reference URLs for large databases that are uploaded as regular
KNOWN_LARGE_DB_COMMENTS = [
    "from https://alexandria.icams.rub.de/",
]


def build_query(args: argparse.Namespace) -> dict:
    query: dict[str, Any] = {}

    if args.program_name:
        query["results.method.simulation.program_name"] = args.program_name

    if args.elements:
        query["results.material.elements"] = {"all": args.elements}

    if args.structural_type:
        query["results.material.structural_type"] = args.structural_type

    if args.n_elements_min is not None or args.n_elements_max is not None:
        nel_q: dict[str, Any] = {}
        if args.n_elements_min is not None:
            nel_q["gte"] = args.n_elements_min
        if args.n_elements_max is not None:
            nel_q["lte"] = args.n_elements_max
        query["results.material.n_elements"] = nel_q

    # exclusions
    not_clauses: list[dict] = []

    # exclude-large-dbs convenience flag that excludes all known large
    exclude_ext = []
    exclude_comments = []

    if args.exclude_large_dbs:
        exclude_ext = list(KNOWN_EXTERNAL_DBS)
        exclude_comments = list(KNOWN_LARGE_DB_COMMENTS)

    # exclude-external-db explicit list of external_db values to exclude
    if args.exclude_external_db:
        for db_name in args.exclude_external_db:
            if db_name not in exclude_ext:
                exclude_ext.append(db_name)

    # exclude-comment exclude entries whose comment metadata
    if args.exclude_comment:
        for cmt in args.exclude_comment:
            if cmt not in exclude_comments:
                exclude_comments.append(cmt)

    # build NOT clauses for external_db exclusion
    for db_name in exclude_ext:
        not_clauses.append({"not": {"external_db": db_name}})

    # build NOT clauses for comment exclusion
    for cmt in exclude_comments:
        not_clauses.append({"not": {"comment": cmt}})

    # merge NOT clauses into the query using AND
    if not_clauses:
        if query:
            # wrap existing query + not_clauses in an AND
            query = {"and": [query] + not_clauses}
        elif len(not_clauses) == 1:
            query = not_clauses[0]
        else:
            query = {"and": not_clauses}

    return query


# main download loop

def download_all(
    client: NomadClient,
    query: dict,
    output_dir: Path,
    page_size: int = 100,
    shard_size: int = 50_000,
    resume: bool = False,
    max_entries: Optional[int] = None,
    delay: float = 0.1,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.json"

    ckpt = Checkpoint.load(ckpt_path) if resume else Checkpoint()

    if ckpt.completed:
        log.info("Download already completed (checkpoint says so). Use --no-resume to restart.")
        return

    # count total
    if ckpt.query_total is None:
        log.info("Counting total entries matching query...")
        ckpt.query_total = client.count_entries(query)
        log.info(f"Total entries matching query: {ckpt.query_total:,}")
        ckpt.save(ckpt_path)

    total = ckpt.query_total
    if max_entries is not None:
        total = min(total, max_entries)

    if total == 0:
        log.info("No entries match the query.")
        return

    writer = ShardWriter(output_dir, shard_size, start_shard=ckpt.shard_index)
    pbar = tqdm(
        total=total,
        initial=ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed,
        desc="Downloading structures",
        unit="entry",
    )

    page_after = ckpt.last_page_after_value

    # we request the archive + metadata together using entries/archive/query
    required = ARCHIVE_REQUIRED

    try:
        while True:
            # check if we've hit the user-specified max
            n_processed = ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed
            if max_entries is not None and n_processed >= max_entries:
                log.info(f"Reached max_entries={max_entries}, stopping.")
                break

            # fetch one page of archive data
            try:
                result = client.query_archives_page(
                    query=query,
                    required=required,
                    page_size=page_size,
                    page_after_value=page_after,
                )
            except RuntimeError as e:
                log.error(f"Fatal API error: {e}")
                break

            data = result.get("data", [])
            if not data:
                log.info("No more data returned — download complete.")
                ckpt.completed = True
                break

            # process entries
            for entry in data:
                xyz = extract_structure_extxyz(entry)
                if xyz is not None:
                    writer.write(xyz)
                    ckpt.total_downloaded += 1
                else:
                    ckpt.total_skipped += 1
                pbar.update(1)

            # update pagination cursor
            pagination = result.get("pagination", {})
            next_after = pagination.get("next_page_after_value")

            if next_after is None or next_after == page_after:
                log.info("Pagination exhausted — download complete.")
                ckpt.completed = True
                break

            page_after = next_after
            ckpt.last_page_after_value = page_after
            ckpt.shard_index = writer.current_shard

            # save checkpoint every page
            ckpt.save(ckpt_path)

            # be a good citizen small delay between pages
            time.sleep(delay)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Progress saved to checkpoint.")
    finally:
        writer.close()
        ckpt.shard_index = writer.current_shard
        ckpt.save(ckpt_path)
        pbar.close()

    log.info(
        f"Done. Downloaded: {ckpt.total_downloaded:,} | "
        f"Skipped (no structure): {ckpt.total_skipped:,} | "
        f"Failed: {ckpt.total_failed:,}"
    )


# alternative parallel archive download using entry IDs

def download_parallel(
    client: NomadClient,
    query: dict,
    output_dir: Path,
    page_size: int = 100,
    shard_size: int = 50_000,
    max_workers: int = 4,
    resume: bool = False,
    max_entries: Optional[int] = None,
    delay: float = 0.05,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.json"
    ids_path = output_dir / "entry_ids.json"

    ckpt = Checkpoint.load(ckpt_path) if resume else Checkpoint()

    if ckpt.completed:
        log.info("Download already completed.")
        return

    # phase 1 collect entry IDs
    if ids_path.exists() and resume:
        log.info("Loading cached entry IDs...")
        entry_ids = json.loads(ids_path.read_text())
    else:
        log.info("Phase 1: Collecting entry IDs (metadata only)...")
        total = client.count_entries(query)
        log.info(f"Total entries: {total:,}")

        entry_ids = []
        page_after = None
        pbar = tqdm(total=total, desc="Collecting IDs", unit="entry")

        while True:
            pagination: dict[str, Any] = {
                "page_size": 1000,  # larger pages OK for metadata
                "order_by": "entry_id",
                "order": "asc",
            }
            if page_after:
                pagination["page_after_value"] = page_after

            result = client._post_with_retry("entries/query", {
                "query": query,
                "pagination": pagination,
                "required": {"include": ["entry_id"]},
            })

            data = result.get("data", [])
            if not data:
                break

            for d in data:
                entry_ids.append(d["entry_id"])
            pbar.update(len(data))

            next_after = result["pagination"].get("next_page_after_value")
            if not next_after or next_after == page_after:
                break
            page_after = next_after

            if max_entries and len(entry_ids) >= max_entries:
                entry_ids = entry_ids[:max_entries]
                break

            time.sleep(delay)

        pbar.close()
        ids_path.write_text(json.dumps(entry_ids))
        log.info(f"Collected {len(entry_ids):,} entry IDs.")

    # phase 2 fetch archives in parallel batches
    start_idx = ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed
    remaining_ids = entry_ids[start_idx:]
    log.info(f"Phase 2: Downloading archives for {len(remaining_ids):,} entries...")

    writer = ShardWriter(output_dir, shard_size, start_shard=ckpt.shard_index)
    pbar = tqdm(total=len(entry_ids), initial=start_idx, desc="Downloading", unit="entry")

    batch_size = page_size

    def fetch_batch(batch_ids: list[str]) -> list[dict]:
        result = client._post_with_retry("entries/archive/query", {
            "query": {"entry_id:any": batch_ids},
            "required": ARCHIVE_REQUIRED,
            "pagination": {"page_size": len(batch_ids)},
        })
        return result.get("data", [])

    try:
        # split into batches
        batches = [
            remaining_ids[i:i + batch_size]
            for i in range(0, len(remaining_ids), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # submit batches with controlled concurrency
            future_map = {}
            submitted = 0
            batch_iter = iter(batches)

            # initial submission
            for _ in range(min(max_workers * 2, len(batches))):
                try:
                    batch = next(batch_iter)
                    future = executor.submit(fetch_batch, batch)
                    future_map[future] = batch
                    submitted += 1
                except StopIteration:
                    break

            while future_map:
                for future in as_completed(future_map):
                    batch = future_map.pop(future)
                    try:
                        entries = future.result()
                        for entry in entries:
                            xyz = extract_structure_extxyz(entry)
                            if xyz is not None:
                                writer.write(xyz)
                                ckpt.total_downloaded += 1
                            else:
                                ckpt.total_skipped += 1
                            pbar.update(1)
                    except Exception as e:
                        log.error(f"Batch failed: {e}")
                        ckpt.total_failed += len(batch)
                        pbar.update(len(batch))

                    # submit next batch
                    try:
                        next_batch = next(batch_iter)
                        new_future = executor.submit(fetch_batch, next_batch)
                        future_map[new_future] = next_batch
                    except StopIteration:
                        pass

                    # checkpoint periodically
                    if (ckpt.total_downloaded + ckpt.total_skipped) % (batch_size * 10) == 0:
                        ckpt.shard_index = writer.current_shard
                        ckpt.save(ckpt_path)

                    time.sleep(delay)

        ckpt.completed = True

    except KeyboardInterrupt:
        log.info("\nInterrupted. Progress saved.")
    finally:
        writer.close()
        ckpt.shard_index = writer.current_shard
        ckpt.save(ckpt_path)
        pbar.close()

    log.info(
        f"Done. Downloaded: {ckpt.total_downloaded:,} | "
        f"Skipped: {ckpt.total_skipped:,} | "
        f"Failed: {ckpt.total_failed:,}"
    )


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download atomic structures from the NOMAD database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # output
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("nomad_structures"),
        help="Output directory for shard files and checkpoint (default: nomad_structures)",
    )

    # filters
    parser.add_argument(
        "--program-name", type=str, default=None,
        help="Filter by simulation code, e.g. VASP, 'Quantum Espresso', FHI-aims",
    )
    parser.add_argument(
        "--elements", nargs="+", type=str, default=None,
        help="Filter: entries must contain ALL specified elements, e.g. --elements Ti O",
    )
    parser.add_argument(
        "--structural-type", type=str, default=None,
        choices=["bulk", "2D", "1D", "molecule / cluster", "atom", "unavailable"],
        help="Filter by structural type",
    )
    parser.add_argument(
        "--n-elements-min", type=int, default=None,
        help="Minimum number of distinct elements",
    )
    parser.add_argument(
        "--n-elements-max", type=int, default=None,
        help="Maximum number of distinct elements",
    )

    # exclusion filters
    parser.add_argument(
        "--exclude-large-dbs", action="store_true",
        help=(
            "Exclude entries from known large external databases: "
            "AFLOW, Materials Project, OQMD, and Alexandria. "
            "This is the recommended flag if you want only 'native' NOMAD data."
        ),
    )
    parser.add_argument(
        "--exclude-external-db", nargs="+", type=str, default=None,
        metavar="DB",
        help=(
            "Exclude entries tagged with specific external_db values. "
            'Known values include: "AFLOW", "The Materials Project", "OQMD". '
            "Case-sensitive."
        ),
    )
    parser.add_argument(
        "--exclude-comment", nargs="+", type=str, default=None,
        metavar="TEXT",
        help=(
            "Exclude entries whose `comment` metadata field matches "
            "any of these strings. Use for databases like Alexandria that tag "
            'their entries with e.g. "https://alexandria.icams.rub.de/".'
        ),
    )

    # download params
    parser.add_argument(
        "--page-size", type=int, default=100,
        help="Entries per API request (default: 100). Don't go above ~200.",
    )
    parser.add_argument(
        "--shard-size", type=int, default=50_000,
        help="Structures per output shard file (default: 50000)",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Stop after downloading this many entries (default: unlimited)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Seconds to wait between API requests (default: 0.1)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Parallel workers for --parallel mode (default: 4)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="HTTP request timeout in seconds (default: 120)",
    )

    # modes
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if one exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just count entries matching the query, don't download",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use two-phase parallel download (collect IDs first, then fetch in parallel)",
    )

    # API
    parser.add_argument(
        "--api-url", type=str, default=NOMAD_API_BASE,
        help=f"NOMAD API base URL (default: {NOMAD_API_BASE})",
    )

    args = parser.parse_args()

    client = NomadClient(base_url=args.api_url, timeout=args.timeout)
    query = build_query(args)

    log.info(f"Query: {json.dumps(query, indent=2) if query else '{} (all entries)'}")

    if args.dry_run:
        total = client.count_entries(query)
        log.info(f"Total entries matching query: {total:,}")
        return

    if args.parallel:
        download_parallel(
            client=client,
            query=query,
            output_dir=args.output_dir,
            page_size=args.page_size,
            shard_size=args.shard_size,
            max_workers=args.max_workers,
            resume=args.resume,
            max_entries=args.max_entries,
            delay=args.delay,
        )
    else:
        download_all(
            client=client,
            query=query,
            output_dir=args.output_dir,
            page_size=args.page_size,
            shard_size=args.shard_size,
            resume=args.resume,
            max_entries=args.max_entries,
            delay=args.delay,
        )


if __name__ == "__main__":
    main()
