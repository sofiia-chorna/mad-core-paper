#!/usr/bin/env python3
"""
download_nomad_structures.py
=============================
Bulk-download ALL atomic structures from the NOMAD Repository via the v1 REST API.

Only structural information is retrieved (positions, lattice vectors, species, PBC).
No energies, electronic structure, or other properties are downloaded.

Strategy
--------
We use the `entries/archive/query` endpoint with `page_after_value`-based pagination
(cursor pagination), which has no depth limit — unlike page-number pagination or the
OPTIMADE endpoint, both of which cap out at ~10k results.

We request only `run.system[-1].atoms` (the last/final structure in each entry) plus
minimal metadata for bookkeeping. Results are written incrementally to disk as
compressed extxyz (.xyz.gz) shards to keep memory bounded.

The script is idempotent: it tracks progress in a simple JSON checkpoint file so it
can be resumed after interruptions.

Requirements
------------
    pip install requests ase tqdm

Usage
-----
    # Download everything EXCEPT large external databases (recommended!)
    python download_nomad_structures.py --exclude-large-dbs --output-dir nomad_native

    # Download everything (will take days for 12M+ entries)
    python download_nomad_structures.py --output-dir nomad_structures

    # Filter by DFT code
    python download_nomad_structures.py --program-name VASP --output-dir nomad_vasp

    # Filter by elements
    python download_nomad_structures.py --elements Ti O --output-dir nomad_TiO

    # Filter by structural type
    python download_nomad_structures.py --structural-type bulk --output-dir nomad_bulk

    # Exclude specific external databases by name
    python download_nomad_structures.py --exclude-external-db AFLOW "The Materials Project" OQMD

    # Exclude entries referencing a specific URL in their comment (e.g. Alexandria)
    python download_nomad_structures.py --exclude-comment "https://alexandria.icams.rub.de/"

    # Dry run: just count how many entries match
    python download_nomad_structures.py --dry-run

    # Resume an interrupted download
    python download_nomad_structures.py --output-dir nomad_structures --resume

    # Control parallelism and batch sizes
    python download_nomad_structures.py --output-dir nomad_structures \
        --page-size 100 --shard-size 50000 --max-workers 4
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All valid element symbols (Z=1..118)
_VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

NOMAD_API_BASE = "https://nomad-lab.eu/prod/v1/api/v1"

# What we request from each entry's archive
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

# Metadata fields we always request alongside the archive
METADATA_INCLUDE = [
    "entry_id",
    "upload_id",
    "results.material.chemical_formula_hill",
    "results.material.chemical_formula_reduced",
    "results.material.material_id",
    "results.material.structural_type",
    "results.method.simulation.program_name",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """Tracks download progress for resumability."""
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


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

class NomadClient:
    """Thin wrapper around the NOMAD v1 API with retry logic."""

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
        """POST with exponential backoff on 429/503/5xx."""
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
        """Return total number of entries matching query."""
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
        """Fetch one page of archive data."""
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

    def skip_page_metadata(
        self,
        query: dict,
        page_size: int = 100,
        page_after_value: Optional[str] = None,
    ) -> Optional[str]:
        """
        Lightweight metadata-only request to advance the pagination cursor
        past a problematic page. Returns the next_page_after_value, or None
        if pagination is exhausted.
        """
        pagination: dict[str, Any] = {
            "page_size": page_size,
            "order_by": "entry_id",
            "order": "asc",
        }
        if page_after_value is not None:
            pagination["page_after_value"] = page_after_value

        payload = {
            "query": query,
            "pagination": pagination,
            "required": {"include": ["entry_id"]},
        }
        result = self._post_with_retry("entries/query", payload)
        return result.get("pagination", {}).get("next_page_after_value")

    def collect_dataset_dois(self, query: dict, max_dois: int = 10000) -> list[dict]:
        """
        Collect all unique dataset DOIs for entries matching the query
        using a single aggregation request.

        Returns a list of {"doi": str, "count": int} dicts.
        """
        payload = {
            "query": query,
            "pagination": {"page_size": 0},
            "required": {"include": ["entry_id"]},
            "aggregations": {
                "dois": {
                    "terms": {
                        "quantity": "datasets.doi",
                        "size": max_dois,
                    }
                }
            },
        }
        result = self._post_with_retry("entries/query", payload)
        agg = result.get("aggregations", {}).get("dois", {})
        terms = agg.get("terms", {})
        data = terms.get("data", [])
        return [
            {"doi": item.get("value", ""), "count": item.get("count", 0)}
            for item in data
            if item.get("value")
        ]


# ---------------------------------------------------------------------------
# Structure extraction → extxyz lines
# ---------------------------------------------------------------------------

def _clean_element_label(label: str) -> Optional[str]:
    """
    Extract a valid element symbol from a NOMAD site label.

    Handles: "Sr1" → "Sr", "Fe_sv" → "Fe", "FE" → "Fe", "o" → "O",
             "Ca_pv" → "Ca", "r" → None (not a valid element).

    Returns the cleaned symbol or None if no valid element can be extracted.
    """
    # Extract the alphabetic prefix (before digits, underscores, +, etc.)
    m = re.match(r'([A-Za-z]{1,2})', label)
    if not m:
        return None

    prefix = m.group(1)

    # Try as 2-letter element first (title-cased): "FE" → "Fe", "sr" → "Sr"
    if len(prefix) >= 2:
        candidate2 = prefix[:2].capitalize()
        if candidate2 in _VALID_ELEMENTS:
            return candidate2

    # Fall back to 1-letter element: "O2" → "O", "H" → "H"
    candidate1 = prefix[0].upper()
    if candidate1 in _VALID_ELEMENTS:
        return candidate1

    return None


def compute_min_distance(
    positions: np.ndarray,
    lattice: Optional[np.ndarray] = None,
    pbc: Optional[list[bool]] = None,
) -> float:
    """
    Compute the minimum interatomic distance in a structure.

    Parameters
    ----------
    positions : (N, 3) array in Angstrom
    lattice : (3, 3) array in Angstrom, or None for non-periodic
    pbc : list of 3 bools, or None (assumed all True if lattice given)

    Returns
    -------
    Minimum pairwise distance in Angstrom. Returns inf if < 2 atoms.
    """
    n = len(positions)
    if n < 2:
        return float("inf")

    if lattice is not None:
        # Minimum image convention for periodic systems
        if pbc is None:
            pbc = [True, True, True]

        inv_lat = np.linalg.inv(lattice)
        frac = positions @ inv_lat

        # Pairwise fractional differences: (N, N, 3)
        delta_frac = frac[:, None, :] - frac[None, :, :]

        # Apply minimum image convention only along periodic directions
        for dim in range(3):
            if pbc[dim]:
                delta_frac[:, :, dim] -= np.round(delta_frac[:, :, dim])

        # Convert back to Cartesian
        delta_cart = delta_frac @ lattice
    else:
        delta_cart = positions[:, None, :] - positions[None, :, :]

    # Squared distances
    dist_sq = np.sum(delta_cart ** 2, axis=-1)

    # Mask diagonal (self-distances)
    np.fill_diagonal(dist_sq, float("inf"))

    return float(np.sqrt(dist_sq.min()))


def extract_structure_extxyz(
    entry: dict,
    min_dist: Optional[float] = None,
) -> tuple[Optional[str], list[str]]:
    """
    Convert a single NOMAD archive entry dict into an extxyz-format string.

    Returns (xyz_string, dataset_dois) where xyz_string is None if the entry
    has no parseable structure or fails validation, and dataset_dois is a list
    of DOI strings from the entry's datasets.

    Parameters
    ----------
    entry : dict
        Single entry from the NOMAD archive query response.
    min_dist : float, optional
        If set, skip structures where any two atoms are closer than this
        distance (in Angstrom). Uses minimum image convention for periodic
        systems.
    """
    try:
        archive = entry.get("archive", {})
        runs = archive.get("run", [])
        if not runs:
            return None, []

        # Get the last run's last system (we requested system[-1])
        systems = runs[-1].get("system", []) if runs else []
        if not systems:
            return None, []

        system = systems[-1] if isinstance(systems, list) else systems
        atoms = system.get("atoms", {})
        if not atoms:
            return None, []

        # Positions: NOMAD stores in SI (metres), convert to Angstrom
        positions = atoms.get("positions")
        if positions is None:
            return None, []
        positions = np.array(positions, dtype=np.float64) * 1e10  # m → Å

        n_atoms = len(positions)
        if n_atoms == 0:
            return None, []

        # Labels (element symbols)
        # NOMAD labels can be site labels ("Sr1", "O2"), POTCAR names
        # ("Ca_sv", "Fe_pv"), or even lowercase ("r", "o"). We extract
        # the element symbol and validate it.
        labels = atoms.get("labels")
        if labels is None:
            # Fall back to species
            species = atoms.get("species")
            if species is not None:
                labels = [sp.get("name", "X") if isinstance(sp, dict) else str(sp)
                          for sp in species]
            else:
                return None, []

        # Clean labels: "Sr1" → "Sr", "Ca_sv" → "Ca", "FE" → "Fe", "o" → "O"
        cleaned_labels = []
        for lbl in labels:
            sym = _clean_element_label(lbl)
            if sym is None:
                log.debug(f"Invalid element label '{lbl}', skipping entry")
                return None, []
            cleaned_labels.append(sym)

        labels = cleaned_labels

        if len(labels) != n_atoms:
            return None, []

        # Lattice vectors: NOMAD stores in SI (metres) → Angstrom
        lattice = atoms.get("lattice_vectors")
        pbc = atoms.get("periodic")

        # Convert lattice to Angstrom for distance check and extxyz output
        lat_angstrom = None
        if lattice is not None:
            lat_angstrom = np.array(lattice, dtype=np.float64) * 1e10

        # Minimum distance check: skip structures with unreasonably close atoms
        if min_dist is not None and n_atoms >= 2:
            pbc_list = pbc if pbc is not None else (
                [True, True, True] if lat_angstrom is not None else None
            )
            d_min = compute_min_distance(positions, lat_angstrom, pbc_list)
            if d_min < min_dist:
                log.debug(
                    f"Min distance {d_min:.3f} Å < {min_dist} Å, "
                    f"skipping entry {entry.get('entry_id', '?')}"
                )
                return None, []

        # Build the comment/properties line
        entry_id = entry.get("entry_id", "unknown")

        # Get metadata from the entry-level fields
        formula = ""
        material_id = ""
        program = ""
        structural_type = ""
        dataset_dois = []
        if "results" in entry:
            mat = entry["results"].get("material", {})
            formula = mat.get("chemical_formula_hill", "") or ""
            material_id = mat.get("material_id", "") or ""
            structural_type = mat.get("structural_type", "") or ""
            method = entry["results"].get("method", {})
            sim = method.get("simulation", {})
            program = sim.get("program_name", "") or ""

        # Extract dataset DOIs
        datasets = entry.get("datasets", [])
        if datasets:
            for ds in datasets:
                doi = ds.get("doi") if isinstance(ds, dict) else None
                if doi:
                    dataset_dois.append(doi)

        lines = io.StringIO()

        # Line 1: atom count
        lines.write(f"{n_atoms}\n")

        # Line 2: comment line with properties
        comment_parts = []

        if lat_angstrom is not None:
            lat_str = " ".join(f"{x:.10f}" for x in lat_angstrom.flatten())
            comment_parts.append(f'Lattice="{lat_str}"')

        if pbc is not None:
            pbc_str = " ".join("T" if p else "F" for p in pbc)
            comment_parts.append(f'pbc="{pbc_str}"')
        elif lat_angstrom is not None:
            # If lattice is given but PBC is not, assume periodic
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
        if dataset_dois:
            # Join multiple DOIs with semicolons (no spaces, safe for extxyz)
            doi_str = ";".join(dataset_dois)
            comment_parts.append(f'doi="{doi_str}"')

        lines.write(" ".join(comment_parts) + "\n")

        # Atom lines
        for i in range(n_atoms):
            symbol = labels[i]
            x, y, z = positions[i]
            lines.write(f"{symbol:>4s} {x:20.10f} {y:20.10f} {z:20.10f}\n")

        return lines.getvalue(), dataset_dois

    except Exception as e:
        log.debug(f"Failed to extract structure from entry: {e}")
        return None, []


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes structures to compressed extxyz shard files."""

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


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

# Known large external databases imported into NOMAD.
# These are the values of the `external_db` metadata field.
KNOWN_EXTERNAL_DBS = [
    "AFLOW",
    "The Materials Project",
    "OQMD",
]

# Known reference URLs for large databases that are uploaded as regular
# NOMAD entries (not tagged via external_db). Entries whose `comment`
# metadata field contains any of these strings will be excluded.
KNOWN_LARGE_DB_COMMENTS = [
    "https://alexandria.icams.rub.de/",
]


def build_query(args: argparse.Namespace) -> dict:
    """Build NOMAD query dict from CLI arguments."""
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

    # ---- Exclusions ----
    # Collect all NOT clauses; we'll AND them together at the end.
    not_clauses: list[dict] = []

    # --exclude-large-dbs: convenience flag that excludes all known large
    # external databases (AFLOW, Materials Project, OQMD) AND known large
    # database uploads (Alexandria via its comment field).
    exclude_ext = []
    exclude_comments = []

    if args.exclude_large_dbs:
        exclude_ext = list(KNOWN_EXTERNAL_DBS)
        exclude_comments = list(KNOWN_LARGE_DB_COMMENTS)

    # --exclude-external-db: explicit list of external_db values to exclude
    if args.exclude_external_db:
        for db_name in args.exclude_external_db:
            if db_name not in exclude_ext:
                exclude_ext.append(db_name)

    # --exclude-comment: exclude entries whose `comment` metadata
    # matches any of these strings (exact match).
    if args.exclude_comment:
        for cmt in args.exclude_comment:
            if cmt not in exclude_comments:
                exclude_comments.append(cmt)

    # Build NOT clauses for external_db exclusion.
    # The external_db field is a simple string field on entries.
    # We exclude each value individually with NOT clauses.
    for db_name in exclude_ext:
        not_clauses.append({"not": {"external_db": db_name}})

    # Build NOT clauses for comment exclusion.
    # The `comment` field is a string on each entry's metadata.
    for cmt in exclude_comments:
        not_clauses.append({"not": {"comment": cmt}})

    # Merge NOT clauses into the query using AND
    if not_clauses:
        if query:
            # Wrap existing query + not_clauses in an AND
            query = {"and": [query] + not_clauses}
        elif len(not_clauses) == 1:
            query = not_clauses[0]
        else:
            query = {"and": not_clauses}

    return query


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

def download_all(
    client: NomadClient,
    query: dict,
    output_dir: Path,
    page_size: int = 100,
    shard_size: int = 50_000,
    resume: bool = False,
    max_entries: Optional[int] = None,
    delay: float = 0.1,
    min_dist: Optional[float] = None,
):
    """
    Main download loop with checkpoint-based resumability.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.json"

    ckpt = Checkpoint.load(ckpt_path) if resume else Checkpoint()

    if ckpt.completed:
        log.info("Download already completed (checkpoint says so). Use --no-resume to restart.")
        return

    # Count total
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
    doi_path = output_dir / "dataset_dois.txt"
    all_dois: set[str] = set()
    # Load previously collected DOIs if resuming
    if doi_path.exists():
        all_dois = set(doi_path.read_text().strip().splitlines())

    pbar = tqdm(
        total=total,
        initial=ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed,
        desc="Downloading structures",
        unit="entry",
    )

    page_after = ckpt.last_page_after_value

    # We request the archive + metadata together using entries/archive/query.
    # This endpoint returns archive data alongside entry metadata.
    required = ARCHIVE_REQUIRED

    try:
        while True:
            # Check if we've hit the user-specified max
            n_processed = ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed
            if max_entries is not None and n_processed >= max_entries:
                log.info(f"Reached max_entries={max_entries}, stopping.")
                break

            # Fetch one page of archive data
            try:
                result = client.query_archives_page(
                    query=query,
                    required=required,
                    page_size=page_size,
                    page_after_value=page_after,
                )
            except RuntimeError:
                # The archive endpoint failed for this page (e.g. 500 on a
                # bad entry). Use the lightweight metadata endpoint to skip
                # past this page and continue.
                log.warning(
                    f"Archive request failed at cursor={page_after}. "
                    f"Skipping {page_size} entries via metadata fallback..."
                )
                try:
                    next_after = client.skip_page_metadata(
                        query=query,
                        page_size=page_size,
                        page_after_value=page_after,
                    )
                except RuntimeError:
                    log.error("Metadata fallback also failed. Stopping.")
                    break

                if next_after is None or next_after == page_after:
                    log.error("Could not advance cursor. Stopping.")
                    break

                ckpt.total_failed += page_size
                pbar.update(page_size)
                page_after = next_after
                ckpt.last_page_after_value = page_after
                ckpt.save(ckpt_path)
                time.sleep(delay)
                continue

            data = result.get("data", [])
            if not data:
                log.info("No more data returned — download complete.")
                ckpt.completed = True
                break

            # Process entries
            for entry in data:
                xyz, dois = extract_structure_extxyz(entry, min_dist=min_dist)
                if xyz is not None:
                    writer.write(xyz)
                    ckpt.total_downloaded += 1
                else:
                    ckpt.total_skipped += 1
                all_dois.update(dois)
                pbar.update(1)

            # Update pagination cursor
            pagination = result.get("pagination", {})
            next_after = pagination.get("next_page_after_value")

            if next_after is None or next_after == page_after:
                log.info("Pagination exhausted — download complete.")
                ckpt.completed = True
                break

            page_after = next_after
            ckpt.last_page_after_value = page_after
            ckpt.shard_index = writer.current_shard

            # Save checkpoint every page
            ckpt.save(ckpt_path)

            # Be a good citizen: small delay between pages
            time.sleep(delay)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Progress saved to checkpoint.")
    finally:
        writer.close()
        ckpt.shard_index = writer.current_shard
        ckpt.save(ckpt_path)
        pbar.close()
        # Save any DOIs collected entry-by-entry (partial, best-effort)
        if all_dois:
            doi_path.write_text("\n".join(sorted(all_dois)) + "\n")

    # After download completes (or on resume with --collect-dois),
    # do an authoritative aggregation query to get ALL dataset DOIs
    if ckpt.completed:
        _collect_and_save_dois(client, query, doi_path)

    log.info(
        f"Done. Downloaded: {ckpt.total_downloaded:,} | "
        f"Skipped (no structure): {ckpt.total_skipped:,} | "
        f"Failed: {ckpt.total_failed:,}"
    )


# ---------------------------------------------------------------------------
# Alternative: parallel archive download using entry IDs
# ---------------------------------------------------------------------------

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
    min_dist: Optional[float] = None,
):
    """
    Two-phase parallel download:
      1. Paginate through entries/query to collect all entry_ids (fast, metadata only)
      2. Fetch archives in parallel batches using entries/archive/query with
         entry_id filters

    This is faster for very large downloads but uses more memory for the ID list.
    For 12M+ entries, phase 1 alone will take a while. Consider the sequential
    approach (download_all) for simplicity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint.json"
    ids_path = output_dir / "entry_ids.json"

    ckpt = Checkpoint.load(ckpt_path) if resume else Checkpoint()

    if ckpt.completed:
        log.info("Download already completed.")
        return

    # Phase 1: collect entry IDs
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

    # Phase 2: fetch archives in parallel batches
    start_idx = ckpt.total_downloaded + ckpt.total_skipped + ckpt.total_failed
    remaining_ids = entry_ids[start_idx:]
    log.info(f"Phase 2: Downloading archives for {len(remaining_ids):,} entries...")

    writer = ShardWriter(output_dir, shard_size, start_shard=ckpt.shard_index)
    doi_path = output_dir / "dataset_dois.txt"
    all_dois: set[str] = set()
    if doi_path.exists():
        all_dois = set(doi_path.read_text().strip().splitlines())

    pbar = tqdm(total=len(entry_ids), initial=start_idx, desc="Downloading", unit="entry")

    batch_size = page_size

    def fetch_batch(batch_ids: list[str]) -> list[dict]:
        """Fetch archive data for a batch of entry IDs."""
        result = client._post_with_retry("entries/archive/query", {
            "query": {"entry_id:any": batch_ids},
            "required": ARCHIVE_REQUIRED,
            "pagination": {"page_size": len(batch_ids)},
        })
        return result.get("data", [])

    try:
        # Split into batches
        batches = [
            remaining_ids[i:i + batch_size]
            for i in range(0, len(remaining_ids), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches with controlled concurrency
            future_map = {}
            submitted = 0
            batch_iter = iter(batches)

            # Initial submission
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
                            xyz, dois = extract_structure_extxyz(entry, min_dist=min_dist)
                            if xyz is not None:
                                writer.write(xyz)
                                ckpt.total_downloaded += 1
                            else:
                                ckpt.total_skipped += 1
                            all_dois.update(dois)
                            pbar.update(1)
                    except Exception as e:
                        log.error(f"Batch failed: {e}")
                        ckpt.total_failed += len(batch)
                        pbar.update(len(batch))

                    # Submit next batch
                    try:
                        next_batch = next(batch_iter)
                        new_future = executor.submit(fetch_batch, next_batch)
                        future_map[new_future] = next_batch
                    except StopIteration:
                        pass

                    # Checkpoint periodically
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
        if all_dois:
            doi_path.write_text("\n".join(sorted(all_dois)) + "\n")

    if ckpt.completed:
        _collect_and_save_dois(client, query, doi_path)

    log.info(
        f"Done. Downloaded: {ckpt.total_downloaded:,} | "
        f"Skipped: {ckpt.total_skipped:,} | "
        f"Failed: {ckpt.total_failed:,}"
    )


def _collect_and_save_dois(client: NomadClient, query: dict, doi_path: Path):
    """Collect all dataset DOIs via aggregation and save to file."""
    log.info("Collecting all dataset DOIs via aggregation query...")
    try:
        doi_records = client.collect_dataset_dois(query)
        if doi_records:
            lines = []
            for rec in sorted(doi_records, key=lambda r: -r["count"]):
                lines.append(f"{rec['doi']}\t{rec['count']} entries")
            doi_path.write_text("\n".join(lines) + "\n")
            log.info(
                f"Collected {len(doi_records):,} unique dataset DOIs → {doi_path}"
            )
        else:
            log.info("No dataset DOIs found for this query.")
    except Exception as e:
        log.warning(f"Failed to collect DOIs via aggregation: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bulk-download atomic structures from the NOMAD database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("nomad_structures"),
        help="Output directory for shard files and checkpoint (default: nomad_structures)",
    )

    # Filters
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

    # Exclusion filters
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

    # Download params
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

    # Modes
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
    parser.add_argument(
        "--collect-dois", action="store_true",
        help=(
            "Collect all dataset DOIs for entries matching the query "
            "(via aggregation) and save to dataset_dois.txt. "
            "Can be run standalone without downloading structures."
        ),
    )

    # Validation
    parser.add_argument(
        "--min-distance", type=float, default=None,
        metavar="ANGSTROM",
        help=(
            "Skip structures where any two atoms are closer than this "
            "distance (in Angstrom). Recommended: 0.5. Uses minimum image "
            "convention for periodic systems."
        ),
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

    if args.collect_dois:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        doi_path = args.output_dir / "dataset_dois.txt"
        _collect_and_save_dois(client, query, doi_path)
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
            min_dist=args.min_distance,
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
            min_dist=args.min_distance,
        )


if __name__ == "__main__":
    main()
