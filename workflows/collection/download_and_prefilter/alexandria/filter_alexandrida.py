"""
Apply the Pyykko covalent-radius distance filter to Alexandria-format JSON
datasets (e.g. https://alexandria.icams.rub.de/data/pbe/benchmarks/...).

Alexandria distributes datasets as bzip2-compressed JSON. Each file is
either:

    {"entries": [<entry>, <entry>, ...]}      <- common case
or
    [<entry>, <entry>, ...]                   <- bare list

Each entry contains a pymatgen-serialized structure under the "structure"
key with:

    structure["lattice"]["matrix"]   -> 3x3 list (cell in A)
    structure["sites"]               -> list of site dicts, each with:
        species[0]["element"]            -> element symbol (str)
        xyz                              -> Cartesian coords (3-vector, A) [preferred]
        abc                              -> fractional coords (3-vector)   [fallback]

PBC is implicit in pymatgen structures (always True in 3D); the filter
treats every Alexandria entry as fully periodic.

The output is also Alexandria-format JSON: a bzip2-compressed file with
{"entries": [...]} containing only the structures that passed. A
sidecar JSON for rejections gets the same layout plus extra
"_rejection" diagnostics on each entry.

Usage:
    python alexandria_filter.py P000.json.bz2 -o P000_passed.json.bz2 \\
        --rejected P000_rejected.json.bz2 --scale 0.5
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as ASE_Z

# Reuse the core filter.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyykko_filter import build_min_distance_matrix, find_close_contacts


# -----------------------------------------------------------------------------
# I/O helpers: transparent bz2 / gzip / plain JSON
# -----------------------------------------------------------------------------
def _open_read(path: str) -> io.BufferedReader:
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rb")
    if p.endswith(".gz"):
        return gzip.open(p, "rb")
    return open(p, "rb")


def _open_write(path: str) -> io.BufferedWriter:
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "wb")
    if p.endswith(".gz"):
        return gzip.open(p, "wb")
    return open(p, "wb")


def load_alexandria(path: str) -> tuple[list[dict], dict]:
    """
    Load an Alexandria-style file. Returns (entries, top_level_meta) where
    top_level_meta is anything other than 'entries' that we want to preserve
    on output (e.g. metadata fields). For bare-list inputs, meta is {}.
    """
    with _open_read(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict) and "entries" in data:
        meta = {k: v for k, v in data.items() if k != "entries"}
        return data["entries"], meta
    raise ValueError(
        f"unrecognized JSON top-level for {path}: expected list or dict with "
        f"'entries' key, got {type(data).__name__}"
    )


def save_alexandria(path: str, entries: list[dict], meta: dict | None = None) -> None:
    """Write entries in Alexandria layout. Preserves top-level meta if given."""
    payload: dict = dict(meta) if meta else {}
    payload["entries"] = entries
    with _open_write(path) as f:
        f.write(json.dumps(payload).encode("utf-8"))


# -----------------------------------------------------------------------------
# Structure extraction
# -----------------------------------------------------------------------------
def _entry_to_atoms(entry: dict) -> Atoms:
    """
    Convert one Alexandria entry to an ASE Atoms. Reads the pymatgen-style
    structure dict; supports both 'xyz' (Cartesian) and 'abc' (fractional)
    site coordinates.
    """
    s = entry.get("structure")
    if s is None:
        raise KeyError("entry has no 'structure' field")

    lattice = s.get("lattice", {})
    matrix = lattice.get("matrix")
    if matrix is None:
        raise KeyError("structure has no 'lattice.matrix'")
    cell = np.asarray(matrix, dtype=float)

    sites = s.get("sites", [])
    if not sites:
        raise ValueError("structure has zero sites")

    Z = np.empty(len(sites), dtype=int)
    pos = np.empty((len(sites), 3), dtype=float)
    have_cartesian = "xyz" in sites[0]

    for k, site in enumerate(sites):
        species = site.get("species")
        if not species:
            raise KeyError(f"site {k} has no species")
        # Take the dominant (first-listed) element. Disordered sites with
        # mixed occupancies use the same logic as pymatgen.Structure.species.
        elem = species[0].get("element")
        if elem is None:
            raise KeyError(f"site {k} species[0] has no 'element'")
        Z[k] = ASE_Z[elem]
        if have_cartesian:
            pos[k] = site["xyz"]
        else:
            frac = np.asarray(site["abc"], dtype=float)
            pos[k] = frac @ cell

    return Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)


# -----------------------------------------------------------------------------
# Filter driver
# -----------------------------------------------------------------------------
def filter_alexandria(
    in_path: str,
    out_path: str,
    rejected_path: str | None = None,
    scale: float = 0.5,
    backend: str = "auto",
    verbose: bool = False,
    log_every: int = 10000,
) -> dict:
    """
    Filter an Alexandria-format JSON file.

    Returns dict with: passed, rejected, total, rejection_pairs (Counter),
    skipped (entries that couldn't be parsed into Atoms).
    """
    print(f"Loading {in_path} ...", flush=True)
    t0 = time.time()
    entries, meta = load_alexandria(in_path)
    print(f"  {len(entries):,} entries loaded in {time.time()-t0:.1f}s", flush=True)

    mdm = build_min_distance_matrix(scale=scale)
    pair_counter: Counter = Counter()
    passed_entries: list[dict] = []
    rejected_entries: list[dict] = []
    skipped = 0

    t0 = time.time()
    for n, entry in enumerate(entries, start=1):
        try:
            atoms = _entry_to_atoms(entry)
        except Exception as e:
            skipped += 1
            if verbose:
                eid = entry.get("entry_id", entry.get("id", f"#{n}"))
                print(f"  SKIP {eid}: {e}", flush=True)
            continue

        i, j, d, S = find_close_contacts(atoms, mdm, backend=backend)
        if i.size == 0:
            passed_entries.append(entry)
        else:
            k = int(np.argmin(d))
            syms = atoms.get_chemical_symbols()
            Zi = int(atoms.numbers[i[k]])
            Zj = int(atoms.numbers[j[k]])
            pair_key = tuple(sorted([syms[i[k]], syms[j[k]]]))
            pair_counter[pair_key] += 1

            if verbose:
                offset = (int(S[k][0]), int(S[k][1]), int(S[k][2]))
                tag = ("self-image" if i[k] == j[k]
                       else "cross-image" if offset != (0, 0, 0)
                       else "in-cell")
                eid = entry.get("entry_id", entry.get("id", f"#{n}"))
                print(f"  reject {eid}: {syms[i[k]]}({int(i[k])})-"
                      f"{syms[j[k]]}({int(j[k])}) d={float(d[k]):.3f} "
                      f"cutoff={float(mdm[Zi,Zj]):.3f} S={offset} [{tag}]",
                      flush=True)

            if rejected_path is not None:
                # Add diagnostic info under a single "_rejection" key so it's
                # easy to strip back out and doesn't collide with Alexandria's
                # own field names.
                rec = dict(entry)
                rec["_rejection"] = {
                    "bad_i":            int(i[k]),
                    "bad_j":            int(j[k]),
                    "bad_elements":     f"{syms[i[k]]}-{syms[j[k]]}",
                    "bad_distance":     float(d[k]),
                    "bad_cutoff":       float(mdm[Zi, Zj]),
                    "bad_offset":       [int(s) for s in S[k]],
                    "bad_is_self_image": bool(i[k] == j[k]),
                    "n_bad_pairs":      int(i.size),
                }
                rejected_entries.append(rec)

        if log_every and n % log_every == 0:
            rate = n / max(time.time() - t0, 1e-9)
            top = pair_counter.most_common(3)
            top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top)
            print(f"  [{n:>10d}] {rate:7.0f} entry/s | "
                  f"passed={len(passed_entries):,} "
                  f"rejected={len(rejected_entries) if rejected_path else sum(pair_counter.values())}"
                  + (f" | top: {top_str}" if top else ""),
                  flush=True)

    # Write outputs.
    print(f"Writing {len(passed_entries):,} passed entries to {out_path} ...",
          flush=True)
    save_alexandria(out_path, passed_entries, meta=meta)

    n_rej = sum(pair_counter.values())
    if rejected_path is not None and rejected_entries:
        print(f"Writing {len(rejected_entries):,} rejected entries to "
              f"{rejected_path} ...", flush=True)
        save_alexandria(rejected_path, rejected_entries, meta=meta)
    elif rejected_path is not None:
        print(f"No rejections; skipping {rejected_path}", flush=True)

    if n_rej > 0:
        print(f"\nRejection causes by element pair "
              f"(unique pairs: {len(pair_counter)}):")
        for (a, b), c in pair_counter.most_common():
            pct = 100.0 * c / n_rej
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")
    if skipped > 0:
        print(f"\nSkipped {skipped} entries (could not parse structure)")

    return {
        "passed": len(passed_entries),
        "rejected": n_rej,
        "skipped": skipped,
        "total": len(entries),
        "rejection_pairs": pair_counter,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Filter an Alexandria-format JSON dataset using Pyykko "
                    "covalent-radius distance cutoffs.",
        epilog="Input/output paths ending in .bz2 or .gz are transparently "
               "(de)compressed.",
    )
    p.add_argument("input", help="Alexandria JSON file (.json / .json.bz2 / .json.gz).")
    p.add_argument("-o", "--output", required=True,
                   help="Output JSON path for passing entries.")
    p.add_argument("--rejected", default=None,
                   help="Optional output JSON path for rejected entries "
                        "(with _rejection diagnostic field added).")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor (default 0.5).")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per rejected/skipped entry.")
    args = p.parse_args()

    stats = filter_alexandria(
        args.input, args.output,
        rejected_path=args.rejected,
        scale=args.scale,
        backend=args.backend,
        verbose=args.verbose,
    )
    print(f"Done: passed={stats['passed']} rejected={stats['rejected']} "
          f"skipped={stats['skipped']} total={stats['total']}")
