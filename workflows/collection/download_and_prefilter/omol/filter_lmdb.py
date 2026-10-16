from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.io.jsonio import decode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyykko_filter import build_min_distance_matrix, find_close_contacts


# row to Atoms (robust to ASE's __ndarray__ JSON encoding)
def _maybe_decode(x):
    if isinstance(x, dict) and "__ndarray__" in x:
        return decode(x)
    return x


def _row_to_atoms(row) -> Atoms:
    numbers = _maybe_decode(row.numbers)
    positions = _maybe_decode(row.positions)
    cell = _maybe_decode(getattr(row, "cell", None))
    pbc = _maybe_decode(getattr(row, "pbc", None))

    if cell is not None:
        cell = np.asarray(cell)
    if pbc is None:
        pbc = False

    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


# filter driver
def filter_aselmdb(
    in_path: str,
    out_path: str,
    rejected_path: str | None = None,
    scale: float = 0.5,
    backend: str = "auto",
    verbose: bool = False,
    log_every: int = 10000,
) -> dict:
    # idempotent rerun semantics of the bash driver, which deletes
    for p in (out_path, rejected_path):
        if p is not None and os.path.exists(p):
            os.remove(p)
            # ASE LMDB backends usually create a path -lock sidecar too
            lock = p + "-lock"
            if os.path.exists(lock):
                os.remove(lock)

    db_in  = connect(in_path)
    db_out = connect(out_path)
    db_rej = connect(rejected_path) if rejected_path else None

    mdm = build_min_distance_matrix(scale=scale)

    passed = rejected = skipped = total = 0
    pair_counter:  Counter = Counter()
    error_counter: Counter = Counter()
    t0 = time.time()

    for row in db_in.select():
        total += 1

        # build the structural Atoms used for the filter
        try:
            atoms = _row_to_atoms(row)
        except Exception as e:
            skipped += 1
            error_counter[f"parse:{type(e).__name__}"] += 1
            if verbose:
                print(f"  SKIP id={row.id}: {type(e).__name__}: {e}",
                      flush=True)
            continue

        # run the distance filter, catching per-row errors
        try:
            i, j, d, S = find_close_contacts(atoms, mdm, backend=backend)
        except Exception as e:
            skipped += 1
            msg = str(e)
            if "exceeds matrix size" in msg or "max_z" in msg:
                error_counter["filter:superheavy_Z"] += 1
            elif "not invertible" in msg or "singular" in msg.lower():
                error_counter["filter:singular_cell"] += 1
            else:
                error_counter[f"filter:{type(e).__name__}"] += 1
            if verbose:
                print(f"  SKIP id={row.id}: filter error: {e}", flush=True)
            continue

        # singlePointCalculator (energy/forces/stress) when present
        kvp  = dict(row.key_value_pairs) if row.key_value_pairs else {}
        data = dict(row.data) if row.data else None
        try:
            full_atoms = row.toatoms()
        except Exception:
            # fallback write the structural atoms only (loses calc results)
            full_atoms = atoms

        if i.size == 0:
            passed += 1
            db_out.write(full_atoms, key_value_pairs=kvp, data=data)
        else:
            rejected += 1
            kk = int(np.argmin(d))
            syms = atoms.get_chemical_symbols()
            pair_counter[tuple(sorted([syms[i[kk]], syms[j[kk]]]))] += 1

            if verbose:
                offset = (int(S[kk][0]), int(S[kk][1]), int(S[kk][2]))
                tag = ("self-image" if i[kk] == j[kk]
                       else "cross-image" if offset != (0, 0, 0)
                       else "in-cell")
                print(f"  reject id={row.id}: "
                      f"{syms[i[kk]]}({int(i[kk])})-{syms[j[kk]]}({int(j[kk])}) "
                      f"d={float(d[kk]):.3f} cutoff={float(mdm[atoms.numbers[i[kk]], atoms.numbers[j[kk]]]):.3f} "
                      f"S={offset} [{tag}]",
                      flush=True)

            if db_rej is not None:
                db_rej.write(full_atoms, key_value_pairs=kvp, data=data)

        if log_every and total % log_every == 0:
            rate = total / max(time.time() - t0, 1e-9)
            top = pair_counter.most_common(3)
            top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top)
            print(f"[{total:>10d}] {rate:7.0f} struct/s | "
                  f"passed={passed:,} rejected={rejected:,} "
                  f"skipped={skipped:,}"
                  + (f" | top: {top_str}" if top else ""),
                  flush=True)

    # final summary
    if rejected > 0:
        print(f"\nRejection causes by element pair "
              f"(unique pairs: {len(pair_counter)}):")
        for (a, b), c in pair_counter.most_common():
            pct = 100.0 * c / rejected
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")
    if skipped > 0:
        print(f"\nSkipped {skipped:,} structures (per-row errors)")
        if error_counter:
            print("  breakdown:")
            for cat, c in error_counter.most_common():
                pct = 100.0 * c / skipped
                print(f"    {cat:<28s} {c:>8,d}  ({pct:5.1f}%)")

    return {
        "passed":   passed,
        "rejected": rejected,
        "skipped":  skipped,
        "total":    total,
        "rejection_pairs": pair_counter,
        "error_counter":   error_counter,
    }


# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Filter an ASE LMDB database (.aselmdb) using Pyykko "
                    "covalent-radius distance cutoffs. Preserves all row "
                    "metadata: calculator results (energy/forces/stress), "
                    "key-value pairs, and custom data fields."
    )
    p.add_argument("input", help="Input .aselmdb file.")
    p.add_argument("-o", "--output", required=True,
                   help="Output .aselmdb for passing structures.")
    p.add_argument("--rejected", default=None,
                   help="Optional output .aselmdb for rejected structures.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor (default 0.5).")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per rejected/skipped structure.")
    args = p.parse_args()

    stats = filter_aselmdb(
        args.input, args.output,
        rejected_path=args.rejected,
        scale=args.scale,
        backend=args.backend,
        verbose=args.verbose,
    )
    print(f"Done: passed={stats['passed']:,} rejected={stats['rejected']:,} "
          f"skipped={stats['skipped']:,} total={stats['total']:,}")
