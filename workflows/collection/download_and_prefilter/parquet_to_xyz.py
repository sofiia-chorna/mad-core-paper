"""
Convert a ColabFit-style parquet dataset to extxyz.

Auto-detects three layouts:
  1. Configurations only (no energy / forces / stress).
  2. Configurations with inline property columns (single combined table).
  3. Configurations linked to a separate property_objects table via id.

Usage:
    # Configurations only or with inline props
    python parquet_to_xyz.py colabfit_data/foo/co/ -o foo.xyz

    # Pull energy / forces / stress from a separate po/ table
    python parquet_to_xyz.py colabfit_data/foo/co/ -o foo.xyz \\
        --properties colabfit_data/foo/po/

Inputs follow the same path conventions as the filter: file, directory,
glob pattern, or list. Streams in batches; suitable for large datasets.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as pa_ds
from ase import Atoms
from ase.io import write as ase_write

# Reuse the parquet helpers from the filter module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyykko_filter_parquet import (
    _resolve_inputs,
    _row_to_atoms,
    DEFAULT_COLUMNS,
)


# Property-column name aliases. ColabFit datasets vary in spelling.
ENERGY_COLS  = ("energy", "potential_energy", "free_energy", "total_energy")
FORCES_COLS  = ("forces", "atomic_forces", "force")
STRESS_COLS  = ("stress", "cauchy_stress", "virial")
ID_COLS      = ("id", "_id", "configuration_id", "config_id")


def _detect_col(schema_names: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name (case-insensitive), or None."""
    lower_map = {n.lower(): n for n in schema_names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _attach_energy(atoms: Atoms, val) -> None:
    if val is not None and np.isfinite(float(val)):
        atoms.info["energy"] = float(val)


def _attach_forces(atoms: Atoms, val) -> None:
    if val is None:
        return
    arr = np.asarray(val, dtype=float)
    if arr.size == 0:
        return
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return
        arr = arr.reshape(-1, 3)
    if arr.shape[0] == len(atoms) and arr.shape[1] == 3:
        atoms.arrays["forces"] = arr


def _attach_stress(atoms: Atoms, val) -> None:
    if val is None:
        return
    arr = np.asarray(val, dtype=float).ravel()
    # Accept 6-vector (Voigt) or 9-vector (3x3 flat); store as 3x3 matrix,
    # which is what ASE's extxyz writer expects and what MLIP training
    # pipelines (MACE, NequIP, etc.) read.
    if arr.size == 9:
        atoms.info["stress"] = arr.reshape(3, 3)
    elif arr.size == 6:
        # Voigt -> full tensor (xx, yy, zz, yz, xz, xy)
        s = np.array([[arr[0], arr[5], arr[4]],
                      [arr[5], arr[1], arr[3]],
                      [arr[4], arr[3], arr[2]]])
        atoms.info["stress"] = s
    # otherwise silently skip (unknown layout)


def load_properties(po_paths, id_col_candidates=ID_COLS) -> dict:
    """
    Load property_objects parquet into {configuration_id: {energy, forces, stress}}.
    Streams the input in batches; auto-detects column names.
    """
    files = _resolve_inputs(po_paths)
    dataset = pa_ds.dataset(files, format="parquet")
    schema_names = set(dataset.schema.names)

    # Find the FK-back-to-config id column. Prefer "configuration_id".
    cid_col = _detect_col(schema_names, ("configuration_id", "config_id"))
    if cid_col is None:
        # Some normalized schemas use just "id" linking back; try anyway.
        cid_col = _detect_col(schema_names, ("id",))
    if cid_col is None:
        raise ValueError(
            f"property table has no configuration_id column. Schema: "
            f"{sorted(schema_names)}"
        )

    e_col = _detect_col(schema_names, ENERGY_COLS)
    f_col = _detect_col(schema_names, FORCES_COLS)
    s_col = _detect_col(schema_names, STRESS_COLS)
    if not any((e_col, f_col, s_col)):
        raise ValueError(
            f"property table has no energy/forces/stress column. Schema: "
            f"{sorted(schema_names)}"
        )

    print(f"  property table columns: id={cid_col!r}, energy={e_col!r}, "
          f"forces={f_col!r}, stress={s_col!r}", flush=True)

    out: dict = defaultdict(dict)
    for batch in dataset.scanner(batch_size=2000).to_batches():
        for row in batch.to_pylist():
            cid = row.get(cid_col)
            if cid is None:
                continue
            if e_col and row.get(e_col) is not None:
                out[cid]["energy"] = row[e_col]
            if f_col and row.get(f_col) is not None:
                out[cid]["forces"] = row[f_col]
            if s_col and row.get(s_col) is not None:
                out[cid]["stress"] = row[s_col]
    return dict(out)


def convert_parquet_to_xyz(
    in_paths,
    out_path: str,
    properties_paths=None,
    columns: dict | None = None,
    batch_size: int = 1000,
    log_every: int = 10000,
) -> int:
    """
    Stream-convert a parquet dataset to extxyz. Returns number of frames written.
    """
    cols = dict(DEFAULT_COLUMNS)
    if columns:
        cols.update(columns)

    files = _resolve_inputs(in_paths)
    print(f"Reading {len(files)} parquet file(s); first: {files[0]}", flush=True)
    dataset = pa_ds.dataset(files, format="parquet")
    schema_names = set(dataset.schema.names)

    # Auto-detect inline property columns.
    e_col = _detect_col(schema_names, ENERGY_COLS)
    f_col = _detect_col(schema_names, FORCES_COLS)
    s_col = _detect_col(schema_names, STRESS_COLS)
    id_col = _detect_col(schema_names, ID_COLS)
    if any((e_col, f_col, s_col)):
        print(f"  inline properties: energy={e_col!r} forces={f_col!r} "
              f"stress={s_col!r}", flush=True)

    # Optionally load external properties table.
    props_map: dict = {}
    if properties_paths is not None:
        if id_col is None:
            raise ValueError(
                f"configurations table has no id column; cannot link to "
                f"property table. Available columns: {sorted(schema_names)}"
            )
        print(f"Loading external property table...", flush=True)
        props_map = load_properties(properties_paths)
        print(f"  loaded {len(props_map):,} configurations with properties",
              flush=True)

    n = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for batch in dataset.scanner(batch_size=batch_size).to_batches():
            for row in batch.to_pylist():
                atoms = _row_to_atoms(row, cols)

                # 1. Inline properties.
                if e_col is not None:
                    _attach_energy(atoms, row.get(e_col))
                if f_col is not None:
                    _attach_forces(atoms, row.get(f_col))
                if s_col is not None:
                    _attach_stress(atoms, row.get(s_col))

                # 2. External properties (override or fill in).
                if props_map and id_col is not None:
                    cid = row.get(id_col)
                    p = props_map.get(cid)
                    if p:
                        if "energy" in p:
                            _attach_energy(atoms, p["energy"])
                        if "forces" in p:
                            _attach_forces(atoms, p["forces"])
                        if "stress" in p:
                            _attach_stress(atoms, p["stress"])

                ase_write(f, atoms, format="extxyz", append=True)
                n += 1
                if log_every and n % log_every == 0:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  [{n:>10d} frames] {rate:7.0f} struct/s",
                          flush=True)

    print(f"Wrote {n:,} frames to {out_path}", flush=True)
    return n


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Convert a ColabFit-style parquet dataset to extxyz.",
        epilog="Inputs can be one or more parquet files, a directory of "
               "parquet files (e.g. <dataset>/co/), or a shell glob.",
    )
    p.add_argument("inputs", nargs="+",
                   help="Input parquet file(s), directory, or glob.")
    p.add_argument("-o", "--output", required=True,
                   help="Output extxyz file path.")
    p.add_argument("--properties", default=None, nargs="+",
                   help="Optional path(s) to property_objects table "
                        "(file, dir, or glob). Pulls energy / forces / "
                        "stress and links by configuration id.")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--col-numbers",   default="atomic_numbers")
    p.add_argument("--col-positions", default="positions")
    p.add_argument("--col-cell",      default="cell")
    p.add_argument("--col-pbc",       default="pbc")
    args = p.parse_args()

    columns = {
        "numbers":   args.col_numbers,
        "positions": args.col_positions,
        "cell":      args.col_cell,
        "pbc":       args.col_pbc,
    }

    convert_parquet_to_xyz(
        args.inputs,
        args.output,
        properties_paths=args.properties,
        columns=columns,
        batch_size=args.batch_size,
    )
