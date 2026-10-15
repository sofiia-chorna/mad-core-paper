from __future__ import annotations
import bz2
import gzip
import json
import os
import numpy as np
from ase.data import atomic_numbers as ASE_Z

from ..records import make_record


def _open(path):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt")
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def read(path):
    base = os.path.basename(path)
    with _open(path) as f:
        data = json.load(f)
    entries = data["entries"] if isinstance(data, dict) else data
    for frame, e in enumerate(entries):
        s = e["structure"]
        cell = np.asarray(s["lattice"]["matrix"], dtype=float)
        pbc = np.asarray(s["lattice"].get("pbc", [True, True, True]), dtype=bool)
        sites = s["sites"]
        n = len(sites)
        numbers = np.empty(n, dtype=np.int64)
        pos = np.empty((n, 3), dtype=float)
        have_cart = "xyz" in sites[0]
        props0 = sites[0].get("properties", {})
        have_f = "forces" in props0
        forces = np.empty((n, 3), dtype=float) if have_f else None
        for k, site in enumerate(sites):
            numbers[k] = ASE_Z[site["species"][0]["element"]]
            pos[k] = site["xyz"] if have_cart else (np.asarray(site["abc"], float) @ cell)
            if have_f:
                forces[k] = site["properties"]["forces"]
        energy = e.get("energy")
        d = e.get("data", {})
        stress = np.asarray(d["stress"], float) if "stress" in d else None
        # mat_id is NOT globally unique for the pressure set (same material at
        # multiple pressures, one per file) -> prefix the source-file stem so
        # configuration_id stays unique per row. (pbesol mat_ids are already unique;
        # the prefix is harmless there.)
        native = e.get("entry_id") or d.get("mat_id")
        stem = base.split(".")[0]
        source_id = f"{stem}:{native}" if native else None
        yield make_record(
            numbers, pos, cell, pbc,
            energy=energy, forces=forces, stress=stress,
            src_file=base, frame_idx=frame, source_id=source_id,
        )
