from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

try:
    import vesin
    _HAS_VESIN = True
except ImportError:
    _HAS_VESIN = False

# pyykko 2009 single-bond covalent radii in Angstrom (Z to r_cov)
PYYKKO_COVALENT_RADII: dict[int, float] = {
    1: 0.32,  2: 0.46,
    3: 1.33,  4: 1.02,  5: 0.85,  6: 0.75,  7: 0.71,  8: 0.63,  9: 0.64, 10: 0.67,
    11: 1.55, 12: 1.39, 13: 1.26, 14: 1.16, 15: 1.11, 16: 1.03, 17: 0.99, 18: 0.96,
    19: 1.96, 20: 1.71,
    21: 1.48, 22: 1.36, 23: 1.34, 24: 1.22, 25: 1.19, 26: 1.16, 27: 1.11, 28: 1.10,
    29: 1.12, 30: 1.18,
    31: 1.24, 32: 1.21, 33: 1.21, 34: 1.16, 35: 1.14, 36: 1.17,
    37: 2.10, 38: 1.85,
    39: 1.63, 40: 1.54, 41: 1.47, 42: 1.38, 43: 1.28, 44: 1.25, 45: 1.25, 46: 1.20,
    47: 1.28, 48: 1.36,
    49: 1.42, 50: 1.40, 51: 1.40, 52: 1.36, 53: 1.33, 54: 1.31,
    55: 2.32, 56: 1.96,
    57: 1.80,
    58: 1.63, 59: 1.76, 60: 1.74, 61: 1.73, 62: 1.72, 63: 1.68, 64: 1.69,
    65: 1.68, 66: 1.67, 67: 1.66, 68: 1.65, 69: 1.64, 70: 1.70, 71: 1.62,
    72: 1.52, 73: 1.46, 74: 1.37, 75: 1.31, 76: 1.29, 77: 1.22, 78: 1.23, 79: 1.24,
    80: 1.33,
    81: 1.44, 82: 1.44, 83: 1.51, 84: 1.45, 85: 1.47, 86: 1.42,
    87: 2.23, 88: 2.01,
    89: 1.86,
    90: 1.75, 91: 1.69, 92: 1.70, 93: 1.71, 94: 1.72, 95: 1.66, 96: 1.66,
    97: 1.68, 98: 1.68, 99: 1.65, 100: 1.67, 101: 1.73, 102: 1.76, 103: 1.61,
}


def build_min_distance_matrix(scale: float = 0.5, max_z: int = 103) -> np.ndarray:
    radii = np.full(max_z + 1, np.nan, dtype=np.float64)
    for z, r in PYYKKO_COVALENT_RADII.items():
        if z <= max_z:
            radii[z] = r
    return scale * (radii[:, None] + radii[None, :])


def supported_elements() -> set[int]:
    return set(PYYKKO_COVALENT_RADII.keys())


def _neighbor_pairs_ase(atoms: Atoms, cutoff: float):
    return neighbor_list("ijdS", atoms, cutoff=cutoff, self_interaction=False)


def _neighbor_pairs_vesin(atoms: Atoms, cutoff: float):
    box = np.asarray(atoms.cell.array, dtype=np.float64).copy()
    pbc = np.asarray(atoms.pbc, dtype=bool)

    # for non-periodic axes, vesin still needs a sensible bounding box
    if not pbc.all():
        pos = atoms.positions
        for ax in range(3):
            if not pbc[ax] and np.linalg.norm(box[ax]) < 2 * cutoff:
                # replace with axis-aligned vector spanning the data + margin
                span = float(pos[:, ax].max() - pos[:, ax].min()) if len(pos) else 0.0
                v = np.zeros(3)
                v[ax] = span + 4 * cutoff
                box[ax] = v

    nl = vesin.NeighborList(cutoff=cutoff, full_list=False)
    i, j, d, S = nl.compute(
        points=atoms.positions, box=box, periodic=pbc, quantities="ijdS"
    )
    return i.astype(int), j.astype(int), d, S.astype(int)


def find_close_contacts(
    atoms: Atoms,
    min_dist_matrix: np.ndarray,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z = atoms.numbers
    if Z.size == 0:
        e_i = np.array([], dtype=int)
        return e_i, e_i, e_i.astype(float), np.zeros((0, 3), dtype=int)

    Z_present = np.unique(Z)
    max_z_supported = min_dist_matrix.shape[0] - 1
    if Z_present.max() > max_z_supported:
        too_big = sorted(int(z) for z in Z_present if z > max_z_supported)
        raise ValueError(
            f"Z={too_big} exceeds matrix size (max_z={max_z_supported}); "
            f"rebuild with build_min_distance_matrix(max_z=...)"
        )
    sub = min_dist_matrix[np.ix_(Z_present, Z_present)]
    if np.isnan(sub).any():
        missing = sorted(int(z) for z in Z_present
                         if z not in PYYKKO_COVALENT_RADII)
        raise ValueError(f"No tabulated covalent radius for Z={missing}")

    nl_cutoff = float(sub.max())
    if nl_cutoff <= 0:
        e_i = np.array([], dtype=int)
        return e_i, e_i, e_i.astype(float), np.zeros((0, 3), dtype=int)

    # backend dispatch
    use_vesin = (backend == "vesin") or (backend == "auto" and _HAS_VESIN)
    if backend == "vesin" and not _HAS_VESIN:
        raise ImportError("vesin is not installed; pip install vesin")

    if use_vesin:
        i, j, d, S = _neighbor_pairs_vesin(atoms, nl_cutoff)
    else:
        # self-image with first-nonzero-S 0)
        i, j, d, S = _neighbor_pairs_ase(atoms, nl_cutoff)
        S = S.astype(int)
        if i.size > 0:
            i_lt_j = i < j
            i_eq_j = i == j
            first_nz_idx = np.argmax(S != 0, axis=1)
            first_nz_val = S[np.arange(S.shape[0]), first_nz_idx]
            canon = i_lt_j | (i_eq_j & (first_nz_val > 0))
            i, j, d, S = i[canon], j[canon], d[canon], S[canon]

    if i.size == 0:
        return i, j, d, S

    # filter to violations
    pair_min = min_dist_matrix[Z[i], Z[j]]
    bad = d < pair_min
    return i[bad], j[bad], d[bad], S[bad]


def has_close_contacts(
    atoms: Atoms, min_dist_matrix: np.ndarray, backend: str = "auto",
) -> bool:
    i, _, _, _ = find_close_contacts(atoms, min_dist_matrix, backend=backend)
    return i.size > 0


def describe_violations(
    atoms: Atoms,
    min_dist_matrix: np.ndarray,
    backend: str = "auto",
) -> list[dict]:
    i, j, d, S = find_close_contacts(atoms, min_dist_matrix, backend=backend)
    syms = atoms.get_chemical_symbols()
    Z = atoms.numbers
    order = np.argsort(d)
    out = []
    for k in order:
        cutoff = float(min_dist_matrix[Z[i[k]], Z[j[k]]])
        out.append({
            "i": int(i[k]),
            "j": int(j[k]),
            "elements": f"{syms[i[k]]}-{syms[j[k]]}",
            "distance": float(d[k]),
            "cutoff": cutoff,
            "offset": (int(S[k][0]), int(S[k][1]), int(S[k][2])),
            "is_self_image": bool(i[k] == j[k]),
        })
    return out


# stream filter for extxyz shards
def filter_extxyz(
    in_path: str,
    out_path: str,
    rejected_path: str | None = None,
    scale: float = 0.5,
    log_every: int = 10000,
    verbose: bool = False,
    backend: str = "auto",
) -> dict:
    from ase.io import iread, write
    from collections import Counter
    import time

    mdm = build_min_distance_matrix(scale=scale)
    passed = rejected = 0
    pair_counter: Counter = Counter()
    t0 = time.time()

    out_fh = open(out_path, "w")
    rej_fh = open(rejected_path, "w") if rejected_path else None
    try:
        for n, atoms in enumerate(iread(in_path), start=1):
            i, j, d, S = find_close_contacts(atoms, mdm, backend=backend)
            if i.size == 0:
                passed += 1
                write(out_fh, atoms, format="extxyz", append=True)
            else:
                rejected += 1
                # worst pair smallest distance
                k = int(np.argmin(d))
                syms = atoms.get_chemical_symbols()
                Zi = int(atoms.numbers[i[k]])
                Zj = int(atoms.numbers[j[k]])
                # canonical element-pair key (alphabetical, so C-O and O-C aggregate)
                pair_key = tuple(sorted([syms[i[k]], syms[j[k]]]))
                pair_counter[pair_key] += 1

                if verbose:
                    offset = (int(S[k][0]), int(S[k][1]), int(S[k][2]))
                    tag = "self-image" if i[k] == j[k] else (
                        "cross-image" if offset != (0, 0, 0) else "in-cell")
                    print(f"  reject #{rejected:>7d} frame={n:<8d} "
                          f"{syms[i[k]]:>2s}({int(i[k])})-"
                          f"{syms[j[k]]:>2s}({int(j[k])}) "
                          f"d={float(d[k]):.3f} cutoff={float(mdm[Zi,Zj]):.3f} "
                          f"S={offset} [{tag}]", flush=True)

                if rej_fh is not None:
                    atoms.info["bad_i"] = int(i[k])
                    atoms.info["bad_j"] = int(j[k])
                    atoms.info["bad_elements"] = f"{syms[i[k]]}-{syms[j[k]]}"
                    atoms.info["bad_distance"] = float(d[k])
                    atoms.info["bad_cutoff"] = float(mdm[Zi, Zj])
                    atoms.info["bad_offset"] = np.array(
                        [int(S[k][0]), int(S[k][1]), int(S[k][2])], dtype=int
                    )
                    atoms.info["bad_is_self_image"] = int(i[k] == j[k])
                    atoms.info["n_bad_pairs"] = int(i.size)
                    write(rej_fh, atoms, format="extxyz", append=True)

            if log_every and n % log_every == 0:
                rate = n / max(time.time() - t0, 1e-9)
                top = pair_counter.most_common(3)
                top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top)
                print(f"[{n:>10d}] {rate:7.0f} struct/s | "
                      f"passed={passed} rejected={rejected}"
                      + (f" | top: {top_str}" if top else ""), flush=True)
    finally:
        out_fh.close()
        if rej_fh is not None:
            rej_fh.close()

    # end-of-run summary of which element pairs caused rejections
    if rejected > 0:
        print(f"\nRejection causes by element pair "
              f"(unique pairs: {len(pair_counter)}):")
        for (a, b), c in pair_counter.most_common():
            pct = 100.0 * c / rejected
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")

    return {
        "passed": passed,
        "rejected": rejected,
        "total": passed + rejected,
        "rejection_pairs": pair_counter,
    }


# CLI
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Filter unphysical structures from an extxyz shard "
                    "using Pyykko covalent-radius distance cutoffs."
    )
    p.add_argument("input", help="Input extxyz file.")
    p.add_argument("output", help="Output extxyz for passing structures.")
    p.add_argument("--rejected", default=None,
                   help="Optional path to write rejected structures.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor "
                        "(0.5 permissive, 0.6 standard, 0.75 conservative). "
                        "Default: 0.5.")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per rejection with the worst pair details.")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto",
                   help="Neighbor-list backend. 'auto' prefers vesin if installed.")
    args = p.parse_args()

    stats = filter_extxyz(
        args.input, args.output,
        rejected_path=args.rejected,
        scale=args.scale,
        verbose=args.verbose,
        backend=args.backend,
    )
    print(f"Done: passed={stats['passed']} rejected={stats['rejected']} "
          f"total={stats['total']}")
