from __future__ import annotations
import numpy as np

# Required + optional keys of a record dict:
#   numbers    : np.ndarray[int64]   (n_atoms,)
#   positions  : np.ndarray[float64] (n_atoms, 3)   cartesian, Angstrom
#   cell       : np.ndarray[float64] (3, 3)         zeros if non-periodic
#   pbc        : np.ndarray[bool]    (3,)
#   energy     : float | None
#   forces     : np.ndarray[float64] (n_atoms, 3) | None
#   stress     : np.ndarray[float64] (3, 3) | None
#   src_file   : str                 basename of the source file
#   frame_idx  : int                 0-based index within src_file
#   source_id  : str | None          native unique id from the source, if any


def make_record(numbers, positions, cell, pbc, *, src_file, frame_idx,
                energy=None, forces=None, stress=None, source_id=None,
                extra_names=None) -> dict:
    return {
        "extra_names": list(extra_names) if extra_names else [],
        "numbers": np.ascontiguousarray(numbers, dtype=np.int64),
        "positions": np.ascontiguousarray(positions, dtype=np.float64).reshape(-1, 3),
        "cell": np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3),
        "pbc": np.ascontiguousarray(np.asarray(pbc, dtype=bool)).reshape(3),
        "energy": None if energy is None else float(energy),
        "forces": None if forces is None else np.ascontiguousarray(forces, dtype=np.float64).reshape(-1, 3),
        "stress": None if stress is None else np.ascontiguousarray(stress, dtype=np.float64).reshape(3, 3),
        "src_file": str(src_file),
        "frame_idx": int(frame_idx),
        "source_id": None if source_id is None else str(source_id),
    }


def validate_record(rec: dict) -> None:
    n = rec["numbers"].shape[0]
    assert n > 0, "empty structure"
    assert rec["positions"].shape == (n, 3), f"positions shape {rec['positions'].shape}"
    assert rec["cell"].shape == (3, 3), f"cell shape {rec['cell'].shape}"
    assert rec["pbc"].shape == (3,), f"pbc shape {rec['pbc'].shape}"
    assert np.isfinite(rec["positions"]).all(), "non-finite positions"
    assert np.isfinite(rec["cell"]).all(), "non-finite cell"
    assert (rec["numbers"] > 0).all() and (rec["numbers"] < 119).all(), "bad atomic numbers"
    if rec["forces"] is not None:
        assert rec["forces"].shape == (n, 3), f"forces shape {rec['forces'].shape}"
        assert np.isfinite(rec["forces"]).all(), "non-finite forces"
    if rec["stress"] is not None:
        assert rec["stress"].shape == (3, 3), f"stress shape {rec['stress'].shape}"
    if rec["energy"] is not None:
        assert np.isfinite(rec["energy"]), "non-finite energy"
