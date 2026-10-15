from __future__ import annotations
import hashlib
import numpy as np

POS_DECIMALS = 6
CELL_DECIMALS = 6
DIGEST_SIZE = 16


def _canon_geom_bytes(numbers, positions, cell, pbc) -> bytes:
    fields = [
        np.ascontiguousarray(numbers, dtype="<i8").tobytes(),
        np.ascontiguousarray(np.round(np.asarray(positions, float), POS_DECIMALS), dtype="<f8").tobytes(),
        np.ascontiguousarray(np.round(np.asarray(cell, float), CELL_DECIMALS), dtype="<f8").tobytes(),
        np.ascontiguousarray(np.asarray(pbc, dtype=bool), dtype="?").tobytes(),
    ]
    return b"".join(len(f).to_bytes(8, "little") + f for f in fields)


def structure_hash(numbers, positions, cell, pbc) -> str:
    return hashlib.blake2b(_canon_geom_bytes(numbers, positions, cell, pbc), digest_size=DIGEST_SIZE).hexdigest()


def _mint_config_hash(struct_hash, dataset, src_file, frame_idx, energy) -> str:
    e = "" if energy is None else f"{round(float(energy), 8)}"
    prov = f"{struct_hash}|{dataset}|{src_file}|{frame_idx}|e={e}"
    return hashlib.blake2b(prov.encode("utf-8"), digest_size=DIGEST_SIZE).hexdigest()


def mint_ids(numbers, positions, cell, pbc, *, dataset, src_file, frame_idx, energy=None, source_id=None) -> dict:
    sh = structure_hash(numbers, positions, cell, pbc)
    ch = _mint_config_hash(sh, dataset, src_file, frame_idx, energy)
    cid = f"{dataset}:{source_id}" if source_id else f"{dataset}:{ch}"
    return {"structure_hash": sh, "configuration_hash": ch, "configuration_id": cid}


FROZEN_PARAMS = {
    "algo": "blake2b",
    "digest_size": DIGEST_SIZE,
    "pos_decimals": POS_DECIMALS,
    "cell_decimals": CELL_DECIMALS,
    "config_id": "unified '{dataset_id}:<native_id or config_hash>'",
}
