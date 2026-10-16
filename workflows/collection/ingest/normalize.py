from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ids import FROZEN_PARAMS

SCHEMA = pa.schema([
    # identity
    pa.field("configuration_id", pa.string()),
    pa.field("configuration_hash", pa.string()),
    pa.field("structure_hash", pa.string()),
    pa.field("dataset_id", pa.string()),
    pa.field("source_id", pa.string()),                       # native id, nullable
    pa.field("names", pa.list_(pa.string())),
    # geometry EXACT types systems_from_table reads
    pa.field("atomic_numbers", pa.list_(pa.int64())),
    pa.field("positions", pa.list_(pa.list_(pa.float64()))),
    pa.field("cell", pa.list_(pa.list_(pa.float64()))),
    pa.field("pbc", pa.list_(pa.bool_())),
    # carried training labels (nullable)
    pa.field("energy", pa.float64()),
    pa.field("atomic_forces", pa.list_(pa.list_(pa.float64()))),
    pa.field("cauchy_stress", pa.list_(pa.list_(pa.float64()))),
    pa.field("nsites", pa.int32()),
])


def _build_listlist_f8(rows):
    natoms = np.array([0 if r is None else r.shape[0] for r in rows], dtype=np.int64)
    present = [r.reshape(-1) for r in rows if r is not None]
    flat = np.concatenate(present).astype(np.float64) if present else np.empty(0, np.float64)
    total = int(natoms.sum())
    inner_off = pa.array(np.arange(total + 1, dtype=np.int32) * 3, type=pa.int32())
    inner = pa.ListArray.from_arrays(inner_off, pa.array(flat, type=pa.float64()))
    outer_off = pa.array(np.concatenate(([0], np.cumsum(natoms))).astype(np.int32), type=pa.int32())
    mask = pa.array([r is None for r in rows]) if any(r is None for r in rows) else None
    return pa.ListArray.from_arrays(outer_off, inner, mask=mask)


def _build_list_i64(rows):
    natoms = np.array([r.shape[0] for r in rows], dtype=np.int64)
    flat = np.concatenate(rows).astype(np.int64) if rows else np.empty(0, np.int64)
    off = pa.array(np.concatenate(([0], np.cumsum(natoms))).astype(np.int32), type=pa.int32())
    return pa.ListArray.from_arrays(off, pa.array(flat, type=pa.int64()))


def _build_list_bool(rows):
    flat = np.concatenate(rows).astype(bool) if rows else np.empty(0, bool)
    off = pa.array((np.arange(len(rows) + 1) * 3).astype(np.int32), type=pa.int32())
    return pa.ListArray.from_arrays(off, pa.array(flat, type=pa.bool_()))


class ShardWriter:
    def __init__(self, out_path, dataset_id, *, chunk_rows=20000, compression="snappy"):
        self.out_path = Path(out_path)
        self.tmp_path = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        self.sidecar = self.out_path.with_suffix(".json")
        self.dataset_id = dataset_id
        self.chunk_rows = chunk_rows
        self.compression = compression
        self._writer = None
        self._n = 0           # rows written
        self._inputs = set()
        self._reset_buf()

    def _reset_buf(self):
        self._cid, self._chash, self._shash = [], [], []
        self._sid, self._names = [], []
        self._numbers, self._pos, self._cell, self._pbc = [], [], [], []
        self._energy, self._forces, self._stress, self._nsites = [], [], [], []
        self._buf = 0

    def add(self, rec, ids, names):
        self._cid.append(ids["configuration_id"])
        self._chash.append(ids["configuration_hash"])
        self._shash.append(ids["structure_hash"])
        self._sid.append(rec["source_id"])
        self._names.append(names)
        self._numbers.append(rec["numbers"])
        self._pos.append(rec["positions"])
        self._cell.append(rec["cell"])
        self._pbc.append(rec["pbc"])
        self._energy.append(rec["energy"])
        self._forces.append(rec["forces"])
        self._stress.append(rec["stress"])
        self._nsites.append(int(rec["numbers"].shape[0]))
        self._inputs.add(rec["src_file"])
        self._buf += 1
        if self._buf >= self.chunk_rows:
            self._flush()

    def _flush(self):
        if self._buf == 0:
            return
        batch = pa.table({
            "configuration_id": pa.array(self._cid, pa.string()),
            "configuration_hash": pa.array(self._chash, pa.string()),
            "structure_hash": pa.array(self._shash, pa.string()),
            "dataset_id": pa.array([self.dataset_id] * self._buf, pa.string()),
            "source_id": pa.array(self._sid, pa.string()),
            "names": pa.array(self._names, pa.list_(pa.string())),
            "atomic_numbers": _build_list_i64(self._numbers),
            "positions": _build_listlist_f8(self._pos),
            "cell": _build_listlist_f8(self._cell),
            "pbc": _build_list_bool(self._pbc),
            "energy": pa.array(self._energy, pa.float64()),
            "atomic_forces": _build_listlist_f8(self._forces),
            "cauchy_stress": _build_listlist_f8(self._stress),
            "nsites": pa.array(self._nsites, pa.int32()),
        }, schema=SCHEMA)
        if self._writer is None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.tmp_path, SCHEMA, compression=self.compression)
        self._writer.write_table(batch)
        self._n += self._buf
        self._reset_buf()

    def close(self):
        self._flush()
        if self._writer is not None:
            self._writer.close()
            os.replace(self.tmp_path, self.out_path)
        else:  # zero rows -> write an empty valid file so the shard isn't re-attempted forever
            self.out_path.parent.mkdir(parents=True, exist_ok=True)  # _flush() never ran, so mkdir here
            pq.write_table(SCHEMA.empty_table(), self.tmp_path, compression=self.compression)
            os.replace(self.tmp_path, self.out_path)
        self.sidecar.write_text(json.dumps({
            "dataset_id": self.dataset_id,
            "n_rows": self._n,
            "n_inputs": len(self._inputs),
            "inputs": sorted(self._inputs),
            "id_params": FROZEN_PARAMS,
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        return self._n
