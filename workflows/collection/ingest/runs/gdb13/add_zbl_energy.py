#!/usr/bin/env python3
"""Add ZBL screened-repulsion energy columns to the GDB-13 merged shards.

The merge step (submit_zbl.sh -> zbl_filter.py) already stored `max_force` in every
work/gdb13/merged/<stem>.parquet. This adds the two further quantities the locked
candidate filter cuts on, computed from the stored geometry
(atomic_numbers / positions / cell / pbc):

    zbl_energy_per_atom      total ZBL energy / n_atoms          [eV/atom]
    zbl_max_atomic_energy    most-repulsive single atom's share  [eV/atom]

After this the shards carry all three (max_force, zbl_energy_per_atom,
zbl_max_atomic_energy), so FPS can apply the filter at load time:
    keep iff max_force < 200 AND zbl_energy_per_atom <= 50 AND zbl_max_atomic_energy <= 120

The physics is the verified module fps/zbl_energy.py (selftested to match
the production zbl_filter.py to 3.5e-16); it is imported here, not reimplemented, so the
energy stays consistent with the stored max_force (same ZBL constants, 5 A cutoff, vesin
neighbour list, and the n_atoms<2 guard removed).

Work is split over shards: each invocation owns shards[task_id::num_tasks] and rewrites
them in place (write-to-tmp + atomic replace, skip if the columns already exist), so the
job is restartable and finished shards are never recomputed. Within a shard the per-row
ZBL energy is computed in parallel across the parquet row groups.

Usage
-----
    # add the columns (one stripe of the pool)
    python add_zbl_energy.py --merged-dir .../work/gdb13/merged \
        --task-id 0 --num-tasks 8 --workers 16

    # report the filter impact once everything is done
    python add_zbl_energy.py --merged-dir .../work/gdb13/merged --summarize
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MAD_CORE = "${DEV_ROOT}/mad-core"
sys.path.insert(0, os.path.join(MAD_CORE, "fps"))
from zbl_energy import zbl_energy as zbl_energy_of_structure   # verified ZBL physics

GEOMETRY_COLUMNS = ["atomic_numbers", "positions", "cell", "pbc"]
NEW_COLUMNS = ("zbl_energy_per_atom", "zbl_max_atomic_energy")
ENERGY_PER_ATOM_MAX = 50.0
MAX_ATOMIC_ENERGY_MAX = 120.0
ROW_GROUP_SIZE = 20_000


def _row_group_energies(args):
    """ZBL (energy_per_atom, max_atomic_energy) for one row group, read from disk.

    The worker re-reads only the geometry columns of its row group (cheap) rather than
    receiving them over IPC, which keeps the parent's memory and the pickling cost low.
    """
    path, row_group = args
    table = pq.ParquetFile(path).read_row_groups([row_group], columns=GEOMETRY_COLUMNS)
    numbers = table.column("atomic_numbers").to_pylist()
    positions = table.column("positions").to_pylist()
    cells = table.column("cell").to_pylist()
    pbcs = table.column("pbc").to_pylist()

    n = table.num_rows
    energy_per_atom = np.empty(n)
    max_atomic_energy = np.empty(n)
    for k in range(n):
        energy_per_atom[k], max_atomic_energy[k] = zbl_energy_of_structure(
            numbers[k], positions[k], cells[k], pbcs[k])
    return energy_per_atom, max_atomic_energy


def process_shard(path, pool):
    """Compute the two ZBL columns for one shard and rewrite it in place (atomic)."""
    table = pq.read_table(path)
    if all(name in table.column_names for name in NEW_COLUMNS):
        return "skip", table.num_rows

    n_row_groups = pq.ParquetFile(path).metadata.num_row_groups
    results = pool.map(_row_group_energies, [(path, rg) for rg in range(n_row_groups)])

    # row groups partition the table in order, so concatenating in row-group order
    # aligns exactly with the full-table row order read above.
    energy_per_atom = np.concatenate([r[0] for r in results])
    max_atomic_energy = np.concatenate([r[1] for r in results])
    assert len(energy_per_atom) == table.num_rows, f"{path}: row count mismatch"

    table = table.append_column("zbl_energy_per_atom", pa.array(energy_per_atom))
    table = table.append_column("zbl_max_atomic_energy", pa.array(max_atomic_energy))

    tmp = path + ".tmp"
    pq.write_table(table, tmp, compression="zstd", row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp, path)
    return "done", table.num_rows


def add_columns(merged_dir, task_id, num_tasks, workers):
    shards = sorted(glob.glob(os.path.join(merged_dir, "*.parquet")))
    mine = shards[task_id::num_tasks]
    print(f"[task {task_id}/{num_tasks}] {len(mine)} of {len(shards)} shards, workers={workers}", flush=True)

    with Pool(workers) as pool:
        for i, path in enumerate(mine):
            status, n = process_shard(path, pool)
            print(f"[task {task_id}] [{i + 1}/{len(mine)}] {status:4s} "
                  f"{os.path.basename(path)} rows={n:,}", flush=True)
    print(f"[task {task_id}] all shards handled", flush=True)


def _shard_stats(path):
    table = pq.read_table(path, columns=list(NEW_COLUMNS))
    energy_per_atom = table.column("zbl_energy_per_atom").to_numpy()
    max_atomic_energy = table.column("zbl_max_atomic_energy").to_numpy()
    drop = (energy_per_atom > ENERGY_PER_ATOM_MAX) | (max_atomic_energy > MAX_ATOMIC_ENERGY_MAX)
    return (table.num_rows,
            float(energy_per_atom.max()), float(max_atomic_energy.max()),
            int((energy_per_atom > ENERGY_PER_ATOM_MAX).sum()),
            int((max_atomic_energy > MAX_ATOMIC_ENERGY_MAX).sum()),
            int(drop.sum()))


def summarize(merged_dir, workers):
    shards = sorted(glob.glob(os.path.join(merged_dir, "*.parquet")))
    missing = [p for p in shards if not set(NEW_COLUMNS) <= set(pq.ParquetFile(p).schema_arrow.names)]
    if missing:
        print(f"WARNING: {len(missing)} shards still missing the ZBL energy columns, e.g. "
              f"{os.path.basename(missing[0])}", flush=True)

    done = [p for p in shards if p not in set(missing)]
    with Pool(min(workers, len(done))) as pool:
        stats = pool.map(_shard_stats, done)

    n = sum(s[0] for s in stats)
    n_epa = sum(s[3] for s in stats)
    n_atomic = sum(s[4] for s in stats)
    n_drop = sum(s[5] for s in stats)
    print(f"\n=== GDB-13 ZBL energy filter impact  ({len(done)}/{len(shards)} shards, {n:,} structures) ===")
    print(f"  max zbl_energy_per_atom   : {max(s[1] for s in stats):.2f} eV/atom")
    print(f"  max zbl_max_atomic_energy : {max(s[2] for s in stats):.2f} eV/atom")
    print(f"  drop zbl_energy_per_atom   > {ENERGY_PER_ATOM_MAX:g} : {n_epa:,} ({100 * n_epa / n:.4f}%)")
    print(f"  drop zbl_max_atomic_energy > {MAX_ATOMIC_ENERGY_MAX:g}: {n_atomic:,} ({100 * n_atomic / n:.4f}%)")
    print(f"  dropped by either (union)            : {n_drop:,} ({100 * n_drop / n:.4f}%)")
    print(f"  kept                                 : {n - n_drop:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--summarize", action="store_true",
                    help="report the filter impact across all shards (run after the columns are added)")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.merged_dir, args.workers)
    else:
        add_columns(args.merged_dir, args.task_id, args.num_tasks, args.workers)


if __name__ == "__main__":
    main()
