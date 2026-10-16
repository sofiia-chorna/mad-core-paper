from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from ase import Atoms

# reuse the core filter logic
sys.path.insert(0, ".")
from pyykko_filter import build_min_distance_matrix, find_close_contacts


# default ColabFit-style column names
DEFAULT_COLUMNS = {
    "numbers":   "atomic_numbers",
    "positions": "positions",
    "cell":      "cell",
    "pbc":       "pbc",
}


def _maybe_decode_string(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return None
    # try JSON first (handles 'true'/'false' for pbc and standard arrays)
    try:
        import json as _json
        return _json.loads(s)
    except (ValueError, TypeError):
        pass
    # fallback Python literal_eval (handles 'True'/'False', tuples, etc.)
    import ast as _ast
    try:
        return _ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # last resort numpy fromstring on a whitespace-separated list
        try:
            return np.fromstring(s.strip("[]() "), sep=" ").tolist()
        except Exception:
            raise ValueError(
                f"cannot decode string-encoded array column: {s[:80]!r}..."
            )


def _to_2d(arr, n_atoms: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 3:
        return a
    if a.ndim == 1:
        if a.size % 3 != 0:
            raise ValueError(f"flat coordinate array length {a.size} not divisible by 3")
        return a.reshape(-1, 3)
    raise ValueError(f"unexpected coordinate array shape {a.shape}")


def _to_cell(arr) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return None
    if a.ndim == 2 and a.shape == (3, 3):
        cell = a
    elif a.ndim == 1 and a.size == 9:
        cell = a.reshape(3, 3)
    else:
        raise ValueError(f"unexpected cell shape {a.shape}")
    if not np.any(cell):
        return None
    return cell


def _to_pbc(val) -> np.ndarray:
    if val is None:
        return np.zeros(3, dtype=bool)
    if np.isscalar(val):
        return np.full(3, bool(val))
    a = np.asarray(val).ravel()
    if a.size == 3:
        return a.astype(bool)
    if a.size == 1:
        return np.full(3, bool(a[0]))
    raise ValueError(f"unexpected pbc shape {a.shape}")


def _row_to_atoms(row: dict, cols: dict,
                   positions_format: str = "cartesian") -> Atoms:
    # decode string-encoded columns (some ColabFit dumps store arrays as
    raw_numbers   = _maybe_decode_string(row[cols["numbers"]])
    raw_positions = _maybe_decode_string(row[cols["positions"]])
    raw_cell      = _maybe_decode_string(row.get(cols["cell"]))
    raw_pbc       = _maybe_decode_string(row.get(cols["pbc"]))

    numbers = np.asarray(raw_numbers, dtype=int).ravel()
    positions = _to_2d(raw_positions)
    if positions.shape[0] != numbers.size:
        raise ValueError(
            f"positions has {positions.shape[0]} atoms but atomic_numbers has "
            f"{numbers.size}"
        )
    cell = _to_cell(raw_cell)
    pbc = _to_pbc(raw_pbc)

    if positions_format == "fractional":
        if cell is None:
            raise ValueError(
                "positions_format='fractional' requires a non-null cell, but "
                "this row has no cell. The dataset can't be both fractional "
                "and non-periodic."
            )
        positions = positions @ cell
    elif positions_format != "cartesian":
        raise ValueError(
            f"positions_format must be 'cartesian' or 'fractional', "
            f"got {positions_format!r}"
        )

    return Atoms(numbers=numbers,
                 positions=positions,
                 cell=cell if cell is not None else None,
                 pbc=pbc)


class _LazyParquetWriter:
    def __init__(self, path: str, schema: pa.Schema, compression: str):
        self.path = path
        self.schema = schema
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None

    def write_table(self, table: pa.Table) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path, self.schema, compression=self.compression
            )
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def created(self) -> bool:
        return self._writer is not None or Path(self.path).exists()


def _resolve_inputs(in_paths) -> list[str]:
    if isinstance(in_paths, (str, Path)):
        in_paths = [in_paths]
    out: list[str] = []
    for entry in in_paths:
        p = Path(entry)
        if p.is_dir():
            files = sorted(p.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no *.parquet files in directory {entry}")
            out.extend(str(f) for f in files)
        elif p.is_file():
            out.append(str(p))
        else:
            # treat as glob pattern
            matched = sorted(glob.glob(str(entry)))
            if not matched:
                raise FileNotFoundError(f"no files match {entry}")
            out.extend(matched)
    if not out:
        raise FileNotFoundError("no input parquet files resolved")
    return out


def _make_rejected_schema(input_schema: pa.Schema) -> pa.Schema:
    extra = [
        pa.field("bad_i",            pa.int64()),
        pa.field("bad_j",            pa.int64()),
        pa.field("bad_elements",     pa.string()),
        pa.field("bad_distance",     pa.float64()),
        pa.field("bad_cutoff",       pa.float64()),
        pa.field("bad_offset",       pa.list_(pa.int64())),
        pa.field("bad_is_self_image", pa.bool_()),
        pa.field("n_bad_pairs",      pa.int64()),
    ]
    return pa.schema(list(input_schema) + extra)


def _rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    if not rows:
        return schema.empty_table()
    return pa.Table.from_pylist(rows, schema=schema)


def filter_parquet(
    in_paths,
    out_path: str,
    rejected_path: str | None = None,
    scale: float = 0.5,
    batch_size: int = 1000,
    log_every: int = 10000,
    verbose: bool = False,
    backend: str = "auto",
    columns: dict | None = None,
    compression: str = "snappy",
    quiet: bool = False,
    positions_format: str = "cartesian",
) -> dict:
    cols = dict(DEFAULT_COLUMNS)
    if columns:
        cols.update(columns)

    mdm = build_min_distance_matrix(scale=scale)

    # resolve inputs single file, directory, glob, or list
    files = _resolve_inputs(in_paths)
    if not quiet:
        print(f"Reading {len(files)} parquet file{'s' if len(files) > 1 else ''}: "
              f"{files[0]}{' ...' if len(files) > 1 else ''}", flush=True)
    dataset = pa_ds.dataset(files, format="parquet")
    in_schema = dataset.schema

    # common ColabFit-schema confusions
    have = set(in_schema.names)
    missing = [c for k, c in cols.items() if c not in have and k != "pbc"]
    if missing:
        # detect ColabFit datasets metadata table (has dataset-level summary
        meta_signals = {"nconfigurations", "authors", "doi",
                        "publication_year", "energy_mean", "nelements"}
        if len(meta_signals & have) >= 3 and "positions" not in have:
            raise ValueError(
                "Input looks like the ColabFit *datasets metadata* table "
                "(columns include nconfigurations / authors / doi / etc.), "
                "not a configurations table. The filter needs the per-structure "
                "data. Look for a sibling parquet file/directory named "
                "'configurations', 'pos', or similar that contains "
                "atomic_numbers + positions + cell + pbc."
            )
        # detect property_objects table (has energy/forces but no structures)
        prop_signals = {"energy", "forces", "stress", "configuration_id",
                        "property_id"}
        if len(prop_signals & have) >= 2 and "positions" not in have:
            raise ValueError(
                "Input looks like a ColabFit *property_objects* table "
                "(has energy / forces / configuration_id but no positions). "
                "The filter needs the configurations table, which holds the "
                "atomic structures. Property objects link to configurations "
                "by ID; filter the configurations file, then join afterwards "
                "if you need per-structure properties."
            )
        # generic case
        raise ValueError(
            f"input parquet missing columns: {missing}. "
            f"Available: {sorted(have)}. Pass columns={{...}} to override names."
        )

    rej_schema = _make_rejected_schema(in_schema) if rejected_path else None

    passed = rejected = 0
    n_total = 0
    n_skipped = 0
    pair_counter: Counter = Counter()
    error_counter: Counter = Counter()

    import time
    t0 = time.time()

    out_writer = _LazyParquetWriter(out_path, in_schema, compression=compression)
    rej_writer = (_LazyParquetWriter(rejected_path, rej_schema, compression=compression)
                  if rejected_path else None)

    try:
        scanner = dataset.scanner(batch_size=batch_size)
        for batch in scanner.to_batches():
            rows = batch.to_pylist()
            passed_rows = []
            rejected_rows = []
            for row in rows:
                n_total += 1

                # per-row error isolation a single bad row (NaN positions
                try:
                    atoms = _row_to_atoms(row, cols, positions_format)
                except Exception as e:
                    n_skipped += 1
                    error_counter[f"parse:{type(e).__name__}"] += 1
                    continue

                try:
                    i, j, d, S = find_close_contacts(atoms, mdm, backend=backend)
                except Exception as e:
                    n_skipped += 1
                    msg = str(e)
                    if "non-finite" in msg or "NaN" in msg or "nan" in msg:
                        error_counter["filter:non_finite_coords"] += 1
                    elif "exceeds matrix size" in msg or "max_z" in msg:
                        error_counter["filter:superheavy_Z"] += 1
                    elif "not invertible" in msg or "singular" in msg.lower():
                        error_counter["filter:singular_cell"] += 1
                    else:
                        error_counter[f"filter:{type(e).__name__}"] += 1
                    continue

                if i.size == 0:
                    passed_rows.append(row)
                    passed += 1
                else:
                    rejected += 1
                    k = int(np.argmin(d))
                    syms = atoms.get_chemical_symbols()
                    Zi = int(atoms.numbers[i[k]])
                    Zj = int(atoms.numbers[j[k]])
                    pair_key = tuple(sorted([syms[i[k]], syms[j[k]]]))
                    pair_counter[pair_key] += 1
                    offset = (int(S[k][0]), int(S[k][1]), int(S[k][2]))

                    if verbose:
                        tag = "self-image" if i[k] == j[k] else (
                            "cross-image" if offset != (0, 0, 0) else "in-cell")
                        print(f"  reject #{rejected:>7d} row={n_total:<8d} "
                              f"{syms[i[k]]:>2s}({int(i[k])})-"
                              f"{syms[j[k]]:>2s}({int(j[k])}) "
                              f"d={float(d[k]):.3f} cutoff={float(mdm[Zi,Zj]):.3f} "
                              f"S={offset} [{tag}]", flush=True)

                    if rej_writer is not None:
                        new_row = dict(row)
                        new_row["bad_i"] = int(i[k])
                        new_row["bad_j"] = int(j[k])
                        new_row["bad_elements"] = f"{syms[i[k]]}-{syms[j[k]]}"
                        new_row["bad_distance"] = float(d[k])
                        new_row["bad_cutoff"] = float(mdm[Zi, Zj])
                        new_row["bad_offset"] = [int(s) for s in S[k]]
                        new_row["bad_is_self_image"] = bool(i[k] == j[k])
                        new_row["n_bad_pairs"] = int(i.size)
                        rejected_rows.append(new_row)

            if passed_rows:
                out_writer.write_table(_rows_to_table(passed_rows, in_schema))
            if rejected_rows and rej_writer is not None:
                rej_writer.write_table(_rows_to_table(rejected_rows, rej_schema))

            if log_every and n_total // log_every > (n_total - len(rows)) // log_every:
                rate = n_total / max(time.time() - t0, 1e-9)
                top = pair_counter.most_common(3)
                top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top)
                print(f"[{n_total:>10d}] {rate:7.0f} struct/s | "
                      f"passed={passed} rejected={rejected}"
                      + (f" | top: {top_str}" if top else ""), flush=True)
    finally:
        out_writer.close()
        if rej_writer is not None:
            rej_writer.close()

    if rejected > 0 and not quiet:
        print(f"\nRejection causes by element pair "
              f"(unique pairs: {len(pair_counter)}):")
        for (a, b), c in pair_counter.most_common():
            pct = 100.0 * c / rejected
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")
    if n_skipped > 0 and not quiet:
        print(f"\nSkipped {n_skipped:,} rows (per-row errors, not rejections):")
        for cat, c in error_counter.most_common():
            pct = 100.0 * c / n_skipped
            print(f"  {cat:<28s}  {c:>8,d}  ({pct:5.1f}%)")

    return {
        "passed":           passed,
        "rejected":         rejected,
        "skipped":          n_skipped,
        "total":            passed + rejected + n_skipped,
        "rejection_pairs":  pair_counter,
        "error_counter":    error_counter,
    }


# parallel driver (one worker per input shard)
def _shard_worker(args):
    shard_path, out_file, rej_file, kwargs = args
    return filter_parquet(
        [shard_path], out_file,
        rejected_path=rej_file,
        log_every=0,
        quiet=True,
        **kwargs,
    )


def filter_parquet_parallel(
    in_paths,
    out_dir: str,
    rejected_dir: str | None = None,
    n_workers: int = 4,
    scale: float = 0.5,
    batch_size: int = 1000,
    backend: str = "auto",
    columns: dict | None = None,
    compression: str = "snappy",
    positions_format: str = "cartesian",
) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    files = _resolve_inputs(in_paths)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    rej_dir_p = None
    if rejected_dir is not None:
        rej_dir_p = Path(rejected_dir)
        rej_dir_p.mkdir(parents=True, exist_ok=True)

    # one task per shard. Skip shards whose output already exists (idempotent
    tasks = []
    skipped = 0
    for shard in files:
        stem = Path(shard).stem
        out_file = str(out_dir_p / f"passed_{stem}.parquet")
        rej_file = str(rej_dir_p / f"rejected_{stem}.parquet") if rej_dir_p else None
        if Path(out_file).exists():
            skipped += 1
            continue
        kwargs = dict(
            scale=scale, batch_size=batch_size, backend=backend,
            columns=columns, compression=compression,
            positions_format=positions_format,
        )
        tasks.append((shard, out_file, rej_file, kwargs))

    n = len(tasks)
    print(f"Processing {n} shard{'s' if n != 1 else ''} across {n_workers} "
          f"worker{'s' if n_workers != 1 else ''}"
          + (f" ({skipped} already done, skipping)" if skipped else ""),
          flush=True)

    t0 = time.time()
    per_shard = []
    failed_shards: list[tuple[str, str]] = []   # (path, error_message)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_shard_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            shard_path = futs[fut][0]
            try:
                stats = fut.result()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"  [FAIL] {Path(shard_path).name}: {err}", flush=True)
                failed_shards.append((shard_path, err))
                continue
            per_shard.append((shard_path, stats))
            completed += 1
            top = stats["rejection_pairs"].most_common(2)
            top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top) or "-"
            rate_total = sum(s[1]["total"] for s in per_shard) / max(time.time() - t0, 1e-9)

            # flag silent skipping prominently if more rows were skipped
            n_skip = stats.get("skipped", 0)
            warn = ""
            if n_skip > stats["passed"] and n_skip > 0:
                err_cat = ""
                if stats.get("error_counter"):
                    top_err = stats["error_counter"].most_common(1)
                    if top_err:
                        err_cat = f" [top: {top_err[0][0]}]"
                warn = f"  !! skipped={n_skip}{err_cat} !!"

            print(f"  [{completed:>3d}/{n}] {Path(shard_path).name:<32s}  "
                  f"passed={stats['passed']:>7d} rejected={stats['rejected']:>5d}  "
                  f"top: {top_str}  ({rate_total:.0f} struct/s overall){warn}",
                  flush=True)

    # aggregate
    total_passed   = sum(s["passed"]   for _, s in per_shard)
    total_rejected = sum(s["rejected"] for _, s in per_shard)
    total_skipped  = sum(s.get("skipped", 0) for _, s in per_shard)
    combined: Counter = Counter()
    combined_errors: Counter = Counter()
    for _, s in per_shard:
        combined.update(s["rejection_pairs"])
        combined_errors.update(s.get("error_counter", Counter()))

    if total_rejected > 0:
        print(f"\nCombined rejection causes by element pair "
              f"(unique pairs: {len(combined)}):")
        for (a, b), c in combined.most_common():
            pct = 100.0 * c / total_rejected
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")

    if total_skipped > 0:
        print(f"\nCombined per-row errors: {total_skipped:,} rows skipped")
        for cat, c in combined_errors.most_common():
            pct = 100.0 * c / total_skipped
            print(f"  {cat:<28s}  {c:>8,d}  ({pct:5.1f}%)")

    if failed_shards:
        print(f"\n!!! WARNING: {len(failed_shards)} of {n} shards FAILED "
              f"and their data was LOST !!!", flush=True)
        for path, err in failed_shards:
            print(f"  {Path(path).name}: {err}", flush=True)
        print(f"  -> dataset is INCOMPLETE; rerun with fewer workers or "
              f"investigate worker deaths (likely OOM)", flush=True)

    return {
        "passed":          total_passed,
        "rejected":        total_rejected,
        "skipped":         total_skipped,
        "total":           total_passed + total_rejected + total_skipped,
        "rejection_pairs": combined,
        "error_counter":   combined_errors,
        "per_shard":       [(p, s) for p, s in per_shard],
        "failed_shards":   failed_shards,
    }


# CLI
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Filter a ColabFit-style parquet dataset using Pyykko "
                    "covalent-radius distance cutoffs.",
        epilog="Inputs can be one or more parquet files, a directory of "
               "parquet files (e.g. <dataset>/co/), or a shell glob.",
    )
    p.add_argument("inputs", nargs="+",
                   help="Input parquet file(s), directory, or glob pattern.")
    p.add_argument("-o", "--output", required=True,
                   help="Output parquet for passing structures.")
    p.add_argument("--rejected", default=None,
                   help="Optional output parquet for rejected structures.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor (default 0.5).")
    p.add_argument("--batch-size", type=int, default=1000,
                   help="pyarrow batch size (default 1000).")
    p.add_argument("--verbose", action="store_true",
                   help="Print one line per rejection.")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto")
    p.add_argument("--col-numbers",   default="atomic_numbers")
    p.add_argument("--col-positions", default="positions")
    p.add_argument("--col-cell",      default="cell")
    p.add_argument("--col-pbc",       default="pbc")
    p.add_argument("--compression", default="snappy",
                   choices=["snappy", "zstd", "gzip", "none"])
    p.add_argument("--positions-format", default="cartesian",
                   choices=["cartesian", "fractional"],
                   help="How to interpret the positions column. Default "
                        "'cartesian' matches the extxyz spec / ColabFit "
                        "standard. Use 'fractional' for the few datasets "
                        "where positions are stored as fractional coords "
                        "in [0, 1] (e.g. some PWMLFF / VASP-derived sets).")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes for parallel shard "
                        "processing. With workers > 1, output paths must be "
                        "directories (one file per shard).")
    p.add_argument("--print-schema", action="store_true",
                   help="Print the input schema and exit (useful for "
                        "discovering column names).")
    args = p.parse_args()

    columns = {
        "numbers":   args.col_numbers,
        "positions": args.col_positions,
        "cell":      args.col_cell,
        "pbc":       args.col_pbc,
    }
    compression = None if args.compression == "none" else args.compression

    if args.print_schema:
        files = _resolve_inputs(args.inputs)
        ds = pa_ds.dataset(files, format="parquet")
        print(f"Resolved {len(files)} file(s); schema:")
        for f in ds.schema:
            print(f"  {f.name:<30s}  {f.type}")
        sys.exit(0)

    if args.workers > 1:
        stats = filter_parquet_parallel(
            args.inputs,
            out_dir=args.output,
            rejected_dir=args.rejected,
            n_workers=args.workers,
            scale=args.scale,
            batch_size=args.batch_size,
            backend=args.backend,
            columns=columns,
            compression=compression,
            positions_format=args.positions_format,
        )
    else:
        stats = filter_parquet(
            args.inputs, args.output,
            rejected_path=args.rejected,
            scale=args.scale,
            batch_size=args.batch_size,
            verbose=args.verbose,
            backend=args.backend,
            columns=columns,
            compression=compression,
            positions_format=args.positions_format,
        )
    print(f"Done: passed={stats['passed']:,} rejected={stats['rejected']:,} "
          f"skipped={stats.get('skipped', 0):,} total={stats['total']:,}")
