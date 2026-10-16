from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pyykko_filter import build_min_distance_matrix, find_close_contacts


_LATTICE_RE = re.compile(rb'Lattice=(?:"([^"]*)"|(\S+))')
_PBC_RE     = re.compile(rb'pbc=(?:"([^"]*)"|(\S+))')


def _extract_one_frame(buf: bytes, pos: int) -> tuple[int, int] | None:
    L = len(buf)
    if pos >= L:
        return None
    nl = buf.find(b"\n", pos)
    if nl == -1:
        return None
    try:
        n_atoms = int(buf[pos:nl].strip())
    except ValueError:
        # malformed count line surface as ValueError to the caller
        raise ValueError(
            f"expected integer atom count, got {buf[pos:nl][:40]!r} at byte {pos}"
        )
    cm_end = buf.find(b"\n", nl + 1)
    if cm_end == -1:
        return None
    cursor = cm_end + 1
    for _ in range(n_atoms):
        nxt = buf.find(b"\n", cursor)
        if nxt == -1:
            return None
        cursor = nxt + 1
    return (pos, cursor)


def _split_extxyz_frames(text: bytes) -> list[bytes]:
    frames: list[bytes] = []
    pos = 0
    while True:
        try:
            r = _extract_one_frame(text, pos)
        except ValueError:
            # truncation or malformed mid-stream to stop, return what we got
            break
        if r is None:
            break
        start, end = r
        frames.append(text[start:end])
        pos = end
    return frames


def _parse_frame_to_atoms(frame_bytes: bytes) -> Atoms:
    lines = frame_bytes.split(b"\n")
    n_atoms = int(lines[0].strip())
    comment = lines[1]

    cell = None
    m = _LATTICE_RE.search(comment)
    if m:
        lattice_str = (m.group(1) or m.group(2)).decode()
        cell = np.fromstring(lattice_str, sep=" ").reshape(3, 3)

    pbc = np.array([False, False, False])
    m = _PBC_RE.search(comment)
    if m:
        pbc_str = (m.group(1) or m.group(2)).decode().strip()
        pbc = np.array([c in ("T", "t", "1", "True", "true")
                        for c in pbc_str.split()])
        if pbc.size == 1:
            pbc = np.repeat(pbc, 3)
    elif cell is not None:
        # has cell but no explicit pbc default to periodic
        pbc = np.array([True, True, True])

    syms = []
    pos_arr = np.empty((n_atoms, 3), dtype=float)
    for k in range(n_atoms):
        toks = lines[2 + k].split()
        syms.append(toks[0].decode())
        pos_arr[k, 0] = float(toks[1])
        pos_arr[k, 1] = float(toks[2])
        pos_arr[k, 2] = float(toks[3])

    if cell is not None:
        return Atoms(symbols=syms, positions=pos_arr, cell=cell, pbc=pbc)
    return Atoms(symbols=syms, positions=pos_arr, pbc=pbc)


def _stream_frame_bytes(
    in_path: str, read_chunk_bytes: int = 64 * 1024 * 1024,
):
    with open(in_path, "rb") as fp:
        buf = b""
        pos = 0
        while True:
            chunk = fp.read(read_chunk_bytes)
            if not chunk:
                break
            # keep only the unread tail + new chunk
            if pos:
                buf = buf[pos:]
                pos = 0
            buf = buf + chunk
            # extract all complete frames
            while True:
                try:
                    r = _extract_one_frame(buf, pos)
                except ValueError:
                    return
                if r is None:
                    break
                start, end = r
                yield bytes(buf[start:end])
                pos = end
        # end of file drain whatever remains
        while True:
            try:
                r = _extract_one_frame(buf, pos)
            except ValueError:
                return
            if r is None:
                return
            start, end = r
            yield bytes(buf[start:end])
            pos = end


# worker
def _process_batch_pure(
    batch_idx: int,
    frames_bytes: bytes,
    scale: float,
    backend: str,
) -> dict:
    mdm = build_min_distance_matrix(scale=scale)
    frames = _split_extxyz_frames(frames_bytes)

    passed_chunks: list[bytes] = []
    rejected_chunks: list[bytes] = []
    pair_counter: Counter = Counter()
    skipped = 0

    for fr in frames:
        try:
            atoms = _parse_frame_to_atoms(fr)
        except Exception:
            skipped += 1
            continue

        i, j, d, S = find_close_contacts(atoms, mdm, backend=backend)
        if i.size == 0:
            passed_chunks.append(fr)
        else:
            rejected_chunks.append(fr)
            kk = int(np.argmin(d))
            syms = atoms.get_chemical_symbols()
            pair_counter[tuple(sorted([syms[i[kk]], syms[j[kk]]]))] += 1

    return {
        "batch_idx":       batch_idx,
        "passed_bytes":    b"".join(passed_chunks),
        "rejected_bytes":  b"".join(rejected_chunks),
        "passed_count":    len(passed_chunks),
        "rejected_count":  len(rejected_chunks),
        "skipped_count":   skipped,
        "pair_counter":    pair_counter,
    }


# resume scanner
def _count_frames(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return len(_split_extxyz_frames(data))
    except FileNotFoundError:
        return 0


def _scan_existing_parts(
    out_dir: Path, rej_dir: Path | None,
) -> tuple[int, int]:
    part_indices: set[int] = set()
    for d in (out_dir, rej_dir):
        if d is not None and d.is_dir():
            for p in d.glob("part_*.xyz"):
                try:
                    n = int(p.stem.split("_")[1])
                    part_indices.add(n)
                except (IndexError, ValueError):
                    continue
    if not part_indices:
        return 0, -1

    total = 0
    for n in sorted(part_indices):
        p_pass = out_dir / f"part_{n:04d}.xyz"
        n_passed = _count_frames(p_pass)
        n_rejected = 0
        if rej_dir is not None:
            n_rejected = _count_frames(rej_dir / f"part_{n:04d}.xyz")
        total += n_passed + n_rejected
    return total, max(part_indices)


# main parallel driver
def filter_extxyz_parallel(
    in_path: str,
    out_dir: str,
    rejected_dir: str | None = None,
    scale: float = 0.5,
    n_workers: int = 8,
    batch_size: int = 500,
    backend: str = "auto",
    log_every: int = 50000,
    fsync_every: int = 50,
    queue_size: int | None = None,
) -> dict:
    if queue_size is None:
        queue_size = 4 * n_workers

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    rej_dir_p: Path | None = None
    if rejected_dir is not None:
        rej_dir_p = Path(rejected_dir)
        rej_dir_p.mkdir(parents=True, exist_ok=True)

    skip_count, max_idx = _scan_existing_parts(out_dir_p, rej_dir_p)
    new_idx = max_idx + 1
    new_pass_path = out_dir_p / f"part_{new_idx:04d}.xyz"
    new_rej_path = (rej_dir_p / f"part_{new_idx:04d}.xyz") if rej_dir_p else None

    if skip_count > 0:
        print(f"Resuming: {skip_count:,} frames already done; new output -> "
              f"{new_pass_path.name}", flush=True)
    else:
        print(f"Fresh run: writing to {new_pass_path}", flush=True)

    in_size = Path(in_path).stat().st_size
    print(f"Opening {in_path} ({in_size / 1e9:.2f} GB) with {n_workers} "
          f"workers, batch_size={batch_size} ...", flush=True)

    pair_counter: Counter = Counter()
    passed = rejected = skipped = total = 0
    skipped_resumed = 0
    last_logged = 0
    failed_batches = 0
    batches_since_sync = 0
    next_batch_idx = 0
    t0 = time.time()

    fp_pass = open(new_pass_path, "wb")
    fp_rej = open(new_rej_path, "wb") if new_rej_path else None

    in_flight: deque = deque()

    def _fsync_outputs():
        try:
            fp_pass.flush()
            os.fsync(fp_pass.fileno())
            if fp_rej is not None:
                fp_rej.flush()
                os.fsync(fp_rej.fileno())
        except OSError:
            pass

    def _drain_one(blocking: bool):
        nonlocal passed, rejected, skipped, total, last_logged
        nonlocal failed_batches, batches_since_sync
        if not in_flight:
            return False
        fut = in_flight[0]
        if not blocking and not fut.done():
            return False
        in_flight.popleft()
        try:
            r = fut.result()
        except Exception as e:
            failed_batches += 1
            print(f"  WORKER FAIL: {type(e).__name__}: {e}", flush=True)
            return True

        passed   += r["passed_count"]
        rejected += r["rejected_count"]
        skipped  += r["skipped_count"]
        total    += r["passed_count"] + r["rejected_count"] + r["skipped_count"]
        pair_counter.update(r["pair_counter"])

        if r["passed_bytes"]:
            fp_pass.write(r["passed_bytes"])
        if fp_rej is not None and r["rejected_bytes"]:
            fp_rej.write(r["rejected_bytes"])

        batches_since_sync += 1
        if batches_since_sync >= fsync_every:
            batches_since_sync = 0
            _fsync_outputs()

        if log_every and total - last_logged >= log_every:
            last_logged = (total // log_every) * log_every
            elapsed = max(time.time() - t0, 1e-9)
            rate = total / elapsed
            top = pair_counter.most_common(3)
            top_str = ", ".join(f"{a}-{b}:{c}" for (a, b), c in top)
            print(f"[{total:>10d}] {rate:7.0f} frame/s | "
                  f"passed={passed:,} rejected={rejected:,} "
                  f"in_flight={len(in_flight)}"
                  + (f" | top: {top_str}" if top else ""),
                  flush=True)
        return True

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            batch_buffer: list[bytes] = []
            input_idx = 0

            for frame_bytes in _stream_frame_bytes(in_path):
                input_idx += 1
                if input_idx <= skip_count:
                    skipped_resumed += 1
                    continue

                batch_buffer.append(frame_bytes)
                if len(batch_buffer) >= batch_size:
                    fut = pool.submit(
                        _process_batch_pure,
                        next_batch_idx, b"".join(batch_buffer), scale, backend,
                    )
                    next_batch_idx += 1
                    in_flight.append(fut)
                    batch_buffer = []

                    while len(in_flight) >= queue_size:
                        _drain_one(blocking=True)

                while _drain_one(blocking=False):
                    pass

            # final partial batch
            if batch_buffer:
                fut = pool.submit(
                    _process_batch_pure,
                    next_batch_idx, b"".join(batch_buffer), scale, backend,
                )
                next_batch_idx += 1
                in_flight.append(fut)

            while in_flight:
                _drain_one(blocking=True)
    finally:
        _fsync_outputs()
        fp_pass.close()
        if fp_rej is not None:
            fp_rej.close()

    if rejected > 0:
        print(f"\nRejection causes by element pair "
              f"(unique pairs: {len(pair_counter)}):")
        for (a, b), c in pair_counter.most_common():
            pct = 100.0 * c / rejected
            print(f"  {a:>2s}-{b:<2s}  {c:>8d}  ({pct:5.1f}%)")
    if skipped > 0:
        print(f"Skipped {skipped} unparseable frames")
    if skipped_resumed > 0:
        print(f"Resumed: {skipped_resumed:,} input frames skipped (already done)")
    if failed_batches > 0:
        print(f"WARNING: {failed_batches} batches failed in workers")

    return {
        "passed":          passed,
        "rejected":        rejected,
        "skipped":         skipped,
        "total":           total,
        "rejection_pairs": pair_counter,
        "skipped_resumed": skipped_resumed,
        "failed_batches":  failed_batches,
        "out_part":        str(new_pass_path),
        "rejected_part":   str(new_rej_path) if new_rej_path else None,
    }


# directory driver iterate over input files, one parallel run per file
import csv
import json


def filter_extxyz_dir_parallel(
    in_dir: str,
    out_dir: str,
    rejected_dir: str | None = None,
    pattern: tuple[str, ...] = ("*.xyz", "*.extxyz"),
    scale: float = 0.5,
    n_workers: int = 8,
    batch_size: int = 500,
    backend: str = "auto",
    log_every: int = 50000,
    fsync_every: int = 50,
    only: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> dict:
    in_dir_p  = Path(in_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    rej_dir_p: Path | None = None
    if rejected_dir is not None:
        rej_dir_p = Path(rejected_dir)
        rej_dir_p.mkdir(parents=True, exist_ok=True)

    # discover input files matching any of the patterns
    files: list[Path] = []
    seen: set[Path] = set()
    for pat in pattern:
        for p in sorted(in_dir_p.glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            stem = p.stem
            if only and stem not in only:
                continue
            if stem in exclude:
                continue
            files.append(p)

    if not files:
        print(f"No matching files in {in_dir} with patterns {pattern}",
              flush=True)
        return {"files": [], "totals": {}}

    print(f"Found {len(files)} input files in {in_dir}", flush=True)

    audit_path = out_dir_p / "audit.csv"
    audit_rows: list[dict] = []

    # load existing audit.csv if present, so re-runs accumulate rather than
    if audit_path.exists():
        with open(audit_path) as f:
            audit_rows = list(csv.DictReader(f))
        # index by stem for quick update
        existing_stems = {r["stem"] for r in audit_rows}
    else:
        existing_stems = set()

    totals = {"passed": 0, "rejected": 0, "skipped": 0, "total": 0,
              "skipped_resumed": 0, "failed_batches": 0}

    for k, in_path in enumerate(files, start=1):
        stem = in_path.stem
        per_out_dir = out_dir_p / stem
        per_rej_dir = (rej_dir_p / stem) if rej_dir_p else None
        stats_path  = per_out_dir / "stats.json"

        # already done?
        if stats_path.exists():
            with open(stats_path) as f:
                prev = json.load(f)
            print(f"\n[{k}/{len(files)}] SKIP {stem} (already done): "
                  f"passed={prev.get('passed', '?')} "
                  f"rejected={prev.get('rejected', '?')}", flush=True)
            for kk in totals:
                totals[kk] += prev.get(kk, 0)
            continue

        print(f"\n[{k}/{len(files)}] {stem} ({in_path.stat().st_size / 1e9:.2f} GB)",
              flush=True)
        try:
            stats = filter_extxyz_parallel(
                str(in_path),
                out_dir=str(per_out_dir),
                rejected_dir=str(per_rej_dir) if per_rej_dir else None,
                scale=scale,
                n_workers=n_workers,
                batch_size=batch_size,
                backend=backend,
                log_every=log_every,
                fsync_every=fsync_every,
            )
            # strip non-serializable fields before writing stats.json
            stats_to_write = {kk: v for kk, v in stats.items()
                              if kk not in ("rejection_pairs",)}
            stats_to_write["rejection_pairs_top"] = [
                [a, b, c] for (a, b), c in
                stats["rejection_pairs"].most_common(20)
            ]
            with open(stats_path, "w") as f:
                json.dump(stats_to_write, f, indent=2)
            for kk in totals:
                totals[kk] += stats.get(kk, 0)
            row = {"stem": stem, "status": "ok",
                   "passed":   stats["passed"],
                   "rejected": stats["rejected"],
                   "skipped":  stats["skipped"],
                   "total":    stats["total"]}
        except Exception as e:
            print(f"  ERROR processing {stem}: {type(e).__name__}: {e}",
                  flush=True)
            row = {"stem": stem, "status": "error",
                   "error": f"{type(e).__name__}: {e}",
                   "passed": "", "rejected": "", "skipped": "", "total": ""}

        # update audit row for this stem (replace if existed, else append)
        audit_rows = [r for r in audit_rows if r.get("stem") != stem]
        audit_rows.append(row)
        # rewrite audit.csv after every file (durable across crashes)
        fieldnames = ["stem", "status", "passed", "rejected", "skipped",
                      "total", "error"]
        with open(audit_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in audit_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n=== Directory totals across {len(files)} files ===")
    for kk, vv in totals.items():
        print(f"  {kk}: {vv:,}")
    print(f"  audit: {audit_path}")
    return {"files": [p.stem for p in files], "totals": totals,
            "audit_csv": str(audit_path)}


# CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Parallelized covalent-radius distance filter for extxyz "
                    "shards. Output is a directory of session part files; "
                    "supports resume after kill -9."
    )
    p.add_argument("input", help="Input extxyz file (.xyz or .extxyz), OR "
                                  "a directory of such files (processed "
                                  "one at a time, one output subdir each).")
    p.add_argument("-o", "--output", required=True,
                   help="File mode: output directory for part_NNNN.xyz. "
                        "Directory mode: parent directory; per-input "
                        "subdirectories are created inside.")
    p.add_argument("--rejected", default=None,
                   help="Output directory for rejected frames. Required for "
                        "reliable resume.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Covalent-radius scale factor (default 0.5).")
    p.add_argument("--workers", type=int, default=8,
                   help="Worker processes (default 8).")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Frames per worker batch (default 500).")
    p.add_argument("--backend", choices=["auto", "vesin", "ase"], default="auto")
    p.add_argument("--fsync-every", type=int, default=50,
                   help="Flush + fsync output every N batches (default 50).")
    p.add_argument("--pattern", nargs="+", default=["*.xyz", "*.extxyz"],
                   help="Directory mode: glob patterns for input files "
                        "(default: *.xyz *.extxyz).")
    p.add_argument("--only", nargs="*", default=[],
                   help="Directory mode: only process input stems in this list.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Directory mode: skip input stems in this list.")
    args = p.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        # directory mode
        result = filter_extxyz_dir_parallel(
            str(in_path),
            out_dir=args.output,
            rejected_dir=args.rejected,
            pattern=tuple(args.pattern),
            scale=args.scale,
            n_workers=args.workers,
            batch_size=args.batch_size,
            backend=args.backend,
            fsync_every=args.fsync_every,
            only=tuple(args.only),
            exclude=tuple(args.exclude),
        )
        t = result["totals"]
        print(f"\nFinal: passed={t['passed']:,} rejected={t['rejected']:,} "
              f"skipped={t['skipped']:,} total={t['total']:,}")
    else:
        # single-file mode
        stats = filter_extxyz_parallel(
            args.input,
            out_dir=args.output,
            rejected_dir=args.rejected,
            scale=args.scale,
            n_workers=args.workers,
            batch_size=args.batch_size,
            backend=args.backend,
            fsync_every=args.fsync_every,
        )
        print(f"Done: passed={stats['passed']:,} rejected={stats['rejected']:,} "
              f"skipped={stats['skipped']:,} total={stats['total']:,}")
