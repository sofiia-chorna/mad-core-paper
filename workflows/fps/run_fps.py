from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import socket
import sys
import time
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from coresets import fps_sharded


HERE = Path(__file__).resolve().parent
ID_COLS = ["configuration_id", "dataset_id", "names"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="text file with one feature parquet path per line")
    p.add_argument("--scaler", required=True)
    p.add_argument("--K", type=int, default=None,
                   help="number of structures to select, omit for a full FPS over the whole pool")
    p.add_argument("--output", required=True, help="output parquet path")
    p.add_argument("--max-force", type=float, default=None,
                   help="if set, drop pool rows with max_force >= this (and NaN max_force). "
                        "Requires a max_force column in every shard (e.g. zbl_filtering output).")
    p.add_argument("--max-energy-per-atom", type=float, default=None,
                   help="if set, drop pool rows with zbl_energy_per_atom > this (and NaN). "
                        "Requires a zbl_energy_per_atom column in every shard.")
    p.add_argument("--max-atomic-energy", type=float, default=None,
                   help="if set, drop pool rows with zbl_max_atomic_energy > this (and NaN). "
                        "Requires a zbl_max_atomic_energy column in every shard.")
    p.add_argument("--exclude-cids", default=None,
                   help="text file with configuration_ids (one per line) to drop from the pool. "
                        "Use when the ZBL energy columns aren't in the shards but the garbage cids "
                        "are known (counted with the energy drops).")
    p.add_argument("--feature-col", default="phace_features_bb_mean",
                   help="name of the feature column in each shard parquet")
    p.add_argument("--seed", type=int, default=0,
                   help="global row index of the first FPS pick")
    p.add_argument("--checkpoint-dir", default=None, help="dir for FPS checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=0, help="0 = log2 cadence only")
    p.add_argument("--resume-from", default=None,
                   help="prefix of a previous run's checkpoint files (same string as that run's "
                        "--checkpoint-dir + cookie). Reads {prefix}_latest.txt and the matching "
                        "_ids.npy / _eps.npy; rebuilds per-rank FPS state by batched matrix "
                        "distance; then continues to the new --K.")
    p.add_argument("--timings-csv", default=str(HERE / "logs" / "timings.csv"), help="CSV log")
    p.add_argument("--notes", default="", help="free-text note column in timings.csv")
    p.add_argument("--emit-features", action="store_true",
                   help="ALSO write per-rank with_features/part_<rank>.parquet "
                        "(configuration_id, dataset_id, phace_features_bb_mean RAW) for the selected "
                        "rows. Makes the coreset self-contained so the accrete reads features directly "
                        "(no gather / pool re-scan). The lean fps.parquet is written unchanged.")
    return p.parse_args()


def load_scaler(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        mean_ = np.asarray(obj["mean_"], dtype=np.float32)
        scale_ = float(obj["scale_"])

    else:
        mean_ = np.asarray(obj.mean_, dtype=np.float32)
        scale_ = float(obj.scale_)

    return {"mean_": mean_, "scale_": scale_}


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    return ((X - scaler["mean_"]) / scaler["scale_"]).astype(np.float32)


def get_rank_world():
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


def init_dist(rank, world_size):
    import torch.distributed as dist
    if "MASTER_ADDR" not in os.environ:
        node = os.environ.get("SLURM_NODELIST", "localhost").split(",")[0]

        if "[" in node:
            base, suf = node.split("[", 1)
            node = base + suf.split("-")[0].split(",")[0].rstrip("]")

        os.environ["MASTER_ADDR"] = node

    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=3))


def log(msg, rank=None):
    ts = datetime.now().strftime("%H:%M:%S")
    pre = f"[{ts}]" + (f"[r{rank}]" if rank is not None else "")
    print(f"{pre} {msg}", flush=True)


def _load_features_arrow(feats_col):
    chunks = feats_col.combine_chunks()
    flat = chunks.values.to_numpy(zero_copy_only=False)
    n = len(chunks)
    d = flat.size // n
    return flat.reshape(n, d).astype(np.float32, copy=False)


def load_shard(path: str, scaler: dict, max_force: float = None,
               max_energy_per_atom: float = None, max_atomic_energy: float = None,
               exclude_cids: set = None, keep_raw: bool = False,
               feature_col: str = "phace_features_bb_mean"):
    names = pq.ParquetFile(path).schema_arrow.names
    cols = [feature_col] + [c for c in ID_COLS if c in names]
    force_col = "zbl_max_force" if "zbl_max_force" in names else ("max_force" if "max_force" in names else None)
    if max_force is not None:
        if force_col is None:
            raise ValueError(f"--max-force set but no zbl_max_force/max_force column in {path}")
        cols.append(force_col)
    if max_energy_per_atom is not None:
        if "zbl_energy_per_atom" not in names:
            raise ValueError(f"--max-energy-per-atom set but no zbl_energy_per_atom column in {path}")
        cols.append("zbl_energy_per_atom")
    if max_atomic_energy is not None:
        if "zbl_max_atomic_energy" not in names:
            raise ValueError(f"--max-atomic-energy set but no zbl_max_atomic_energy column in {path}")
        cols.append("zbl_max_atomic_energy")
    t = pq.read_table(path, columns=cols)

    n = t.num_rows
    if n == 0:
        return np.empty((0, 0), np.float32), {c: [] for c in ID_COLS}, 0, 0, 0, 0, None

    X = _load_features_arrow(t.column(feature_col))   # RAW features (kept if keep_raw)
    X_scaled = apply_scaler(X, scaler)
    ids = {c: (t.column(c).to_pylist() if c in t.column_names else [None] * n) for c in ID_COLS}

    finite = np.isfinite(X_scaled).all(axis=1)
    n_nonfinite = int((~finite).sum())

    force_mask = np.ones(n, dtype=bool)
    if max_force is not None:
        force_mask = t.column(force_col).to_numpy() < max_force  # NaN -> False

    energy_mask = np.ones(n, dtype=bool)
    if max_energy_per_atom is not None:
        energy_mask &= t.column("zbl_energy_per_atom").to_numpy() <= max_energy_per_atom  # NaN -> False
    if max_atomic_energy is not None:
        energy_mask &= t.column("zbl_max_atomic_energy").to_numpy() <= max_atomic_energy  # NaN -> False
    if exclude_cids:
        excluded = np.fromiter((c in exclude_cids for c in ids["configuration_id"]), dtype=bool, count=n)
        energy_mask &= ~excluded

    keep_mask = finite & force_mask & energy_mask
    n_maxforce = int((finite & ~force_mask).sum())            # disjoint: force drops among finite rows
    n_energy = int((finite & force_mask & ~energy_mask).sum())  # disjoint: energy drops the others kept

    if n_nonfinite or n_maxforce or n_energy:
        keep = np.flatnonzero(keep_mask)
        X_scaled = X_scaled[keep]
        if keep_raw:
            X = X[keep]
        ids = {c: [ids[c][i] for i in keep] for c in ID_COLS}

    return X_scaled, ids, X_scaled.shape[0], n_nonfinite, n_maxforce, n_energy, (X if keep_raw else None)


def main():
    args = parse_args()

    rank, world_size = get_rank_world()
    is_rank0 = (rank == 0)
    init_dist(rank, world_size)
    if world_size > 1:
        import torch.distributed as dist

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)
        device = f"cuda:{torch.cuda.current_device()}"
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        device = "cpu"
        gpu_name = "(none)"

    if is_rank0:
        log(f"world_size={world_size}  device={device}  gpu={gpu_name}")
        log(f"manifest: {args.manifest}")
        log(f"K={args.K if args.K is not None else 'full'}  seed={args.seed}  output={args.output}")

    with open(args.manifest) as f:
        all_shards = [ln.strip() for ln in f if ln.strip()]
    my_shards = all_shards[rank::world_size]
    log(f"got {len(my_shards)} of {len(all_shards)} shards", rank)

    t0 = time.time()
    scaler = load_scaler(args.scaler)
    t_xform_load = time.time() - t0
    if is_rank0:
        log(f"loaded scaler in {t_xform_load:.2f}s  (mean_norm={np.linalg.norm(scaler['mean_']):.4f}, scale_={scaler['scale_']:.4f})")

    exclude_cids = None
    if args.exclude_cids:
        with open(args.exclude_cids) as f:
            exclude_cids = {ln.strip() for ln in f if ln.strip()}
        if is_rank0:
            log(f"exclude-cids: {len(exclude_cids):,} cids loaded from {args.exclude_cids}")

    t0 = time.time()
    Xs, raw_Xs, id_tables = [], [], {c: [] for c in ID_COLS}
    n_local = 0
    n_nf_local = 0
    n_mf_local = 0
    n_en_local = 0
    feat_dim = None
    for sh in my_shards:
        X_scaled, ids, n, n_nf, n_mf, n_en, X_raw = load_shard(
            sh, scaler, args.max_force, args.max_energy_per_atom, args.max_atomic_energy,
            exclude_cids, keep_raw=args.emit_features, feature_col=args.feature_col)

        n_local += n
        n_nf_local += n_nf
        n_mf_local += n_mf
        n_en_local += n_en

        if n == 0:
            continue

        Xs.append(X_scaled)
        if args.emit_features:
            raw_Xs.append(X_raw)

        for c in ID_COLS:
            id_tables[c].extend(ids[c])

        if feat_dim is None:
            feat_dim = X_scaled.shape[1]

    X_local = np.concatenate(Xs, axis=0) if Xs else np.empty((0, feat_dim or 0), dtype=np.float32)
    raw_X_local = (np.concatenate(raw_Xs, axis=0) if raw_Xs else
                   np.empty((0, feat_dim or 0), dtype=np.float32)) if args.emit_features else None
    del raw_Xs
    t_load_pca = time.time() - t0
    log(f"local: rows={n_local}  dropped(nonfinite={n_nf_local}, maxforce>={args.max_force}={n_mf_local}, "
        f"energy={n_en_local})  d={X_local.shape[1]}  t_load_scale={t_load_pca:.1f}s", rank)

    t0 = time.time()
    X_local_t = torch.from_numpy(X_local).to(device, non_blocking=False)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t_to_gpu = time.time() - t0

    if world_size > 1:
        n_total_t = torch.tensor([n_local], dtype=torch.int64, device=device)
        dist.all_reduce(n_total_t)
        n_total = int(n_total_t.item())
        sizes_t = torch.zeros(world_size, dtype=torch.int64, device=device)
        sizes_t[rank] = n_local
        dist.all_reduce(sizes_t)
        sizes = sizes_t.cpu().tolist()
    else:
        n_total = n_local
        sizes = [n_local]

    offsets = np.cumsum([0] + sizes[:-1])
    my_offset = int(offsets[rank])

    if world_size > 1:
        nd_t = torch.tensor([n_nf_local, n_mf_local, n_en_local], dtype=torch.int64, device=device)
        dist.all_reduce(nd_t)
        n_nonfinite_total, n_maxforce_total, n_energy_total = (
            int(nd_t[0].item()), int(nd_t[1].item()), int(nd_t[2].item()))
    else:
        n_nonfinite_total, n_maxforce_total, n_energy_total = n_nf_local, n_mf_local, n_en_local

    if args.K is None:
        args.K = n_total
        if is_rank0:
            log(f"--K not given -> full FPS over all {n_total:,} structures")

    if is_rank0:
        log(f"global n_structures={n_total:,}  K={args.K:,}")
        if n_nonfinite_total or n_maxforce_total or n_energy_total:
            energy_desc = (f"exclude-cids list ({args.exclude_cids})" if args.exclude_cids
                           else f"zbl_energy_per_atom>{args.max_energy_per_atom} or "
                                f"zbl_max_atomic_energy>{args.max_atomic_energy}")
            log(f"DROPPED {n_nonfinite_total:,} non-finite-feature rows + "
                f"{n_maxforce_total:,} rows with max_force>={args.max_force} + "
                f"{n_energy_total:,} rows by {energy_desc}")

    if args.K > n_total:
        if is_rank0:
            log(f"WARNING: K ({args.K:,}) > available structures ({n_total:,}), clamping K to {n_total:,}")
        args.K = n_total

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    selected, epsilons = fps_sharded(
        X_local_t,
        n_to_select=args.K,
        seed=args.seed,
        checkpoint_path=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        verbose=is_rank0,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t_fps = time.time() - t0
    if is_rank0:
        log(f"FPS done: K={args.K:,} in {t_fps:.1f}s ({args.K/max(t_fps,1e-9):.1f} selects/s)")

    sel_np = selected.detach().cpu().numpy().astype(np.int64)
    eps_np = epsilons.detach().cpu().numpy().astype(np.float32) if isinstance(epsilons, torch.Tensor) else np.asarray(epsilons, dtype=np.float32)
    mask = (sel_np >= my_offset) & (sel_np < my_offset + n_local)
    my_pos_in_sel = np.where(mask)[0] # positions in the global selected[] array
    my_local_idx = sel_np[mask] - my_offset # local row indices in this rank's X

    part = {
        "fps_order": my_pos_in_sel.astype(np.int64),
        "epsilon_sq": eps_np[my_pos_in_sel],
        "input_row": sel_np[mask].astype(np.int64), # global row in manifest concat
    }

    for c in ID_COLS:
        part[c] = [id_tables[c][i] for i in my_local_idx]

    # self-contained coreset features each rank writes its own selected rows' RAW features
    if args.emit_features:
        wf_dir = Path(args.output).parent / "with_features"
        wf_dir.mkdir(parents=True, exist_ok=True)
        raw_sel = raw_X_local[my_local_idx]
        d_feat = raw_sel.shape[1] if raw_sel.size else (feat_dim or 0)
        feat_arr = pa.ListArray.from_arrays(
            pa.array(np.arange(len(raw_sel) + 1, dtype=np.int32) * d_feat, type=pa.int32()),
            pa.array(raw_sel.reshape(-1), type=pa.float32()))
        wf_table = pa.table({
            "fps_order": pa.array(my_pos_in_sel.astype(np.int64), pa.int64()),
            "configuration_id": pa.array(part["configuration_id"], pa.string()),
            "dataset_id": pa.array(part["dataset_id"], pa.string()),
            args.feature_col: feat_arr,
        })
        wf_path = wf_dir / f"part_{rank:04d}.parquet"
        wf_tmp = wf_path.with_suffix(".parquet.tmp")
        pq.write_table(wf_table, wf_tmp, compression="snappy")
        os.replace(wf_tmp, wf_path)
        log(f"wrote {wf_path} ({wf_table.num_rows} rows + raw features)", rank)

    if world_size > 1:
        gathered = [None] * world_size if is_rank0 else None
        dist.gather_object(part, gathered, dst=0)
    else:
        gathered = [part]

    if not is_rank0:
        if world_size > 1:
            dist.destroy_process_group()
        return 0

    t0 = time.time()
    fps_order_all = np.concatenate([p["fps_order"] for p in gathered])
    order = np.argsort(fps_order_all)
    fps_order_out = fps_order_all[order]
    eps_out = np.concatenate([p["epsilon_sq"] for p in gathered])[order]
    input_row_out = np.concatenate([p["input_row"]  for p in gathered])[order]
    cols_out = {c: [] for c in ID_COLS}

    for p in gathered:
        for c in ID_COLS:
            cols_out[c].extend(p[c])

    for c in ID_COLS:
        cols_out[c] = [cols_out[c][i] for i in order]

    arrays = {
        "fps_order":          pa.array(fps_order_out.tolist(), type=pa.int64()),
        "epsilon_sq":         pa.array(eps_out.tolist(), type=pa.float32()),
        "input_row":          pa.array(input_row_out.tolist(), type=pa.int64()),
        "configuration_id":   pa.array(cols_out["configuration_id"], type=pa.string()),
        "dataset_id":         pa.array(cols_out["dataset_id"], type=pa.string()),
        "names":              pa.array(cols_out["names"], type=pa.list_(pa.string())),
    }
    out_table = pa.table(arrays)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(out_table, tmp, compression="snappy")
    os.replace(tmp, out_path)
    t_save = time.time() - t0
    log(f"wrote {out_path} (rows={len(fps_order_out)}) in {t_save:.1f}s")

    sibling_manifest = out_path.parent / "manifest.txt"

    try:
        if Path(args.manifest).resolve() != sibling_manifest.resolve():
            shutil.copyfile(args.manifest, sibling_manifest)

        log(f"manifest copied -> {sibling_manifest}")
    except OSError as e:
        log(f"WARN: could not copy manifest next to result: {e}")

    meta = {
        "K": int(args.K),
        "n_structures": int(n_total),
        "n_dropped_nonfinite": int(n_nonfinite_total),
        "n_dropped_maxforce": int(n_maxforce_total),
        "n_dropped_energy": int(n_energy_total),
        "max_force": args.max_force,
        "max_energy_per_atom": args.max_energy_per_atom,
        "max_atomic_energy": args.max_atomic_energy,
        "exclude_cids_file": str(args.exclude_cids) if args.exclude_cids else None,
        "n_exclude_cids": len(exclude_cids) if exclude_cids else 0,
        "d": int(X_local.shape[1]),
        "feature_col": str(args.feature_col),
        "world_size": int(world_size),
        "seed": int(args.seed),
        "scaler_path": str(args.scaler),
        "manifest": str(sibling_manifest),
        "manifest_relative": "manifest.txt",
        "manifest_source": str(args.manifest),
        "n_shards": len(all_shards),
        "output": str(out_path),
        "host": socket.gethostname(),
        "gpu": gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timings_s": {
            "scaler_load": round(t_xform_load, 3),
            "shard_load_scale": round(t_load_pca, 3),
            "to_gpu": round(t_to_gpu, 3),
            "fps": round(t_fps, 3),
            "save": round(t_save, 3),
        },
    }
    if device.startswith("cuda"):
        meta["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    meta["peak_cpu_mem_gb"] = round(psutil.Process().memory_info().rss / 1e9, 3)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp_meta, meta_path)
    log(f"wrote {meta_path}")

    csv_path = Path(args.timings_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "job_id", "timestamp_utc", "manifest", "n_shards", "n_structures", "K",
        "d", "world_size",
        "t_scaler_load_s", "t_load_scale_s", "t_to_gpu_s", "t_fps_s", "t_save_s", "t_total_s",
        "fps_selects_per_s",
        "peak_gpu_mem_gb", "peak_cpu_mem_gb",
        "output", "success", "notes",
    ]

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)

        t_total = t_xform_load + t_load_pca + t_to_gpu + t_fps + t_save
        w.writerow([
            os.environ.get("SLURM_JOB_ID", ""),
            meta["completed_at_utc"],
            args.manifest,
            len(all_shards),
            n_total,
            args.K,
            X_local.shape[1], world_size,
            f"{t_xform_load:.3f}", f"{t_load_pca:.3f}", f"{t_to_gpu:.3f}",
            f"{t_fps:.3f}", f"{t_save:.3f}", f"{t_total:.3f}",
            f"{args.K/max(t_fps,1e-9):.2f}",
            f"{meta.get('peak_gpu_mem_gb', 0):.3f}",
            f"{meta['peak_cpu_mem_gb']:.3f}",
            str(out_path), "yes", args.notes,
        ])
    log(f"appended timings to {csv_path}")

    if world_size > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
