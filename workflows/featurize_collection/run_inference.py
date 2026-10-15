import argparse
import ast
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import metatomic.torch as mta
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from ase import Atoms
from metatrain.utils.io import load_model as load_metatrain_model

try:
    import vesin.metatomic as vesin_metatomic
except ModuleNotFoundError:
    vesin_metatomic = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--meanstd", action="store_true",
                   help="store concat([mean, std]) of per-atom features per structure (2x feature dim)")
    p.add_argument("--save-atomic", default=None,
                   help="also save raw per-atom features (concat over all structures, in row order) to this .npy")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--chunk-rows", type=int, default=50_000,
                   help="streaming write chunk size (rows per row group)")
    p.add_argument("--keep-cols", default=None,
                   help="comma-separated INPUT columns to keep in the output (default: keep all). "
                        "The 3 phace_* columns (energy, features, predicted forces) are ALWAYS added. "
                        "Use a lean id set (e.g. configuration_id,structure_hash,dataset_id,names,nsites) "
                        "to drop heavy per-atom labels/geometry from the feature file.")
    p.add_argument("--keep-cols-missing-ok", action="store_true",
                   help="if a --keep-cols column is absent from the input schema, skip it (logged) "
                        "instead of failing (needed for heterogeneous sources, e.g. mcloudarchive).")
    p.add_argument("--max-nsites", type=int, default=None)
    p.add_argument("--nl-backend", choices=["metatomic", "ase"], default="ase")
    return p.parse_args()


def _as_numbers(v):
    """Some colabfit exports (Materials_Project, Alexandria_*_PBE_3D) store
    atomic_numbers as a STRING repr "[11, 11, 6, ...]" (arrow type string)
    instead of list<int64>. Parse defensively so those datasets can featurize."""
    if isinstance(v, str):
        return [int(x) for x in ast.literal_eval(v)]
    return v


def systems_from_table(table):
    positions = table.column("positions")
    cell = table.column("cell")
    pbc = table.column("pbc")
    numbers = table.column("atomic_numbers")
    atoms = [
        Atoms(
            numbers=_as_numbers(numbers[i].as_py()),
            positions=positions[i].as_py(),
            cell=cell[i].as_py(),
            pbc=pbc[i].as_py(),
        )
        for i in range(table.num_rows)
    ]
    for a in atoms:
        a.wrap(pbc=a.cell.any(axis=1))
    return mta.systems_to_torch(atoms, dtype=torch.float64)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_phace_columns(energies, feats_mean, forces_flat, natoms_per_row, pa_dtype):

    n = energies.shape[0]
    D = feats_mean.shape[1]

    energy_arr = pa.array(energies, type=pa_dtype)

    # phace_features_bb_mean: list<f> with every row of length D
    feat_values = pa.array(feats_mean.reshape(-1), type=pa_dtype)
    feat_offsets = pa.array(np.arange(n + 1, dtype=np.int32) * D, type=pa.int32())
    feat_arr = pa.ListArray.from_arrays(feat_offsets, feat_values)

    # phace_non_conservative_forces: list<list<f>>, outer = natoms, inner = 3
    natoms = np.asarray(natoms_per_row, dtype=np.int64)
    total_atoms = int(natoms.sum())

    # inner: every triplet is one (x,y,z); inner_offsets has length total_atoms+1
    inner_values = pa.array(forces_flat.reshape(-1), type=pa_dtype)
    inner_offsets = pa.array(np.arange(total_atoms + 1, dtype=np.int32) * 3, type=pa.int32())
    inner_arr = pa.ListArray.from_arrays(inner_offsets, inner_values)

    # outer: row i has natoms[i] triplets
    outer_offsets = pa.array(
        np.concatenate(([0], np.cumsum(natoms))).astype(np.int32),
        type=pa.int32(),
    )
    forces_arr = pa.ListArray.from_arrays(outer_offsets, inner_arr)

    return energy_arr, feat_arr, forces_arr


def build_context(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
        gpu = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        gpu = "(none)"

    torch_dtype = getattr(torch, args.dtype)
    np_dtype = np.float64 if args.dtype == "float64" else np.float32
    pa_dtype = pa.float64() if args.dtype == "float64" else pa.float32()

    log(f"loading model: {args.model}")
    model = load_metatrain_model(args.model).to(device)
    length_unit = model.capabilities().length_unit

    if args.nl_backend == "metatomic":
        nl_calculators = vesin_metatomic.neighbor_lists_for_model(length_unit, model)
        requested_nls = None
    else:
        # ase backend: route NL through vesin.ase_neighbor_list (correct on vesin 0.5.3)
        import vesin
        import nl_ase
        nl_calculators = None
        requested_nls = model.requested_neighbor_lists()
        log(f"nl-backend=ase  ({len(requested_nls)} requested NL options)  vesin={vesin.__version__}")

    options = mta.ModelEvaluationOptions(
        length_unit=length_unit,
        outputs={
            "energy": mta.ModelOutput(per_atom=False),
            "non_conservative_forces": mta.ModelOutput(per_atom=True),
            "features": mta.ModelOutput(per_atom=True),
        },
        selected_atoms=None,
    )
    log(f"host={socket.gethostname()} device={device} gpu={gpu} dtype={args.dtype} model ready")
    return {
        "device": device, "gpu": gpu, "torch_dtype": torch_dtype,
        "np_dtype": np_dtype, "pa_dtype": pa_dtype, "model": model,
        "options": options, "nl_calculators": nl_calculators, "requested_nls": requested_nls,
    }


def featurize_file(args, ctx, parquet, output, batch_size):
    out_path = Path(output)
    sidecar_path = out_path.with_suffix(".json")

    if out_path.exists():
        log(f"output exists, skipping: {out_path}")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    uniq = f"{socket.gethostname()}.{os.getpid()}"
    tmp_out = out_path.with_suffix(out_path.suffix + f".tmp.{uniq}")
    tmp_sidecar = sidecar_path.with_suffix(f".json.tmp.{uniq}")

    device = ctx["device"]; gpu = ctx["gpu"]; dtype = ctx["torch_dtype"]
    np_dtype = ctx["np_dtype"]; pa_dtype = ctx["pa_dtype"]; model = ctx["model"]
    options = ctx["options"]; nl_calculators = ctx["nl_calculators"]; requested_nls = ctx["requested_nls"]

    log(f"src: {parquet}")
    log(f"out: {out_path}  batch={batch_size}")

    pf = pq.ParquetFile(parquet)
    n_total = pf.metadata.num_rows
    log(f"source: {n_total} rows in {pf.num_row_groups} row group(s)")

    # Output schema = kept input columns + 3 phace columns (built once, used for the writer)
    src_schema = pf.schema_arrow
    keep_cols = None
    if args.keep_cols:
        keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
        missing = [c for c in keep_cols if c not in src_schema.names]
        if missing:
            if args.keep_cols_missing_ok:
                log(f"keep-cols: {len(missing)} absent from input, skipping: {missing}")
                keep_cols = [c for c in keep_cols if c in src_schema.names]
            else:
                raise SystemExit(f"--keep-cols not in input schema: {missing}")
        log(f"keep-cols: output carries {keep_cols} + phace_* (dropping {len(src_schema.names) - len(keep_cols)} input cols)")
    kept_fields = [src_schema.field(c) for c in keep_cols] if keep_cols else list(src_schema)
    out_schema = pa.schema(
        kept_fields
        + [
            pa.field("phace_energy", pa_dtype),
            pa.field("phace_features_bb_mean", pa.list_(pa_dtype)),
            pa.field("phace_non_conservative_forces", pa.list_(pa.list_(pa_dtype))),
        ]
    )

    writer = None
    n_processed = 0
    n_src_total = 0
    n_dropped_nsites = 0
    t_inf_total = 0.0
    t_build_total = 0.0
    t_write_total = 0.0
    atomic_chunks = []  # per-atom features accumulated across chunks (for --save-atomic)

    try:
        for chunk_idx, src_batch in enumerate(pf.iter_batches(batch_size=args.chunk_rows)):
            src_table = pa.Table.from_batches([src_batch])
            n_src_total += src_table.num_rows

            # row pre-filter (before inference): drop oversized structures
            if args.max_nsites is not None and "nsites" in src_table.schema.names:
                before = src_table.num_rows
                src_table = src_table.filter(pc.less_equal(src_table.column("nsites"), args.max_nsites))
                n_dropped_nsites += before - src_table.num_rows
            n_chunk = src_table.num_rows
            if n_chunk == 0:
                continue

            t = time.time()
            systems = systems_from_table(src_table)
            t_build_total += time.time() - t

            # ---- per-chunk inference accumulators ----
            energies = np.empty((n_chunk,), dtype=np_dtype)
            feats_mean = None  # allocated on first batch once we know D
            natoms_per_row = np.empty((n_chunk,), dtype=np.int64)
            forces_chunks = []  # list of (natoms_in_batch, 3) numpy arrays

            t_inf = time.time()
            for s_idx in range(0, n_chunk, batch_size):
                e_idx = min(s_idx + batch_size, n_chunk)
                batch = systems[s_idx:e_idx]
                if nl_calculators is not None:
                    for calc in nl_calculators:
                        calc.add_neighbor_list(batch)
                else:
                    batch = [nl_ase.get_system_with_neighbor_lists(s, requested_nls) for s in batch]
                batch = [s.to(dtype=dtype, device=device) for s in batch]

                with torch.jit.optimized_execution(False), torch.no_grad():
                    out = model(batch, options, check_consistency=False)

                e = out["energy"].block(0).values.detach().cpu().numpy().flatten()
                f = out["non_conservative_forces"].block(0).values.squeeze(-1).detach().cpu().numpy()
                feats = out["features"].block(0).values.detach().cpu().numpy()

                if feats_mean is None:
                    d_out = feats.shape[1] * 2 if args.meanstd else feats.shape[1]
                    feats_mean = np.empty((n_chunk, d_out), dtype=np_dtype)

                offset = 0
                for j, s in enumerate(batch):
                    natoms = int(s.positions.shape[0])
                    natoms_per_row[s_idx + j] = natoms
                    blk = feats[offset:offset + natoms]
                    feats_mean[s_idx + j] = (np.concatenate([blk.mean(axis=0), blk.std(axis=0)]) if args.meanstd else blk.mean(axis=0))
                    offset += natoms
                if args.save_atomic:
                    atomic_chunks.append(feats.astype(np_dtype, copy=True))
                forces_chunks.append(f.astype(np_dtype, copy=False))
                energies[s_idx:e_idx] = e

                del out, e, f, feats
                if device.startswith("cuda") and (s_idx // batch_size) % 20 == 0:
                    torch.cuda.empty_cache()

                if (s_idx // batch_size) % args.log_every == 0:
                    elapsed = time.time() - t_inf
                    rate = e_idx / elapsed if elapsed else 0.0
                    cum_done = n_processed + e_idx
                    log(f"  chunk {chunk_idx} {e_idx}/{n_chunk}  cum {cum_done}/{n_total}  ({rate:.1f} struct/s)")

            t_inf_total += time.time() - t_inf

            # ---- build chunk's phace columns + write ----
            t = time.time()
            forces_flat = np.concatenate(forces_chunks, axis=0)
            del forces_chunks
            energy_arr, feat_arr, forces_arr = build_phace_columns(
                energies, feats_mean, forces_flat, natoms_per_row, pa_dtype
            )
            out_base = src_table.select(keep_cols) if keep_cols else src_table
            out_table = (
                out_base
                .append_column("phace_energy", energy_arr)
                .append_column("phace_features_bb_mean", feat_arr)
                .append_column("phace_non_conservative_forces", forces_arr)
            )

            if writer is None:
                writer = pq.ParquetWriter(tmp_out, out_schema, compression="snappy")
            writer.write_table(out_table)
            t_write_total += time.time() - t

            n_processed += n_chunk
            del out_table, energy_arr, feat_arr, forces_arr, energies, feats_mean, natoms_per_row, forces_flat, systems, src_table
            log(f"chunk {chunk_idx}: wrote {n_chunk} rows  ({n_processed}/{n_total} done)")

        if args.save_atomic:
            atomic = np.concatenate(atomic_chunks, axis=0)
            np.save(args.save_atomic, atomic)
            log(f"saved atomic features -> {args.save_atomic}  shape={atomic.shape}")

    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if tmp_out.exists():
            tmp_out.unlink()
        raise

    if writer is not None:
        writer.close()
        os.replace(tmp_out, out_path)
    else:  # zero input rows -> write an empty valid file (no writer was ever opened)
        pq.write_table(out_schema.empty_table(), tmp_out, compression="snappy")
        os.replace(tmp_out, out_path)

    rate = n_processed / t_inf_total if t_inf_total else 0.0
    log(f"inference: {n_processed} structs in {t_inf_total:.1f}s  ({rate:.1f} struct/s)")
    log(f"wrote {out_path} (rows={n_processed}) write_time={t_write_total:.1f}s")

    meta = {
        "src": str(parquet),
        "rows": int(n_processed),
        "n_src_rows": int(n_src_total),
        "n_dropped_nsites": int(n_dropped_nsites),
        "max_nsites": args.max_nsites,
        "struct_per_s": round(rate, 2),
        "elapsed_s": {
            "build_systems": round(t_build_total, 3),
            "inference": round(t_inf_total, 3),
            "write": round(t_write_total, 3),
        },
        "host": socket.gethostname(),
        "gpu": gpu,
        "dtype": args.dtype,
        "chunk_rows": args.chunk_rows,
        "model": str(args.model),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp_sidecar, sidecar_path)
    log(f"sidecar: {sidecar_path}")
    return meta


def main():
    args = parse_args()
    if Path(args.output).exists():
        log(f"output exists, skipping: {args.output}")
        return 0
    ctx = build_context(args)
    featurize_file(args, ctx, args.parquet, args.output, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
