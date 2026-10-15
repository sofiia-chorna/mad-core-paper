import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from coresets.cuda.extension import ensure_compiled, get_ext

RELEASE = "${STORE_ROOT}/mad-core/release"
SCALER = f"{RELEASE}/ecumetric_scaler.npz"
CORESET = f"{RELEASE}/coreset224"
N_CENTERS = 2 ** 24

SMALL_MAX = 512
BATCH_MAX_ROWS = 2_000_000
NEG_INF = float("-inf")

OUT_SCHEMA = pa.schema([
    ("cell", pa.int32()),
    ("cell_rank", pa.int32()),
    ("eps_sq", pa.float32()),
    ("configuration_id", pa.string()),
    ("dataset_group", pa.string()),
    ("dataset_id", pa.string()),
])


def load_bucket(part_dir):
    paths = sorted(glob.glob(f"{part_dir}/part_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parts in {part_dir}")
    table = pa.concat_tables(pq.read_table(p) for p in paths)

    cells = table.column("cell").to_numpy()
    d24 = table.column("d24").to_numpy()
    cid = np.asarray(table.column("configuration_id").to_pandas(), dtype=object)
    features = np.asarray(table.column("features").combine_chunks().flatten()
                          .to_numpy()).reshape(len(cells), 512)
    return table, cells, d24, cid, features


def batched_fps(device, x_pad, lengths, seeds, center_pos):
    n_batch, n_max, _ = x_pad.shape
    x = torch.from_numpy(x_pad).to(device)
    seed = torch.from_numpy(seeds).to(device)
    n_valid = torch.from_numpy(lengths).to(device)
    cpos = torch.from_numpy(center_pos).to(device)

    valid = torch.arange(n_max, device=device)[None, :] < n_valid[:, None]
    norms = (x * x).sum(-1)
    seed_norm = (seed * seed).sum(-1)

    h = norms + seed_norm[:, None] - 2.0 * torch.bmm(x, seed[:, :, None]).squeeze(-1)
    h.clamp_(min=0)
    h[~valid] = NEG_INF

    rows = torch.arange(n_batch, device=device)
    orders = torch.empty((n_batch, n_max), dtype=torch.int64, device=device)
    eps = torch.empty((n_batch, n_max), dtype=torch.float32, device=device)

    for step in range(n_max):
        if step == 0:
            pick = torch.where(cpos >= 0, cpos, h.argmax(dim=1))
        else:
            pick = h.argmax(dim=1)
        orders[:, step] = pick
        eps[:, step] = h[rows, pick]

        if step + 1 < n_max:
            x_new = x[rows, pick]
            d = norms + (x_new * x_new).sum(-1)[:, None] \
                - 2.0 * torch.bmm(x, x_new[:, :, None]).squeeze(-1)
            d.clamp_(min=0)
            h = torch.minimum(h, d)
        h[rows, pick] = NEG_INF

    return orders.cpu().numpy(), eps.cpu().numpy()


def kernel_fps(device, ext, x_np, center_feat, center_pos):
    n = len(x_np)
    x = torch.from_numpy(x_np).to(device)
    center = torch.from_numpy(center_feat).to(device)

    norms = (x * x).sum(dim=1)
    center_norm = (center * center).sum()
    h = norms + center_norm - 2.0 * (x @ center)
    h.clamp_(min=0)

    x_sel = torch.empty((n + 1, 512), dtype=torch.float32, device=device)
    norms_sel = torch.empty(n + 1, dtype=torch.float32, device=device)
    x_sel[0] = center
    norms_sel[0] = center_norm

    voronoi = torch.zeros(n, dtype=torch.int64, device=device)
    dsl = torch.empty(n + 1, dtype=torch.float32, device=device)
    active = torch.zeros(n + 1, dtype=torch.int32, device=device)

    n_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    selected = torch.empty(n, dtype=torch.int32, device=device)
    eps = torch.empty(n, dtype=torch.float32, device=device)

    for i in range(n):
        slot = i + 1
        if i == 0 and center_pos >= 0:
            d_out_idx.fill_(center_pos)
            eps[0] = h[center_pos]
        else:
            ext.argmax_cuda(h, block_vals, block_idxs, d_out_val, d_out_idx, n)
            eps[i] = d_out_val[0]
        selected[i] = d_out_idx[0]

        ext.commit_center_cuda(x, norms, x_sel, norms_sel, h, voronoi, d_out_idx, slot)

        if i + 1 < n:
            active[:slot].zero_()
            ext.mark_active_centers_cuda(h, voronoi, active, n)
            ext.compute_dSL_quarter_masked_cuda(x_sel, norms_sel, dsl, active, slot)
            ext.voronoi_update_cuda(x, norms, x_sel[slot], dsl[:slot], h, voronoi,
                                    norms_sel[slot:slot + 1], slot)

    return selected.cpu().numpy().astype(np.int64), eps.cpu().numpy()


def flush_small_batch(device, entries, features, rank_out, eps_out):
    lengths = np.array([len(entry["rows"]) for entry in entries], dtype=np.int64)
    n_batch, n_max = len(entries), int(lengths.max())

    rows_concat = np.concatenate([entry["rows"] for entry in entries])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    within = np.arange(len(rows_concat), dtype=np.int64) - np.repeat(offsets, lengths)
    slot = np.repeat(np.arange(n_batch, dtype=np.int64), lengths)

    x_pad = np.zeros((n_batch, n_max, 512), dtype=np.float32)
    x_pad.reshape(-1, 512)[slot * n_max + within] = features[rows_concat]

    row_pad = np.full((n_batch, n_max), -1, dtype=np.int64)
    row_pad.reshape(-1)[slot * n_max + within] = rows_concat

    seeds = np.stack([entry["center_feat"] for entry in entries])
    center_pos = np.array([entry["center_pos"] for entry in entries], dtype=np.int64)

    orders, eps = batched_fps(device, x_pad, lengths, seeds, center_pos)

    steps = np.broadcast_to(np.arange(n_max, dtype=np.int64)[None, :], orders.shape)
    valid = steps < lengths[:, None]
    picked_rows = np.take_along_axis(row_pad, orders, axis=1)
    rank_out[picked_rows[valid]] = steps[valid].astype(np.int32)
    eps_out[picked_rows[valid]] = eps[valid]


def process_rows(device, ext, cells, cid, features, mean, inv_scale, timings):
    n_rows = len(cells)
    rank_out = np.full(n_rows, -1, dtype=np.int32)
    eps_out = np.full(n_rows, np.nan, dtype=np.float32)

    order = np.argsort(cells, kind="stable")
    sorted_cells = cells[order]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_cells) != 0)[0] + 1,
                                 [n_rows]])
    unique_cells = sorted_cells[boundaries[:-1]].astype(np.int64)

    center_cid = np.load(f"{CORESET}/configuration_id.npy",
                         mmap_mode="r")[unique_cells].astype("U")
    center_feats = np.array(np.memmap(f"{CORESET}/ecumetric_features.bin",
                                      dtype=np.float32, mode="r",
                                      shape=(N_CENTERS, 512))[unique_cells])
    center_feats -= mean
    center_feats *= inv_scale

    cell_center_pos = np.full(len(unique_cells), -1, dtype=np.int64)
    cell_infos = []
    for k in range(len(unique_cells)):
        rows = order[boundaries[k]:boundaries[k + 1]]
        match = np.nonzero(cid[rows] == center_cid[k])[0]
        if len(match):
            cell_center_pos[k] = match[0]
        cell_infos.append((len(rows), k, rows))
    cell_infos.sort(key=lambda item: item[0])

    batch = []

    def flush():
        nonlocal batch
        if batch:
            flush_small_batch(device, batch, features, rank_out, eps_out)
        batch = []

    for n, k, rows in cell_infos:
        if n <= SMALL_MAX:
            if batch and ((len(batch) + 1) * n > BATCH_MAX_ROWS
                          or n > 2 * max(len(batch[0]["rows"]), 32)):
                flush()
            batch.append({"rows": rows, "center_feat": center_feats[k],
                          "center_pos": cell_center_pos[k]})
        else:
            t0 = time.time()
            sel, eps = kernel_fps(device, ext, features[rows], center_feats[k],
                                  int(cell_center_pos[k]))
            rank_out[rows[sel]] = np.arange(n, dtype=np.int32)
            eps_out[rows[sel]] = eps
            timings.append({"cell": int(unique_cells[k]), "n": int(n),
                            "seconds": round(time.time() - t0, 3)})
    flush()

    return rank_out, eps_out, unique_cells, boundaries, order, cell_center_pos


def verify(d24, rank_out, eps_out, unique_cells, boundaries, order, cell_center_pos):
    problems = {"unranked_rows": int((rank_out < 0).sum()),
                "nan_eps_rows": int(np.isnan(eps_out).sum()),
                "eps_increase_cells": 0, "radius_mismatch_cells": 0,
                "max_radius_rel_diff": 0.0,
                "centers_present": int((cell_center_pos >= 0).sum()),
                "center_not_rank0_cells": 0}

    for k in range(len(unique_cells)):
        rows = order[boundaries[k]:boundaries[k + 1]]
        ranks = rank_out[rows]
        eps = np.empty(len(rows), dtype=np.float32)
        eps[ranks] = eps_out[rows]

        if len(rows) > 2 and np.any(np.diff(eps[1:]) > 1e-5 * np.abs(eps[1:-1]) + 1e-7):
            problems["eps_increase_cells"] += 1

        if cell_center_pos[k] >= 0:
            if ranks[cell_center_pos[k]] != 0:
                problems["center_not_rank0_cells"] += 1
            if len(rows) > 1:
                radius_sq = float(np.max(d24[rows].astype(np.float64) ** 2))
                if radius_sq > 1e-4:
                    difference = abs(float(eps[1]) - radius_sq)
                    rel = difference / radius_sq
                    problems["max_radius_rel_diff"] = max(problems["max_radius_rel_diff"],
                                                          round(rel, 6))
                    # both sides are f32 squared distances reached by different
                    # routes; near-degenerate cells (radius ~1e-2) carry ~1e-5 of
                    # absolute noise, which is >5% relative at that scale.
                    if rel > 0.05 and difference > 1e-4:
                        problems["radius_mismatch_cells"] += 1

    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    name = os.path.basename(os.path.normpath(args.bucket_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    done_marker = f"{args.out_dir}/{name}.done"
    if os.path.exists(done_marker):
        print(f"{name}: already done")
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    ensure_compiled()
    ext = get_ext()

    scaler = np.load(SCALER)
    mean = scaler["mean"].astype(np.float32)
    inv_scale = np.float32(1.0 / float(scaler["scale"]))

    t0 = time.time()
    table, cells, d24, cid, features = load_bucket(args.bucket_dir)
    assert not np.isnan(features).any(), "NaN features in bucket"
    t_load = time.time() - t0
    print(f"{name}: {len(cells):,} rows loaded in {t_load:.0f}s", flush=True)

    timings = []
    t0 = time.time()
    rank_out, eps_out, unique_cells, boundaries, order, cell_center_pos = process_rows(
        device, ext, cells, cid, features, mean, inv_scale, timings)
    t_fps = time.time() - t0

    problems = verify(d24, rank_out, eps_out, unique_cells, boundaries, order,
                      cell_center_pos)

    final = np.lexsort((rank_out, cells))
    out_table = pa.table({
        "cell": pa.array(cells[final]),
        "cell_rank": pa.array(rank_out[final]),
        "eps_sq": pa.array(eps_out[final]),
        "configuration_id": table.column("configuration_id").take(final),
        "dataset_group": table.column("dataset_group").take(final),
        "dataset_id": table.column("dataset_id").take(final),
    }, schema=OUT_SCHEMA)
    pq.write_table(out_table, f"{args.out_dir}/{name}.parquet")

    summary = {"bucket": name, "n_rows": len(cells), "n_cells": len(unique_cells),
               "load_seconds": round(t_load, 1), "fps_seconds": round(t_fps, 1),
               "problems": problems, "large_cell_timings": timings}
    with open(f"{args.out_dir}/{name}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(done_marker, "w") as fh:
        fh.write("done\n")

    print(f"{name}: DONE rows {len(cells):,} cells {len(unique_cells):,} "
          f"fps {t_fps:.0f}s problems {problems}", flush=True)


if __name__ == "__main__":
    main()
