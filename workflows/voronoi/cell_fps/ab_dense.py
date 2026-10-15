import argparse
import time

import numpy as np
import torch

from coresets.cuda.extension import ensure_compiled, get_ext
from run_cell_fps import CORESET, N_CENTERS, SCALER, kernel_fps, load_bucket


def kernel_fps_dense(device, ext, x_np, center_feat, center_pos):
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
            ext.fps_update_cuda(x, norms, x_sel[slot], h, voronoi,
                                norms_sel[slot:slot + 1], slot)

    return selected.cpu().numpy().astype(np.int64), eps.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--target-n", type=int, default=1_000_000)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    ensure_compiled()
    ext = get_ext()

    scaler = np.load(SCALER)
    mean = scaler["mean"].astype(np.float32)
    inv_scale = np.float32(1.0 / float(scaler["scale"]))

    table, cells, d24, cid, features = load_bucket(args.bucket_dir)
    unique, counts = np.unique(cells, return_counts=True)
    cell = int(unique[np.argmin(np.abs(counts - args.target_n))])
    rows = np.where(cells == cell)[0]
    n = len(rows)
    print(f"cell {cell}: n={n:,}", flush=True)

    center_cid = str(np.load(f"{CORESET}/configuration_id.npy",
                             mmap_mode="r")[cell].astype("U"))
    center_feat = np.array(np.memmap(f"{CORESET}/ecumetric_features.bin",
                                     dtype=np.float32, mode="r",
                                     shape=(N_CENTERS, 512))[cell])
    center_feat -= mean
    center_feat *= inv_scale
    match = np.nonzero(cid[rows] == center_cid)[0]
    center_pos = int(match[0]) if len(match) else -1

    x_np = features[rows]

    t0 = time.time()
    sel_skip, eps_skip = kernel_fps(device, ext, x_np, center_feat, center_pos)
    torch.cuda.synchronize()
    t_skip = time.time() - t0
    print(f"voronoi-skip: {t_skip:.1f}s  ({n / t_skip:,.0f} picks/s)", flush=True)

    t0 = time.time()
    sel_dense, eps_dense = kernel_fps_dense(device, ext, x_np, center_feat, center_pos)
    torch.cuda.synchronize()
    t_dense = time.time() - t0
    print(f"dense:        {t_dense:.1f}s  ({n / t_dense:,.0f} picks/s)", flush=True)

    identical_sel = bool(np.array_equal(sel_skip, sel_dense))
    identical_eps = bool(np.array_equal(eps_skip, eps_dense))
    print(f"orders identical: {identical_sel}  eps identical: {identical_eps}")
    print(f"speedup skip/dense: {t_dense / t_skip:.2f}x")


if __name__ == "__main__":
    main()
