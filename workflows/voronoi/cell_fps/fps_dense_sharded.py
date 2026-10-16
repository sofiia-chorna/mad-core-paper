import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from coresets.cuda.extension import ensure_compiled, get_ext
from coresets.cuda.multi.layout import ShardLayout

REBUILD_CENTER_CHUNK = 32768
REBUILD_ROW_TILE = 8192


def discover_layout(X_local, device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    sizes_t = torch.zeros(world_size, dtype=torch.long, device=device)
    sizes_t[rank] = X_local.shape[0]
    dist.all_reduce(sizes_t, op=dist.ReduceOp.SUM)
    sizes = sizes_t.cpu().tolist()
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    return ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)


def rebuild_hausdorff(X_local, norms_local, selected_np, layout, device, verbose):
    n_shard, dim = X_local.shape
    hausdorff = torch.full((n_shard,), float("inf"), dtype=torch.float32, device=device)
    selected_t = torch.from_numpy(selected_np).to(device=device, dtype=torch.long)
    low = layout.shard_offset
    high = low + n_shard

    t0 = time.time()
    for start in range(0, len(selected_t), REBUILD_CENTER_CHUNK):
        centers = selected_t[start:start + REBUILD_CENTER_CHUNK]
        buffer = torch.zeros((len(centers), dim), dtype=torch.float32, device=device)
        mine = (centers >= low) & (centers < high)
        if bool(mine.any()):
            buffer[mine] = X_local[centers[mine] - low].to(torch.float32)
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)

        center_norms = (buffer * buffer).sum(dim=1)
        for row0 in range(0, n_shard, REBUILD_ROW_TILE):
            row1 = min(row0 + REBUILD_ROW_TILE, n_shard)
            base = norms_local[row0:row1].unsqueeze(1) + center_norms.unsqueeze(0)
            distances = torch.addmm(base, X_local[row0:row1], buffer.T,
                                    beta=1.0, alpha=-2.0)
            hausdorff[row0:row1] = torch.minimum(hausdorff[row0:row1],
                                                 distances.min(dim=1).values)

        if verbose and (start // REBUILD_CENTER_CHUNK) % 50 == 0:
            done = min(start + REBUILD_CENTER_CHUNK, len(selected_t))
            print(f"  rebuild {done:,}/{len(selected_t):,} centers "
                  f"({time.time() - t0:.0f}s)", flush=True)

    mine = (selected_t >= low) & (selected_t < high)
    if bool(mine.any()):
        hausdorff[selected_t[mine] - low] = -1.0

    if verbose:
        print(f"  rebuild done in {time.time() - t0:.0f}s", flush=True)
    return hausdorff


def write_checkpoint(path, step, selected, epsilons_np, hausdorff, rank):
    hausdorff_path = f"{path}_k{step:010d}_h{rank:03d}.npy"
    np.save(f"{hausdorff_path}.tmp", hausdorff.cpu().numpy())
    os.replace(f"{hausdorff_path}.tmp.npy", hausdorff_path)

    dist.barrier()

    if rank == 0:
        ids = selected[:step + 1].cpu().numpy()
        ids_path = f"{path}_k{step:010d}_ids.npy"
        eps_path = f"{path}_k{step:010d}_eps.npy"
        np.save(f"{ids_path}.tmp", ids)
        os.replace(f"{ids_path}.tmp.npy", ids_path)
        np.save(f"{eps_path}.tmp", epsilons_np[:step + 1])
        os.replace(f"{eps_path}.tmp.npy", eps_path)
        with open(f"{path}_latest.txt.tmp", "w") as fh:
            fh.write(str(step))
        os.replace(f"{path}_latest.txt.tmp", f"{path}_latest.txt")


def load_resume(prefix):
    with open(f"{prefix}_latest.txt") as fh:
        step = int(fh.read().strip())
    ids = np.load(f"{prefix}_k{step:010d}_ids.npy")
    return step, ids


def fps_dense_sharded(X_local, n_to_select, *, seed=None, checkpoint_path=None,
                      checkpoint_every=0, resume_from=None, log_every=None,
                      verbose=True, max_picks=None):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialised before fps_dense_sharded()")

    ensure_compiled()
    ext = get_ext()

    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = X_local.device
    X_local = X_local.contiguous()
    layout = discover_layout(X_local, device)
    n_global = sum(layout.sizes)
    dim = X_local.shape[1]

    if not 1 <= n_to_select <= n_global:
        raise ValueError(f"n_to_select={n_to_select} outside [1, {n_global}]")
    if n_global >= 2 ** 31:
        raise ValueError(f"N_global={n_global} exceeds the int32 pack range")

    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    n_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    stride = 3 + dim
    pack = torch.empty(stride, dtype=torch.float32, device=device)
    gather_buf = torch.empty(world_size * stride, dtype=torch.float32, device=device)

    voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)
    selected = torch.empty(n_to_select, dtype=torch.long, device=device)
    epsilons_np = np.empty(n_to_select, dtype=np.float32)
    epsilons_np[0] = float("inf")

    def commit(new_global_idx, sel_pos, winner_rank):
        owner = layout.owner_of(new_global_idx)
        if rank == owner:
            local = layout.to_local(new_global_idx)
            hausdorff[local] = -1.0
            voronoi_shard[local] = sel_pos
        if layout.N_shard == 0:
            return
        base = winner_rank * stride
        ext.fps_update_cuda(X_local, norms_local, gather_buf[base + 3: base + 3 + dim],
                            hausdorff, voronoi_shard,
                            gather_buf[base + 2: base + 3], sel_pos)

    def pack_row_by_owner(global_idx, owner_rank):
        as_int = pack.view(torch.int32)
        if rank == owner_rank:
            local = layout.to_local(global_idx)
            pack[0] = 0.0
            as_int[1] = int(global_idx)
            pack[2] = norms_local[local].to(torch.float32)
            pack[3: 3 + dim] = X_local[local].to(torch.float32)
        else:
            pack[0] = float("-inf")
            as_int[1] = -1
            pack[2] = 0.0
            pack[3: 3 + dim] = 0.0
        dist.all_gather_into_tensor(gather_buf, pack)

    if resume_from is not None and os.path.exists(f"{resume_from}_latest.txt"):
        step, selected_prev = load_resume(resume_from)
        n_prev = step + 1
        if n_prev > n_to_select:
            raise ValueError(f"checkpoint has {n_prev} picks > n_to_select={n_to_select}")
        selected[:n_prev] = torch.from_numpy(selected_prev).to(device=device,
                                                              dtype=torch.long)
        eps_file = f"{resume_from}_k{step:010d}_eps.npy"
        if os.path.exists(eps_file):
            loaded = np.load(eps_file).astype(np.float32, copy=False)
            epsilons_np[:loaded.shape[0]] = loaded

        if n_prev == n_to_select:
            if verbose and rank == 0:
                print(f"fps_dense_sharded: checkpoint already at K={n_prev:,}")
            return selected, torch.from_numpy(epsilons_np)

        # a snapshot is only reusable if EVERY rank has one matching its shard
        hausdorff_file = f"{resume_from}_k{step:010d}_h{rank:03d}.npy"
        reuse = os.path.exists(hausdorff_file)
        if reuse:
            reuse = np.load(hausdorff_file, mmap_mode="r").shape[0] == layout.N_shard
        reuse_all = torch.tensor([1 if reuse else 0], dtype=torch.long, device=device)
        dist.all_reduce(reuse_all, op=dist.ReduceOp.MIN)

        if int(reuse_all.item()) == 1:
            hausdorff = torch.from_numpy(np.load(hausdorff_file)).to(device)
            if verbose and rank == 0:
                print(f"fps_dense_sharded: resumed k={n_prev:,} from hausdorff snapshot",
                      flush=True)
        else:
            if verbose and rank == 0:
                print(f"fps_dense_sharded: resumed k={n_prev:,}, rebuilding hausdorff "
                      f"over {n_prev:,} centers", flush=True)
            hausdorff = rebuild_hausdorff(X_local, norms_local, selected_prev, layout,
                                          device, verbose and rank == 0)
        start_i = n_prev
    else:
        hausdorff = torch.full((layout.N_shard,), float("inf"), dtype=torch.float32,
                               device=device)
        first = 0 if seed is None else int(seed)
        if not 0 <= first < n_global:
            raise ValueError(f"seed={first} outside [0, {n_global})")
        selected[0] = first
        owner = layout.owner_of(first)
        pack_row_by_owner(first, owner)
        commit(first, 0, owner)
        if verbose and rank == 0:
            print(f"fps_dense_sharded: N={n_global:,} D={dim} K={n_to_select:,} "
                  f"world={world_size} seed={first}", flush=True)
        start_i = 1

    if checkpoint_path is not None:
        if checkpoint_every > 0:
            steps = set(range(checkpoint_every, n_to_select, checkpoint_every))
        else:
            max_pow = int(math.log2(n_to_select)) if n_to_select > 1 else 0
            steps = {1 << k for k in range(1, max_pow + 1)}
        steps.add(n_to_select - 1)
        checkpoint_steps = {s for s in steps if s < n_to_select}
    else:
        checkpoint_steps = set()

    if log_every is None:
        log_every = max(1, n_to_select // 100)

    stop_at = n_to_select if max_picks is None else min(n_to_select, start_i + max_picks)
    t0 = time.time()

    for i in range(start_i, stop_at):
        if layout.N_shard > 0:
            ext.argmax_cuda(hausdorff, block_vals, block_idxs, d_out_val, d_out_idx,
                            layout.N_shard)
            ext.pack_argmax_with_row_cuda(X_local, norms_local, d_out_val, d_out_idx,
                                          int(layout.shard_offset), pack)
        else:
            ext.fill_pack_with_row_neg_inf_cuda(pack, int(dim))
        dist.all_gather_into_tensor(gather_buf, pack)

        gathered = gather_buf.view(world_size, stride)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        new_idx = int(gathered.view(torch.int32)[winner, 1].item())

        selected[i] = new_idx
        epsilons_np[i] = best_val
        commit(new_idx, i, winner)

        if verbose and rank == 0 and (i % log_every == 0 or i == stop_at - 1):
            elapsed = time.time() - t0
            rate = (i - start_i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  {i:>10,}/{n_to_select:,}  ({rate:>7.1f} pick/s, "
                  f"eps={math.sqrt(max(best_val, 0.0)):.4e}, cum {elapsed:.1f}s)",
                  flush=True)

        if i in checkpoint_steps and checkpoint_path is not None:
            write_checkpoint(checkpoint_path, i, selected, epsilons_np, hausdorff, rank)

    if stop_at < n_to_select:
        return selected[:stop_at], torch.from_numpy(epsilons_np[:stop_at])
    return selected, torch.from_numpy(epsilons_np)
