import pickle

import numpy as np
import torch


def load_scaler(path):
    with open(path, "rb") as fh:
        scaler = pickle.load(fh)
    return np.asarray(scaler["mean_"], dtype=np.float32), float(scaler["scale_"])


def scale_features(x, mean, scale):
    return (x - torch.as_tensor(mean, device=x.device)) / scale


def evaluate_distances(feats, X, point_index, center_index, chunk=4_000_000):
    out = torch.empty(point_index.shape[0], dtype=torch.float32, device=X.device)
    for lo in range(0, point_index.shape[0], chunk):
        hi = min(lo + chunk, point_index.shape[0])
        diff = feats[center_index[lo:hi]] - X[point_index[lo:hi]]
        out[lo:hi] = diff.pow(2).sum(1).sqrt_()
    return out


def first_per_point(point_index, order_value, tie_value):
    order = torch.argsort(tie_value, stable=True)
    order = order[torch.argsort(order_value[order], stable=True)]
    order = order[torch.argsort(point_index[order], stable=True)]
    pt_sorted = point_index[order]
    is_first = torch.ones_like(pt_sorted, dtype=torch.bool)
    is_first[1:] = pt_sorted[1:] != pt_sorted[:-1]
    return order[is_first]


def descend(X, net, max_level, beam_width=512, record_from=0):
    device = X.device
    feats = net["feats"]
    suffix = net["suffix"]
    num_points = X.shape[0]
    num_recorded = max_level - record_from + 1

    beam_ids = torch.full((num_points, beam_width), -1, dtype=torch.int64, device=device)
    beam_d = torch.full((num_points, beam_width), float("inf"), dtype=torch.float32, device=device)
    d0 = (X - feats[0]).pow(2).sum(1).sqrt_()
    beam_ids[:, 0] = 0
    beam_d[:, 0] = d0
    best_d = d0.clone()
    best_id = torch.zeros(num_points, dtype=torch.int64, device=device)
    overflow = torch.zeros(num_points, dtype=torch.bool, device=device)

    assigns = torch.empty((num_points, num_recorded), dtype=torch.int64, device=device)
    dists = torch.empty((num_points, num_recorded), dtype=torch.float32, device=device)
    if record_from == 0:
        assigns[:, 0] = 0
        dists[:, 0] = d0

    all_points = torch.arange(num_points, device=device)
    for level in range(1, max_level + 1):
        csr = net["levels"][level]
        indptr, child, child_dp = csr["indptr"], csr["child"], csr["dp"]
        slack = float(suffix[level + 1])

        valid = beam_ids >= 0
        parent_id = beam_ids[valid]
        parent_d = beam_d[valid]
        parent_pt = all_points.unsqueeze(1).expand_as(beam_ids)[valid]
        start = indptr[parent_id]
        count = indptr[parent_id + 1] - start
        total = int(count.sum())

        cand_pt = torch.empty(0, dtype=torch.int64, device=device)
        cand_id = torch.empty(0, dtype=torch.int64, device=device)
        cand_d = torch.empty(0, dtype=torch.float32, device=device)
        if total > 0:
            offset = torch.arange(total, device=device) - (count.cumsum(0) - count).repeat_interleave(count)
            flat = start.repeat_interleave(count) + offset
            cand_pt = parent_pt.repeat_interleave(count)
            cand_id = child[flat]
            lower_bound = (parent_d.repeat_interleave(count) - child_dp[flat]).abs()
            keep = lower_bound < best_d[cand_pt] + slack
            cand_pt, cand_id = cand_pt[keep], cand_id[keep]
            if cand_pt.numel():
                cand_d = evaluate_distances(feats, X, cand_pt, cand_id)
                winners = first_per_point(cand_pt, cand_d, cand_id)
                win_pt = cand_pt[winners]
                improve = cand_d[winners] < best_d[win_pt]
                best_d[win_pt[improve]] = cand_d[winners][improve]
                best_id[win_pt[improve]] = cand_id[winners][improve]

        if level >= record_from:
            assigns[:, level - record_from] = best_id
            dists[:, level - record_from] = best_d

        pool_pt = torch.cat([parent_pt, cand_pt])
        pool_id = torch.cat([parent_id, cand_id])
        pool_d = torch.cat([parent_d, cand_d])
        keep = pool_d <= best_d[pool_pt] + slack
        pool_pt, pool_id, pool_d = pool_pt[keep], pool_id[keep], pool_d[keep]

        order = torch.argsort(pool_d, stable=True)
        order = order[torch.argsort(pool_pt[order], stable=True)]
        pt_sorted = pool_pt[order]
        n_pool = pt_sorted.shape[0]
        boundary = torch.ones(n_pool, dtype=torch.bool, device=device)
        if n_pool > 1:
            boundary[1:] = pt_sorted[1:] != pt_sorted[:-1]
        idx = torch.arange(n_pool, device=device)
        group_start = idx[boundary]
        group_sizes = torch.diff(torch.cat([group_start, torch.tensor([n_pool], device=device)]))
        rank = idx - group_start.repeat_interleave(group_sizes)
        counts_per_point = torch.zeros(num_points, dtype=torch.int64, device=device)
        counts_per_point.scatter_add_(0, pt_sorted, torch.ones_like(pt_sorted))
        overflow |= counts_per_point > beam_width

        within = rank < beam_width
        beam_ids.fill_(-1)
        beam_d.fill_(float("inf"))
        beam_ids[pt_sorted[within], rank[within]] = pool_id[order][within]
        beam_d[pt_sorted[within], rank[within]] = pool_d[order][within]

    return assigns, dists, overflow


def brute_all_levels(X, feats, max_level, chunk=262_144):
    device = X.device
    num_points = X.shape[0]
    best_d = (X - feats[0]).pow(2).sum(1).sqrt_()
    best_id = torch.zeros(num_points, dtype=torch.int64, device=device)
    assigns = torch.empty((num_points, max_level + 1), dtype=torch.int64, device=device)
    dists = torch.empty((num_points, max_level + 1), dtype=torch.float32, device=device)
    assigns[:, 0] = 0
    dists[:, 0] = best_d
    for level in range(1, max_level + 1):
        for lo in range(2 ** (level - 1), 2 ** level, chunk):
            hi = min(lo + chunk, 2 ** level)
            d = torch.cdist(X, feats[lo:hi])
            chunk_min, chunk_arg = d.min(1)
            improve = chunk_min < best_d
            best_d[improve] = chunk_min[improve]
            best_id[improve] = chunk_arg[improve] + lo
        assigns[:, level] = best_id
        dists[:, level] = best_d
    return assigns, dists


def net_to_device(tree, device):
    net = {
        "feats": tree["feats"].to(device),
        "suffix": tree["suffix"],
        "levels": {},
    }
    for level, csr in tree["levels"].items():
        net["levels"][level] = {key: value.to(device) for key, value in csr.items()}
    return net
