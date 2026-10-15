import os
import pickle
import sys
import time

import numpy as np
import torch

RELEASE = "${STORE_ROOT}/mad-core/release/mad-core"
SCALER = "${STORE_ROOT}/mad-core/mad_union/scaler_madunion_phaceatomic_meanstd.pkl"
HERE = "${DEV_ROOT}/phace-atomic/accrete/runs/coreset224/coverage_cert"
OUT = f"{HERE}/out/stageC"
N = 16777216
DIM = 512
QUERY = 1024
BLOCK = 1 << 19


def load_features(variant):
    if variant == "release":
        raw = np.fromfile(f"{RELEASE}/ecumetric_features.bin", dtype=np.float32).reshape(N, DIM)
    else:
        raw = np.load(f"{HERE}/out/selection_features_madcore.npy")
    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)
    scale = float(scaler["scale_"])
    x = torch.from_numpy(raw).cuda()
    del raw
    return x, scale


def rank_range(rank, world):
    lo = int(round(N * np.sqrt(rank / world)))
    hi = int(round(N * np.sqrt((rank + 1) / world)))
    return lo, hi


def main():
    variant = sys.argv[1]
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("COV_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("COV_WORLD", "1")))
    torch.cuda.set_device(int(os.environ.get("COV_DEVICE", os.environ.get("SLURM_LOCALID", os.environ.get("COV_RANK", "0")))))
    os.makedirs(OUT, exist_ok=True)
    lo, hi = rank_range(rank, world)
    started = time.time()

    x32, scale = load_features(variant)
    norms = torch.empty(N, dtype=torch.float64, device="cuda")
    for s in range(0, N, BLOCK):
        blk = x32[s:s + BLOCK].double()
        norms[s:s + BLOCK] = (blk * blk).sum(dim=1)
    del blk
    print(f"[{variant} rank {rank}] rows [{lo},{hi}) features on GPU ({time.time() - started:.0f}s)", flush=True)

    sep = torch.full((hi - lo,), float("inf"), dtype=torch.float64, device="cuda")
    arg = torch.full((hi - lo,), -1, dtype=torch.int64, device="cuda")

    for qs in range(lo, hi, QUERY):
        qe = min(qs + QUERY, hi)
        q = x32[qs:qe].double()
        best = torch.full((qe - qs,), float("inf"), dtype=torch.float64, device="cuda")
        best_arg = torch.full((qe - qs,), -1, dtype=torch.int64, device="cuda")
        for bs in range(0, qe, BLOCK):
            be = min(bs + BLOCK, qe)
            blk = x32[bs:be].double()
            d = norms[qs:qe, None] + norms[None, bs:be] - 2.0 * (q @ blk.T)
            cols = torch.arange(bs, be, device="cuda")[None, :]
            rows = torch.arange(qs, qe, device="cuda")[:, None]
            d.masked_fill_(cols >= rows, float("inf"))
            val, idx = d.min(dim=1)
            better = val < best
            best = torch.where(better, val, best)
            best_arg = torch.where(better, idx + bs, best_arg)
        sep[qs - lo:qe - lo] = best.clamp_min(0.0) / (scale * scale)
        arg[qs - lo:qe - lo] = best_arg
        if (qs - lo) % (QUERY * 512) == 0:
            print(f"[{variant} rank {rank}] {qs - lo:,}/{hi - lo:,} ({time.time() - started:.0f}s)", flush=True)

    np.save(f"{OUT}/sep_sq_{variant}_rank{rank}.npy", sep.cpu().numpy())
    np.save(f"{OUT}/nearest_{variant}_rank{rank}.npy", arg.cpu().numpy())
    np.save(f"{OUT}/range_{variant}_rank{rank}.npy", np.asarray([lo, hi]))
    print(f"[{variant} rank {rank}] done ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
