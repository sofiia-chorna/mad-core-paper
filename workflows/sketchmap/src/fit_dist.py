import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import minimize

from skmatter.decomposition._sketchmap_torch import TiledStress

analysis = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
voronoi = "${DEV_ROOT}/phace-atomic/voronoi"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--level", type=int, default=20)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--a-high", type=float, default=4.0)
    p.add_argument("--b-high", type=float, default=2.0)
    p.add_argument("--a-low", type=float, default=2.0)
    p.add_argument("--b-low", type=float, default=2.0)
    p.add_argument("--wgamma", type=float, default=0.25)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--row-tile", type=int, default=8192)
    p.add_argument("--col-tile", type=int, default=16384)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--gtol", type=float, default=1e-8)
    p.add_argument("--resume")
    p.add_argument("--points", help="parquet with feat_scaled; uniform weights, ignores --level/--wgamma")
    p.add_argument("--populations",
                   help="Voronoi populations of the landmarks (default: the July "
                        "populations_level<level>.npy of the voronoi analysis)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    ngpu = torch.cuda.device_count()
    torch.cuda.set_device(0)
    devices = [f"cuda:{i}" for i in range(ngpu)]
    N, K = 2 ** args.level, args.n_components

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log(f"[{args.tag}] N={N:,} d={K} world={world} gpus/rank={ngpu} "
        f"total_gpus={world*ngpu} tf32={args.tf32}")

    if args.points:
        import pyarrow.parquet as pq
        col = pq.read_table(args.points, columns=["feat_scaled"]).column("feat_scaled")
        X = col.combine_chunks().flatten().to_numpy().reshape(-1, 512).astype(np.float32)
        X = np.ascontiguousarray(X)
        N = X.shape[0]
        w = np.full(N, 1.0 / N, dtype=np.float32)
        n_eff = float(N)
        log(f"[{args.tag}] points={args.points} N={N:,} uniform weights")
    else:
        feats = np.memmap(f"{voronoi}/out/tree/center_feats_scaled.f32", dtype=np.float32,
                          mode="r", shape=(2 ** 24, 512))
        X = np.ascontiguousarray(feats[:N])
        if args.populations:
            counts = np.load(args.populations).astype(np.float64)
            assert counts.shape == (N,) and counts.min() > 0, (counts.shape, counts.min())
        else:
            counts = np.load(f"{analysis}/populations_level{args.level}.npy").astype(np.float64)
            assert counts.shape == (N,) and counts.sum() == 1_695_275_276, counts.sum()
        log(f"[{args.tag}] populations {args.populations or 'July'}: sum {counts.sum():,.0f}")
        w = counts ** args.wgamma
        w = (w / w.sum()).astype(np.float32)
        n_eff = 1.0 / (w.astype(np.float64) ** 2).sum()
        log(f"[{args.tag}] wgamma={args.wgamma} N_eff={n_eff:,.0f}/{N:,} ({100*n_eff/N:.1f}%)")

    edges = np.linspace(0, N, world + 1).astype(int)
    lo, hi = int(edges[rank]), int(edges[rank + 1])
    params = {"sigma": args.sigma, "a_high": args.a_high, "b_high": args.b_high,
              "a_low": args.a_low, "b_low": args.b_low}
    eng = TiledStress(X, w, K, params, devices=devices, row_tile=args.row_tile,
                      col_tile=args.col_tile, compile_kernel=True, row_range=(lo, hi))
    log(f"[{args.tag}] total_weight={eng.total_weight:.6e} (closed form); "
        f"rank0 rows {lo:,}..{hi:,}")

    if args.resume and os.path.exists(args.resume):
        Y0 = np.load(args.resume).astype(np.float64)
        log(f"[{args.tag}] resumed from {args.resume}")
    else:
        from sklearn.decomposition import PCA
        t0 = time.time()
        Y0 = PCA(n_components=K, random_state=0).fit_transform(X).astype(np.float64)
        log(f"[{args.tag}] PCA init in {time.time()-t0:.1f}s (skips classical_mds)")

    dev = "cuda:0"
    ckpt = f"{run_dir}/results/embeddings/ckpt_{args.tag}.npy"
    state = {"t0": time.time(), "n": 0}

    def fun(x):
        s_loc, g_loc = eng.stress_and_grad(x, 0.0, True)
        buf = torch.as_tensor(np.concatenate([[s_loc], g_loc]), device=dev)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        out = buf.cpu().numpy()
        state["n"] += 1
        if rank == 0:
            if state["n"] % 5 == 0:
                el = time.time() - state["t0"]
                print(f"[{args.tag}] eval {state['n']:4d} stress={out[0]:.8f} "
                      f"{el/state['n']:.1f}s/eval elapsed={el/60:.1f}m", flush=True)
            if state["n"] % args.checkpoint_every == 0:
                np.save(ckpt, x.reshape(N, K).astype(np.float32))
        return float(out[0]), out[1:]

    t0 = time.time()
    res = minimize(fun, Y0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "gtol": args.gtol, "ftol": 1e-16})
    secs = time.time() - t0

    if rank == 0:
        Y = res.x.reshape(N, K).astype(np.float32)
        out = (f"{run_dir}/results/embeddings/"
               f"embedding_l{args.level}_d{K}_{args.tag}.npy")
        np.save(out, Y)
        np.save(f"{run_dir}/results/embeddings/init_{args.tag}.npy", Y0.astype(np.float32))

        # did the fit actually move the points, or only the heavy ones?
        shift = np.linalg.norm(Y - Y0.reshape(N, K), axis=1)
        scale = float(np.median(np.linalg.norm(Y0.reshape(N, K), axis=1)))
        moved = {f"moved_gt_{q}_of_median_radius": float((shift > q * scale).mean())
                 for q in (0.01, 0.1, 0.5)}
        wq = np.argsort(w)[::-1]
        heavy = wq[: max(1, int(n_eff))]
        moved["median_shift_all"] = float(np.median(shift))
        moved["median_shift_top_N_eff"] = float(np.median(shift[heavy]))
        print(f"[{args.tag}] movement vs PCA init: {moved}", flush=True)
        rec = {"tag": args.tag, "level": args.level, "n_components": K,
               "params": params, "wgamma": args.wgamma, "n_eff": n_eff,
               "total_weight": eng.total_weight, "final_stress": float(res.fun),
               "n_evals": state["n"], "nit": int(res.nit),
               "converged": bool(res.success), "message": str(res.message),
               "fit_seconds": secs, "tf32": args.tf32, "gtol": args.gtol, "world": world,
               "gpus": world * ngpu, "embedding": out, "moved_vs_init": moved}
        with open(f"{run_dir}/results/metrics/tiled_{args.tag}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"\n[{args.tag}] DONE stress={res.fun:.8f} evals={state['n']} "
              f"nit={res.nit} {secs/60:.1f}m\n[{args.tag}] {res.message}\n"
              f"[{args.tag}] -> {out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
