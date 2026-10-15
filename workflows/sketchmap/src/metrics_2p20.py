"""Score a 2^20 embedding against its PCA init: does it restructure, or just rescale?"""

import argparse
import collections
import json

import numpy as np
import torch
from scipy.spatial.distance import pdist

analysis = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
voronoi = "${DEV_ROOT}/phace-atomic/voronoi"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"
N = 2 ** 20
QUERY = 50_000
SUB = 5_000
device = "cuda"


def knn_family(E, fam_idx, q, k=5):
    """Mean fraction of each query point's k nearest neighbours sharing its family."""
    Eg = torch.as_tensor(E, device=device, dtype=torch.float32)
    fg = torch.as_tensor(fam_idx, device=device)
    hits = torch.zeros((), device=device, dtype=torch.float64)
    for lo in range(0, len(q), 2048):
        qi = torch.as_tensor(q[lo:lo + 2048], device=device)
        d = torch.cdist(Eg[qi], Eg)
        d[torch.arange(len(qi), device=device), qi] = float("inf")  # drop self
        nn = d.topk(k, largest=False).indices
        hits += (fg[nn] == fg[qi][:, None]).sum(dtype=torch.float64)
    return float(hits / (len(q) * k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--points", help="parquet with family + feat_scaled (random sample)")
    a = p.parse_args()

    emb = np.load(f"{run_dir}/results/embeddings/embedding_l20_d{a.d}_{a.tag}.npy")
    pca = np.load(f"{run_dir}/results/embeddings/init_{a.tag}.npy")
    if a.points:
        import pyarrow.parquet as pq
        t = pq.read_table(a.points, columns=["family", "feat_scaled"])
        fam = np.asarray([str(x) for x in t.column("family").to_pylist()])
        X = (t.column("feat_scaled").combine_chunks().flatten().to_numpy()
             .reshape(-1, 512).astype(np.float32))
    else:
        fam = np.asarray([str(f) for f in
                          np.load(f"{analysis}/family_level20.npy", allow_pickle=True)])
        X = None
    assert emb.shape == (N, a.d) and pca.shape == (N, a.d) and fam.shape == (N,)

    names = sorted(set(fam.tolist()))
    code = {f: i for i, f in enumerate(names)}
    fam_idx = np.array([code[f] for f in fam], dtype=np.int64)
    shares = np.array(list(collections.Counter(fam.tolist()).values()), dtype=np.float64) / N
    chance = float((shares ** 2).sum())

    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(N, QUERY, replace=False))
    fam_emb = knn_family(emb, fam_idx, q)
    fam_pca = knn_family(pca, fam_idx, q)

    if X is None:
        X = np.memmap(f"{voronoi}/out/tree/center_feats_scaled.f32", dtype=np.float32,
                      mode="r", shape=(2 ** 24, 512))
    s = np.sort(rng.choice(N, SUB, replace=False))
    dX = pdist(np.asarray(X[s], dtype=np.float64))
    dE = pdist(emb[s].astype(np.float64))
    dP = pdist(pca[s].astype(np.float64))

    r = np.linalg.norm(emb - emb.mean(0), axis=1)
    rp = np.linalg.norm(pca - pca.mean(0), axis=1)
    out = {
        "tag": a.tag, "d": a.d, "n": N,
        "family_5nn_sketchmap": fam_emb,
        "family_5nn_pca": fam_pca,
        "family_5nn_chance": chance,
        "ratio_sketchmap": fam_emb / chance,
        "ratio_pca": fam_pca / chance,
        "corr_emb_vs_X": float(np.corrcoef(dE, dX)[0, 1]),
        "corr_pca_vs_X": float(np.corrcoef(dP, dX)[0, 1]),
        "corr_emb_vs_pca": float(np.corrcoef(dE, dP)[0, 1]),
        "scale_ratio_emb_over_pca": float(np.median(r) / np.median(rp)),
        "max_r_over_median_r": float(r.max() / np.median(r)),
        "max_r_over_median_r_pca": float(rp.max() / np.median(rp)),
    }
    with open(f"{run_dir}/results/metrics/score_{a.tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n=== {a.tag} (2^20, d={a.d}) — {QUERY:,} query pts, {len(names)} families ===")
    print(f"  family 5NN  sketchmap : {fam_emb:.4f}  = {fam_emb/chance:.2f}x chance")
    print(f"  family 5NN  PCA       : {fam_pca:.4f}  = {fam_pca/chance:.2f}x chance")
    print(f"  family 5NN  chance    : {chance:.4f}")
    print(f"  corr(d_emb, D_512)    : {out['corr_emb_vs_X']:.4f}")
    print(f"  corr(d_pca, D_512)    : {out['corr_pca_vs_X']:.4f}")
    print(f"  corr(d_emb, d_pca)    : {out['corr_emb_vs_pca']:.4f}   <- 1.0 would mean 'PCA'")
    print(f"  median radius emb/pca : {out['scale_ratio_emb_over_pca']:.4f}  "
          f"<- ~uniform rescale if corr_emb_vs_pca ~ 1")
    print(f"  max_r/median_r        : {out['max_r_over_median_r']:.2f} "
          f"(pca {out['max_r_over_median_r_pca']:.2f})")


if __name__ == "__main__":
    main()
