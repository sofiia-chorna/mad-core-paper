import argparse
import json

import numpy as np
import torch
from scipy.spatial.distance import pdist

try:
    from skmatter.decomposition._sketchmap_utils import suggest_sigmoid_params
except ImportError:  # renamed in later scikit-matter
    from skmatter.decomposition._sketchmap_utils import _suggest_sigmoid_params as suggest_sigmoid_params

analysis = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
voronoi = "${DEV_ROOT}/phace-atomic/voronoi"
N = 2 ** 20
WGAMMA = 0.25
BINS = 500

parser = argparse.ArgumentParser()
parser.add_argument("--populations", help="level-20 populations (default: the July file)")
parser.add_argument("--sigma-out", help="write the two histogram peaks and sigma here as JSON")
args = parser.parse_args()

feats = np.memmap(f"{voronoi}/out/tree/center_feats_scaled.f32", dtype=np.float32, mode="r",
                  shape=(2 ** 24, 512))
if args.populations:
    counts = np.load(args.populations).astype(np.float64)
    assert counts.shape == (N,) and counts.min() > 0, (counts.shape, counts.min())
else:
    counts = np.load(f"{analysis}/populations_level20.npy").astype(np.float64)
    assert counts.sum() == 1_695_275_276
print(f"populations {args.populations or 'July'}: sum {counts.sum():,.0f}")

rng = np.random.default_rng(0)
idx4k = np.sort(rng.choice(N, 4000, replace=False))
X4k = np.asarray(feats[idx4k], dtype=np.float64)

print("=== 1. skmatter auto-suggest on a random 4,000-point subset of the 2^20 ===")
print("    (takes DISTANCES, not X)")
sug, ana = suggest_sigmoid_params(pdist(X4k), 8)
for k, v in sug.items():
    print(f"      {k:>8} = {v:.4f}")
print(f"      peak_distance = {ana['peak_distance']:.4f}")
print("    they claim sigma=1.327, a_high=4.525, b_high=2.262")

print("\n=== 2. the weighted correction (auto-suggest cannot see the weights) ===")
device = "cuda"
idx = np.sort(rng.choice(N, 24_000, replace=False))
X = torch.as_tensor(np.asarray(feats[idx], dtype=np.float64), device=device)
D = torch.cdist(X, X, compute_mode="donot_use_mm_for_euclid_dist")
iu = torch.triu_indices(X.shape[0], X.shape[0], offset=1, device=device)
d = D[iu[0], iu[1]]
del D
w = counts ** WGAMMA
w = w / w.sum()
ws = torch.as_tensor(w[idx] / w[idx].sum(), device=device)
pw = ws[iu[0]] * ws[iu[1]]

edges = torch.linspace(0.0, float(d.max()) + 1e-6, BINS + 1, device=device, dtype=d.dtype)
centers = (0.5 * (edges[1:] + edges[:-1])).cpu().numpy()
b = torch.clamp(torch.bucketize(d, edges) - 1, 0, BINS - 1)
hu = torch.zeros(BINS, device=device, dtype=d.dtype)
hu.scatter_add_(b_i := b, torch.ones_like(d)) if False else hu.scatter_add_(0, b, torch.ones_like(d))
hw = torch.zeros(BINS, device=device, dtype=d.dtype)
hw.scatter_add_(0, b, pw)
peak_u = float(centers[int(torch.argmax(hu))])
peak_w = float(centers[int(torch.argmax(hw))])
print(f"      unweighted peak      = {peak_u:.4f}  -> 0.9*peak = {0.9 * peak_u:.4f}")
print(f"      PAIR-WEIGHTED peak   = {peak_w:.4f}  -> 0.9*peak = {0.9 * peak_w:.4f}")
print(f"      gap = {100 * abs(0.9 * peak_u - 0.9 * peak_w) / (0.9 * peak_w):.1f}%   "
      f"(they claim 7% at 2^20 vs 58% at 2^14)")
if args.sigma_out:
    with open(args.sigma_out, "w") as fh:
        json.dump({"populations": args.populations, "peak_unweighted": peak_u,
                   "peak_pair_weighted": peak_w, "sigma": round(0.9 * peak_w, 2)}, fh, indent=1)
    print(f"      wrote {args.sigma_out}: sigma = {round(0.9 * peak_w, 2)}")

print("\n=== 3. filter occupancy: does A/B choice matter? ===")
print(f"{'sigma':>7} {'A':>7} {'B':>7} {'close':>7} {'far':>7} {'mid':>7}")
for sigma in (1.24, 1.327):
    for A, B in ((4.0, 2.0), (4.525, 2.262)):
        Ac = 2.0 ** (A / B) - 1.0
        f = 1.0 - (1.0 + Ac * (d / sigma) ** A) ** (-B / A)
        close = float((pw * (f < 0.1)).sum() / pw.sum())
        far = float((pw * (f > 0.9)).sum() / pw.sum())
        print(f"{sigma:>7.3f} {A:>7.3f} {B:>7.3f} {100 * close:>6.1f}% {100 * far:>6.1f}% "
              f"{100 * (1 - close - far):>6.1f}%")
