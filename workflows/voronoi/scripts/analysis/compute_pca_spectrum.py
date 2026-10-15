import numpy as np

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
N_POINTS = 2 ** 20
LEVEL = 20
N_COLLECTION = 1_695_275_276

features = np.memmap(f"{voronoi_dir}/out/tree/center_feats_scaled.f32", dtype=np.float32,
                     mode="r", shape=(2 ** 24, 512))
X = np.asarray(features[:N_POINTS], dtype=np.float64)

aggregated = np.load(f"{voronoi_dir}/out/analysis/populations.npz")
populations = aggregated["populations"]
level_offsets = aggregated["level_offsets"]
# populations = concatenated per-level bincounts of the assignment columns, built by
# aggregate_voronoi.py over all 8,096 files in out/assign: for level k, entry i counts the
# rows of the full collection whose a{k} column equals i (= structures whose nearest center
# among the first 2^k coreset points is point i). Level k occupies
# populations[level_offsets[k]:level_offsets[k+1]].
voronoi_counts = populations[level_offsets[LEVEL]:level_offsets[LEVEL + 1]].astype(np.float64)

assert voronoi_counts.shape[0] == N_POINTS
assert int(voronoi_counts.sum()) == N_COLLECTION
assert int((voronoi_counts > 0).sum()) == N_POINTS
print(f"level-{LEVEL} voronoi counts: {N_POINTS:,} entries, sum {int(voronoi_counts.sum()):,}, "
      f"min {int(voronoi_counts.min())}, max {int(voronoi_counts.max()):,}", flush=True)


def explained_variance_ratios(X, weights):
    probabilities = weights / weights.sum()
    mean = (probabilities[:, None] * X).sum(0)
    centered = X - mean
    covariance = (centered * probabilities[:, None]).T @ centered
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    return eigenvalues / eigenvalues.sum()


uniform_ratios = explained_variance_ratios(X, np.ones(N_POINTS))
weighted_ratios = explained_variance_ratios(X, voronoi_counts)

np.savez(f"{voronoi_dir}/out/analysis/pca_spectrum_2p20.npz",
         plain=uniform_ratios, weighted=weighted_ratios)

print(f"\n{'components':>10} {'uniform cum.':>13} {'weighted cum.':>14}")
for k in (2, 4, 8, 16, 32, 64, 128, 256):
    print(f"{k:>10} {100 * uniform_ratios[:k].sum():>12.1f}% {100 * weighted_ratios[:k].sum():>13.1f}%")
print(f"\nsaved {voronoi_dir}/out/analysis/pca_spectrum_2p20.npz", flush=True)
