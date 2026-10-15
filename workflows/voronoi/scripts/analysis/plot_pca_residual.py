import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"

spectrum = np.load(f"{voronoi_dir}/out/analysis/pca_spectrum_2p20.npz")
components = np.arange(1, 513)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(components, 1.0 - np.cumsum(spectrum["plain"]), color="#1f77b4", lw=1.8,
          label="uniform")
ax.loglog(components, 1.0 - np.cumsum(spectrum["weighted"]), color="#d62728", lw=1.8,
          label="voronoi-weighted")
ax.legend(fontsize=9, frameon=False)
ax.set_xlabel("number of principal components", fontsize=10)
ax.set_ylabel("residual variance", fontsize=10)
ax.set_xlim(1, 512)
ax.set_ylim(0.02, 1.0)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"], fontsize=9)
ax.set_yticks([0.05, 0.1, 0.2, 0.4, 1.0])
ax.set_yticklabels(["5%", "10%", "20%", "40%", "100%"], fontsize=9)
fig.tight_layout()
fig.savefig(f"{voronoi_dir}/plots/pca_residual_variance.png", dpi=150, bbox_inches="tight")
print(f"saved {voronoi_dir}/plots/pca_residual_variance.png")
