import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

voronoi_dir = "${DEV_ROOT}/phace-atomic/voronoi"
analysis_dir = f"{voronoi_dir}/out/analysis"
plots_dir = f"{voronoi_dir}/plots"
lean_coreset = "${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92/coreset224/out/coreset_224.parquet"
MAX_LEVEL = 24

pop_data = np.load(f"{analysis_dir}/populations.npz")
populations, level_offsets = pop_data["populations"], pop_data["level_offsets"]
hist_data = np.load(f"{analysis_dir}/distance_hist.npz")
dist_hist, dist_bins = hist_data["hist"], hist_data["bins"]
bin_centers = 0.5 * (dist_bins[1:] + dist_bins[:-1])


def histogram_percentile(hist_row, q):
    cumulative = np.cumsum(hist_row)
    index = min(int(np.searchsorted(cumulative, q * cumulative[-1])), len(bin_centers) - 1)
    return bin_centers[index]


def distance_vs_coreset_size():
    levels = np.arange(MAX_LEVEL + 1)
    sizes = 2.0 ** levels
    median = [histogram_percentile(dist_hist[k], 0.5) for k in levels]
    p10 = [histogram_percentile(dist_hist[k], 0.10) for k in levels]
    p90 = [histogram_percentile(dist_hist[k], 0.90) for k in levels]
    p99 = [histogram_percentile(dist_hist[k], 0.99) for k in levels]
    eps_column = np.sqrt(pq.read_table(lean_coreset, columns=["epsilon_sq"]).column("epsilon_sq").to_numpy())
    eps_levels = [20, 21, 22, 23, 24]
    eps_values = [eps_column[2 ** k - 1] for k in eps_levels]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(sizes, p10, p90, alpha=0.18, color="#1f77b4", label="10–90% of structures")
    ax.loglog(sizes, median, "-o", ms=4, color="#1f77b4", label="typical structure (median)")
    ax.loglog(sizes, p99, "--", color="#ff7f0e", label="worst 1%")
    ax.loglog([2 ** k for k in eps_levels], eps_values, ":s", ms=5, color="#d62728",
              label=r"coreset resolution $\varepsilon$")
    ax.set_xlabel("coreset size (number of selected structures)", fontsize=10)
    ax.set_ylabel("distance to assigned coreset structure", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/distance_vs_coreset_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def structures_per_representative():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k, color, label in [(8, "#1f77b4", "coreset of 256"), (12, "#2ca02c", "coreset of 4,096"),
                            (16, "#ff7f0e", "coreset of 65,536"), (20, "#9467bd", "coreset of 1,048,576"),
                            (24, "#d62728", "coreset of 16,777,216")]:
        counts = np.sort(populations[level_offsets[k]:level_offsets[k + 1]])[::-1]
        ax.loglog(np.arange(1, len(counts) + 1), counts, color=color, lw=1.8, label=label)
    ax.set_xlabel("coreset structures sorted by how many structures they are nearest center to", fontsize=10)
    ax.set_ylabel("number of structures it represents", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/structures_per_representative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def concentration_vs_coreset_size():
    levels = range(4, MAX_LEVEL + 1)
    sizes = [2 ** k for k in levels]
    total = populations[level_offsets[24]:level_offsets[25]].sum()
    largest, top1pct, top10pct = [], [], []
    for k in levels:
        counts = np.sort(populations[level_offsets[k]:level_offsets[k + 1]])[::-1]
        largest.append(100 * counts[0] / total)
        top1pct.append(100 * counts[:max(1, len(counts) // 100)].sum() / total)
        top10pct.append(100 * counts[:max(1, len(counts) // 10)].sum() / total)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.semilogx(sizes, largest, "-o", ms=4, color="#d62728", label="largest group")
    ax.semilogx(sizes, top1pct, "-s", ms=4, color="#ff7f0e", label="largest 1% of groups")
    ax.semilogx(sizes, top10pct, "-^", ms=4, color="#1f77b4", label="largest 10% of groups")
    ax.set_xlabel("coreset size", fontsize=10)
    ax.set_ylabel("% of all structures", fontsize=10)
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{plots_dir}/concentration_vs_coreset_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


distance_vs_coreset_size()
structures_per_representative()
concentration_vs_coreset_size()
print("saved distance_vs_coreset_size / structures_per_representative / concentration_vs_coreset_size")
