import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ase.data import chemical_symbols

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/fps/analysis")
import element_map as em

K = 2 ** 24
runs_dir = "${DEV_ROOT}/phace-atomic/accrete/runs"
run_dir = f"{runs_dir}/coreset224"
plots_dir = f"{run_dir}/plots"
anchor_size = 856_012
coreset_10m_full = f"{runs_dir}/madcore_10M/out/coreset_10M_full.parquet"
anchor_parquet = "${STORE_ROOT}/mad-core/mad_union/anchor.parquet"

light = pq.read_table(f"{run_dir}/out/coreset_224_light.parquet",
                      columns=["fps_order", "family", "n_sites", "elements", "configuration_id"])
order = np.argsort(light.column("fps_order").to_numpy())
families = np.asarray(light.column("family").to_pylist())[order]
n_sites = light.column("n_sites").to_numpy()[order]
elements_column = pc.take(light.column("elements").combine_chunks(), pa.array(order))
configuration_ids = np.asarray(light.column("configuration_id").to_pylist())[order]
del light
print("light loaded", flush=True)


def plot_family_decomposition():
    palette = (list(plt.colormaps["tab20"].colors) + list(plt.colormaps["tab20b"].colors)
               + list(plt.colormaps["tab20c"].colors))
    family_10m = pq.read_table(coreset_10m_full, columns=["family"]).column("family").to_pylist()
    canonical_order, seen = [], set()
    for name, _ in Counter(family_10m).most_common():
        name = "mad-union" if name == "mad-coreset" else name
        if name not in seen:
            seen.add(name)
            canonical_order.append(name)
    counts_total = Counter(families.tolist())
    for name, _ in counts_total.most_common():
        if name not in seen:
            seen.add(name)
            canonical_order.append(name)
    color_of = {name: palette[index % len(palette)] for index, name in enumerate(canonical_order)}

    shown_set = {name for name, _ in counts_total.most_common(36)}
    shown = [name for name in canonical_order if name in shown_set]
    other_label = f"{len(counts_total) - len(shown)} more datasets"
    sizes = [2 ** i for i in range(10, 25)]
    counts_by_size = {size: Counter(families[:size].tolist()) for size in sizes}

    figure, axis = plt.subplots(figsize=(13, 7))
    bottom = np.zeros(len(sizes))
    for label in shown + [other_label]:
        if label == other_label:
            fraction = [1.0 - sum(counts_by_size[s].get(n, 0) for n in shown) / s for s in sizes]
            color = "#d9d9d9"
        else:
            fraction = [counts_by_size[s].get(label, 0) / s for s in sizes]
            color = color_of[label]
        axis.bar(range(len(sizes)), fraction, bottom=bottom, width=0.82, color=color, label=label)
        bottom += np.array(fraction)
    axis.set_xticks(range(len(sizes)))
    axis.set_xticklabels([f"$2^{{{i}}}$" for i in range(10, 25)])
    axis.set_xlabel("coreset size $K$", fontsize=13)
    axis.set_ylabel("fraction of picks", fontsize=13)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=7, frameon=True)
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/decomposition_families_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("family decomposition saved", flush=True)


def plot_category_decomposition():
    category_map = json.load(open(f"{run_dir}/out/picks_category_map_224.json"))
    category_colors = json.load(open(f"{run_dir}/out/picks_category_colors_224.json"))
    categories = np.asarray([category_map[cid] for cid in configuration_ids])
    sizes = np.unique(np.geomspace(1, K, 600).astype(int))
    figure, axis = plt.subplots(figsize=(9, 6))
    for category in sorted(set(categories.tolist()), key=lambda c: -int((categories == c).sum())):
        cumulative = np.cumsum(categories == category)
        axis.plot(sizes, cumulative[sizes - 1], linewidth=2.2,
                  color=category_colors.get(category), label=f"{category} ({cumulative[-1]})")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("coreset size $K$", fontsize=13)
    axis.set_ylabel("# structures", fontsize=13)
    axis.legend()
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/decomposition_categories_log_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("category decomposition saved", flush=True)


def plot_element_map():
    values = np.asarray(elements_column.values.to_numpy(zero_copy_only=False), dtype=np.int64)
    offsets = np.asarray(elements_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    pick_of_entry = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    panel_sizes = [2 ** 20, 2 ** 22, 2 ** 24]
    counts_per_size = []
    for size in panel_sizes:
        cut = int(np.searchsorted(pick_of_entry, size, side="left"))
        counts_per_size.append(np.bincount(values[:cut], minlength=em.MAX_Z + 1))
    shared_vmax = int(max(c.max() for c in counts_per_size))
    shared_vmin = int(min(c[c > 0].min() for c in counts_per_size if (c > 0).any()))
    figure, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    for axis, size, counts in zip(axes, panel_sizes, counts_per_size):
        element_counts = {chemical_symbols[z]: int(counts[z]) for z in range(em.MAX_Z + 1) if counts[z] > 0}
        em.draw_pt(axis, element_counts, f"$K=2^{{{int(np.log2(size))}}}$", "counts", 2.0,
                   vmax_counts=shared_vmax, vmin_counts=shared_vmin)
    plt.tight_layout()
    figure.savefig(f"{plots_dir}/element_map_224.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print("element map saved", flush=True)


def plot_mad_retention():
    mad_cids = set(pq.read_table(anchor_parquet, columns=["configuration_id"]).column(0).to_pylist())
    retained = np.cumsum(np.fromiter((cid in mad_cids for cid in configuration_ids), dtype=bool, count=K))
    sizes = np.unique(np.geomspace(anchor_size, K, 400).astype(int))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(sizes, retained[sizes - 1], linewidth=2.4, color="#1f77b4", label="accrete grows (natural)")
    axis.axhline(len(mad_cids), ls="--", lw=1.8, color="#d62728", label=f"force all MAD ({len(mad_cids):,})")
    for size in (anchor_size, 2 ** 23, K):
        axis.plot(size, retained[size - 1], "o", color="#1f77b4", ms=6)
        axis.annotate(f"{int(retained[size - 1]):,}", (size, retained[size - 1]),
                      textcoords="offset points", xytext=(6, -12), fontsize=10)
    axis.set_xscale("log")
    axis.set_xlabel("coreset size $K$", fontsize=13)
    axis.set_ylabel("MAD structures retained", fontsize=13)
    axis.set_xlim(anchor_size, K)
    axis.legend(fontsize=11, loc="center right")
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/mad_retention_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"mad retention saved (retained at 2^24: {int(retained[-1]):,})", flush=True)


def plot_coverage_radius():
    full = pq.read_table(f"{run_dir}/out/coreset_224_full.parquet", columns=["fps_order", "epsilon_sq"])
    epsilon = np.sqrt(full.column("epsilon_sq").to_numpy()[np.argsort(full.column("fps_order").to_numpy())])
    del full
    sizes = np.unique(np.geomspace(2, K, 800).astype(int))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(sizes, epsilon[sizes - 1], linewidth=2.2, color="#1f77b4")
    axis.axvline(anchor_size, ls="--", lw=1.5, color="#d62728")
    axis.annotate(f"accrete anchor\n({anchor_size:,})", (anchor_size, epsilon[sizes - 1].max() * 0.5),
                  xytext=(8, 0), textcoords="offset points", fontsize=10, color="#d62728")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("coreset size $K$", fontsize=13)
    axis.set_ylabel(r"coverage radius $\varepsilon(K)$", fontsize=13)
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/coverage_radius_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"coverage radius saved (eps at 2^24: {epsilon[-1]:.4g})", flush=True)


def plot_discovery():
    first_of_family = {}
    for position, family in enumerate(families):
        if family not in first_of_family:
            first_of_family[family] = position
    family_first = np.sort(np.array(list(first_of_family.values())))
    values = np.asarray(elements_column.values.to_numpy(zero_copy_only=False), dtype=np.int64)
    offsets = np.asarray(elements_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    pick_of_entry = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    first_of_element = {}
    for entry_position, z in zip(pick_of_entry, values):
        if z not in first_of_element:
            first_of_element[int(z)] = int(entry_position)
    element_first = np.sort(np.array(list(first_of_element.values())))
    sizes = np.unique(np.geomspace(2, K, 800).astype(int))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, first_seen, total, label, color in [
        (axes[0], family_first, len(first_of_family), "distinct datasets", "#1f77b4"),
        (axes[1], element_first, len(first_of_element), "distinct elements", "#2ca02c"),
    ]:
        axis.plot(sizes, np.searchsorted(first_seen, sizes, side="right"), linewidth=2.2, color=color)
        axis.axhline(total, ls=":", lw=1.2, color="#7f7f7f")
        axis.axvline(anchor_size, ls="--", lw=1.2, color="#d62728")
        axis.set_xscale("log")
        axis.set_xlabel("coreset size $K$", fontsize=13)
        axis.set_ylabel(f"{label} represented", fontsize=13)
        axis.set_title(f"{label}: {total} total", fontsize=12)
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/discovery_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("discovery saved", flush=True)


def plot_structure_profile():
    values = np.asarray(elements_column.values.to_numpy(zero_copy_only=False), dtype=np.int64)
    offsets = np.asarray(elements_column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    n_species = np.diff(offsets)
    edges = np.unique(np.geomspace(64, K, 60).astype(int))
    centers, size_p10, size_p50, size_p90, species_p50 = [], [], [], [], []
    start = 0
    for end in edges:
        if end <= start:
            continue
        window_sites = n_sites[start:end]
        window_species = n_species[start:end]
        centers.append(np.sqrt(max(start, 1) * end))
        size_p10.append(np.percentile(window_sites, 10))
        size_p50.append(np.percentile(window_sites, 50))
        size_p90.append(np.percentile(window_sites, 90))
        species_p50.append(np.percentile(window_species, 50))
        start = end
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].fill_between(centers, size_p10, size_p90, alpha=0.25, color="#1f77b4", label="10–90%")
    axes[0].plot(centers, size_p50, linewidth=2.2, color="#1f77b4", label="median")
    axes[0].set_ylabel("atoms per structure", fontsize=13)
    axes[1].plot(centers, species_p50, linewidth=2.2, color="#2ca02c", label="median")
    axes[1].set_ylabel("chemical species per structure", fontsize=13)
    for axis in axes:
        axis.axvline(anchor_size, ls="--", lw=1.2, color="#d62728")
        axis.set_xscale("log")
        axis.set_xlabel("pick position (windowed)", fontsize=13)
        axis.legend(fontsize=10)
    axes[0].set_yscale("log")
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/structure_profile_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("structure profile saved", flush=True)


def plot_dataset_entropy():
    sizes = np.unique(np.geomspace(64, K, 300).astype(int))
    family_codes, encoded = np.unique(families, return_inverse=True)
    counts = np.zeros(len(family_codes), dtype=np.int64)
    entropy_values, start = [], 0
    for size in sizes:
        segment = encoded[start:size]
        np.add.at(counts, segment, 1)
        start = size
        probabilities = counts[counts > 0] / size
        entropy_values.append(float(-(probabilities * np.log2(probabilities)).sum()))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(sizes, entropy_values, linewidth=2.2, color="#9467bd")
    axis.axvline(anchor_size, ls="--", lw=1.2, color="#d62728")
    axis.set_xscale("log")
    axis.set_xlabel("coreset size $K$", fontsize=13)
    axis.set_ylabel("dataset Shannon entropy (bits)", fontsize=13)
    figure.tight_layout()
    figure.savefig(f"{plots_dir}/dataset_entropy_224.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"dataset entropy saved (H at 2^24: {entropy_values[-1]:.2f} bits)", flush=True)


plot_family_decomposition()
plot_category_decomposition()
plot_element_map()
plot_mad_retention()
plot_coverage_radius()
plot_discovery()
plot_structure_profile()
plot_dataset_entropy()
print("ALL PLOTS DONE", flush=True)
