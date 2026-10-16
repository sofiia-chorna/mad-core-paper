import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DATA = HERE / "data" / "subset_counts.tsv"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures" / "mad-kset-1col.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

ORDER = [
    "mad15:binary_random", "mad15:mc3d", "mad15:mc3d_ext", "mad15:mc3d_random",
    "mad15:mc3d_random_ext", "mad15:mc3d_rattled", "mad15:mc3d_cluster",
    "mad15:mc3d_surface", "mad15:mc2d", "mad15:shiftml_molcrys",
    "mad15:shiftml_molfrags", "mad15:monomers", "mad15:dimers", "mad15:trimers",
    "mad_next:flashmd_3d", "mad_next:flashmd_2d", "mad_next:flashmd_isolated",
    "mad_next:newtrimers", "mad_next:dimers", "mad_next:trimers", "mad_next:madcat",
    "mad_sol:solvated_ions", "mad_sol:solvents_soup", "mad_sol:solvents_interface",
]

DISPLAY = {
    "mad15:binary_random": "Binary-random", "mad15:mc3d": "MC3D",
    "mad15:mc3d_ext": "MC3D-ext.", "mad15:mc3d_random": "MC3D-random",
    "mad15:mc3d_random_ext": "MC3D-random-ext.", "mad15:mc3d_rattled": "MC3D-rattled",
    "mad15:mc3d_cluster": "MC3D-cluster", "mad15:mc3d_surface": "MC3D-surface",
    "mad15:mc2d": "MC2D", "mad15:shiftml_molcrys": "SHIFTML-molcrys",
    "mad15:shiftml_molfrags": "SHIFTML-molfrags", "mad15:monomers": "Monomers",
    "mad15:dimers": "Dimers", "mad15:trimers": "Trimers",
    "mad_next:flashmd_3d": "FlashMD-3D", "mad_next:flashmd_2d": "FlashMD-2D",
    "mad_next:flashmd_isolated": "FlashMD-isolated", "mad_next:newtrimers": "New-trimers",
    "mad_next:dimers": "Moldimers", "mad_next:trimers": "Moltrimers",
    "mad_next:madcat": "MAD-cat", "mad_sol:solvated_ions": "Solvated ions",
    "mad_sol:solvents_soup": "Solvents soup", "mad_sol:solvents_interface": "Solvents interface",
}

COLOR = {
    "mad15:binary_random": "#B22222", "mad15:mc3d": "#D1493B", "mad15:mc3d_ext": "#E07856",
    "mad15:mc3d_random": "#EF9838", "mad15:mc3d_random_ext": "#F5C542",
    "mad15:mc3d_rattled": "#7A3B2E", "mad15:mc3d_cluster": "#B08343",
    "mad15:mc3d_surface": "#E7B9A6", "mad15:mc2d": "#4C4CA6",
    "mad15:shiftml_molcrys": "#178F82", "mad15:shiftml_molfrags": "#63C3B9",
    "mad15:monomers": "#DDD5C8", "mad15:dimers": "#9B968E", "mad15:trimers": "#5E5852",
    "mad_next:flashmd_3d": "#2E6DB4", "mad_next:flashmd_2d": "#6BA6E0",
    "mad_next:flashmd_isolated": "#AECDEC", "mad_next:newtrimers": "#1F8043",
    "mad_next:dimers": "#4CAF6E", "mad_next:trimers": "#9BD69A", "mad_next:madcat": "#4E6B78",
    "mad_sol:solvated_ions": "#7A3FC0", "mad_sol:solvents_soup": "#C72E82",
    "mad_sol:solvents_interface": "#EE9AC6",
}


def load():
    lines = DATA.read_text().splitlines()
    names = lines[0].split("\t")[1:]
    rows = [[int(v) for v in line.split("\t")] for line in lines[1:]]
    ks = np.asarray([row[0] for row in rows])
    counts = {name: np.asarray([row[i + 1] for row in rows]) for i, name in enumerate(names)}
    return counts, ks


def stacked(ax, keys, counts, colors, ks, xpos=None):
    x = np.arange(len(ks), dtype=float) if xpos is None else xpos
    bottom = np.zeros(len(ks))
    for k in keys:
        row = np.asarray([counts[k][i] / kk for i, kk in enumerate(ks)])
        ax.bar(x, row, bottom=bottom, width=0.90, color=colors[k], linewidth=0)
        bottom += row
    return x


def xtick(ax, ks, x, every=1):
    labels = [f"$2^{{{i + 1}}}$" for i in range(len(ks) - 1)] + ["full"]
    if every > 1:
        labels = [lab if (i % every == 0 or i == len(ks) - 1) else "" for i, lab in enumerate(labels)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.margins(x=0.0)
    ax.set_ylim(0, 1)


def plot(counts, ks):
    fig = plt.figure(figsize=(9.9, 7.4))
    ax = fig.add_axes([0.088, 0.48, 0.897, 0.50])
    x = stacked(ax, ORDER, counts, COLOR, ks)
    xtick(ax, ks, x, every=1)
    ax.tick_params(axis="x", labelsize=17.5, pad=8, length=0)
    ax.tick_params(axis="y", labelsize=17.5)
    ax.set_xlabel("$k$-set size", fontsize=21)
    ax.set_ylabel("fraction of $k$-set", fontsize=21)

    h = [Patch(facecolor=COLOR[s], edgecolor="0.15", linewidth=0.4) for s in ORDER]
    leg = fig.legend(h, [DISPLAY[s] for s in ORDER], loc="upper center",
                     bbox_to_anchor=(0.04, 0.37, 0.92, 0.0), mode="expand",
                     ncol=3, fontsize=18, frameon=False, handlelength=1.0,
                     handletextpad=0.4, labelspacing=0.36, columnspacing=1.0)
    leg._legend_box.align = "left"
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    counts, ks = load()
    plot(counts, ks)
    print("wrote mad-kset-1col.pdf")

if __name__ == "__main__":
    main()
