import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "plotting"))
import element_map as em
import epfl

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures" / "mad-kset-counts.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FAM_ORDER = ["hydrogen", "s-block metals", "p-block elements", "transition metals",
             "lanthanides/actinides"]

# accrete.pdf palette, ordered so no two adjacent stack segments share a hue
FAM_COLOR = {
    "hydrogen": epfl.ACCRETE["charcoal"],
    "s-block metals": epfl.ACCRETE["apricot"],
    "p-block elements": epfl.ACCRETE["petrol"],
    "transition metals": epfl.ACCRETE["brick"],
    "lanthanides/actinides": epfl.ACCRETE["indigo"],
}

# lanthanide/actinide rows pulled up (PT_LAYOUT parks them at rows 8-9, leaving a wide gap)
LN_SHIFT = 0.55
X_LO, X_HI = -0.10, 18.05
Y_LO, Y_HI = 0.45, 10.05

FS_SYM = 16.5  # element symbol
FS_TICK = 15.5  # k-set tick labels
FS_LABEL = 21.0  # axis labels
FS_LEGEND = 18.0
FS_CBAR = 17.0
FS_PANEL = 20.0
FS_COUNT = 12.0  # per-cell count, short "16k" form survives the downscale


def row_y(row):
    return (9 - row) + (LN_SHIFT if row >= 8 else 0.0)


def read_tsv(name):
    lines = (HERE / "data" / name).read_text().splitlines()
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def load():
    header, rows = read_tsv("element_family_counts.tsv")
    assert header[1:] == FAM_ORDER, header
    ks = [int(row[0]) for row in rows]
    cum = np.asarray([[int(v) for v in row[1:]] for row in rows])
    _, rows = read_tsv("element_structure_counts.tsv")
    el_counts = {row[0]: int(row[1]) for row in rows}
    return ks, cum, el_counts


def draw_ptable(ax, el_counts):
    vals = [v for v in el_counts.values() if v > 0]

    # not the palette of panel a, a perceptual ramp reads as a scale
    cmap = plt.colormaps["viridis"]
    norm = mcolors.LogNorm(vmin=min(vals), vmax=max(vals))

    for el, (row, col) in em.PT_LAYOUT.items():
        x, y = col, row_y(row)
        v = el_counts.get(el)
        rgba = cmap(norm(v)) if v else (0.91, 0.91, 0.91, 1.0)
        ax.add_patch(plt.Rectangle((x, y), 0.95, 0.95, facecolor=rgba,
                                   edgecolor="white", linewidth=0.7))
        fc = em._text_color(rgba) if v else "#aaaaaa"
        ax.text(x + 0.475, y + 0.60, el, ha="center", va="center", fontsize=FS_SYM,
                fontweight="bold", color=fc)
        if v:
            ax.text(x + 0.475, y + 0.22, em._fmt_count(v), ha="center", va="center",
                    fontsize=FS_COUNT, color=fc, alpha=0.9)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.axis("off")

    # colorbar into the empty top-middle block (label above, ticks below)
    xw, yh = X_HI - X_LO, Y_HI - Y_LO
    x0, y0, w, h = 3.2, 8.30, 8.2, 0.34
    cax = ax.inset_axes([(x0 - X_LO) / xw, (y0 - Y_LO) / yh, w / xw, h / yh])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [min(vals), 100_000, max(vals)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v / 1000:.1f}k" if v < 10_000 else em._fmt_count(v)
                         for v in ticks])
    cbar.ax.minorticks_off()
    cbar.set_label("# structures containing the element", fontsize=FS_CBAR, labelpad=7)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(labelsize=FS_CBAR - 2)
    cbar.outline.set_linewidth(0.6)


def main():

    # panel a, each distinct element of a structure contributes one count to its family
    ks, cum, el_counts = load()
    fam_idx = {f: i for i, f in enumerate(FAM_ORDER)}

    fig = plt.figure(figsize=(9.9, 10.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.30], hspace=0.06,
                          left=0.085, right=0.995, top=0.925, bottom=0.005)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ---- panel a
    x = np.arange(len(ks), dtype=float)
    bottom = np.zeros(len(ks))
    for f in FAM_ORDER:
        row = np.asarray([cum[i, fam_idx[f]] / cum[i].sum() for i in range(len(ks))])
        axa.bar(x, row, bottom=bottom, width=0.90, color=FAM_COLOR[f], linewidth=0)
        bottom += row
    axa.set_xticks(x)
    axa.set_xticklabels([f"$2^{{{i + 1}}}$" for i in range(len(ks) - 1)] + ["full"],
                        fontsize=FS_TICK)
    axa.tick_params(axis="x", pad=6, length=0)
    axa.tick_params(axis="y", labelsize=FS_TICK)
    axa.set_xlabel("$k$-set size", fontsize=FS_LABEL)
    axa.set_ylabel("fraction of element counts", fontsize=FS_LABEL)
    axa.set_ylim(0, 1)
    axa.margins(x=0.0)
    h = [Patch(facecolor=FAM_COLOR[f], edgecolor="0.15", linewidth=0.4) for f in FAM_ORDER]
    axa.legend(h, FAM_ORDER, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
               fontsize=FS_LEGEND, frameon=False, handlelength=1.1, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.4)
    axa.text(-0.075, 1.20, "a", transform=axa.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

    # ---- panel b
    draw_ptable(axb, el_counts)
    axb.text(-0.075, 1.0, "b", transform=axb.transAxes, fontsize=FS_PANEL,
             fontweight="bold", va="top", ha="left")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    lo, hi = min(el_counts.values()), max(el_counts.values())
    print(f"wrote {OUT}")
    print(f"panel b: full ordering, {len(el_counts)} elements, {lo:,}-{hi:,}")

if __name__ == "__main__":
    sys.exit(main())
