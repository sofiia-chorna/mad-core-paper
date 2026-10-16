import os
from pathlib import Path

import sys

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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures" / "coreset224-elements.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FAM_ORDER = ["hydrogen", "s-block metals", "p-block elements", "transition metals",
             "lanthanides/actinides"]
FAM_COLOR = {
    "hydrogen": epfl.ACCRETE["charcoal"],
    "s-block metals": epfl.ACCRETE["apricot"],
    "p-block elements": epfl.ACCRETE["petrol"],
    "transition metals": epfl.ACCRETE["brick"],
    "lanthanides/actinides": epfl.ACCRETE["indigo"],
}

LN_SHIFT = 0.55
X_LO, X_HI = -0.10, 18.05
Y_LO, Y_HI = 0.45, 10.05

FS_SYM = 18.0
FS_TICK = 15.5
FS_LABEL = 21.0
FS_LEGEND = 18.0
FS_CBAR = 19.0
FS_PANEL = 20.0
FS_COUNT = 13.0


def fmt(v):
    if v >= 1_000_000:
        return f"{v / 1e6:.1f}M" if v < 10_000_000 else f"{v / 1e6:.0f}M"
    return em._fmt_count(v)


def row_y(row):
    return (9 - row) + (LN_SHIFT if row >= 8 else 0.0)


def draw_ptable(ax, el_counts, cbar_in_gap=True):
    vals = list(el_counts.values())
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
            ax.text(x + 0.475, y + 0.22, fmt(v), ha="center", va="center",
                    fontsize=FS_COUNT, color=fc, alpha=0.9)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_aspect("equal")
    ax.axis("off")

    if cbar_in_gap:
        xw, yh = X_HI - X_LO, Y_HI - Y_LO
        x0, y0, w, h = 2.8, 8.30, 8.2, 0.34
        cax = ax.inset_axes([(x0 - X_LO) / xw, (y0 - Y_LO) / yh, w / xw, h / yh])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
        ticks = [min(vals), max(vals)]
        mids = [t for t in (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
                if min(vals) * 2.5 < t < max(vals) / 2.5]
        cbar.set_ticks([ticks[0]] + mids + [ticks[1]])
        cbar.set_ticklabels([fmt(v) for v in [ticks[0]] + mids + [ticks[1]]])
        cbar.ax.minorticks_off()
        cbar.set_label("# structures containing the element", fontsize=FS_CBAR,
                       labelpad=7)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=FS_CBAR - 2)
        cbar.outline.set_linewidth(0.6)
    return el_counts


def main():

    # structures containing each element over the full 2 24 coreset, from prepare_data.py
    lines = (HERE / "data" / "element_structure_counts.tsv").read_text().splitlines()
    el_counts = {line.split("\t")[0]: int(line.split("\t")[1]) for line in lines[1:]}

    fig = plt.figure(figsize=(9.9, 6.4))
    axb = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    draw_ptable(axb, el_counts)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    lo = min(el_counts, key=el_counts.get)
    hi = max(el_counts, key=el_counts.get)
    print(f"wrote {OUT}: {len(el_counts)} elements, {lo}={el_counts[lo]:,} .. "
          f"{hi}={el_counts[hi]:,}")

if __name__ == "__main__":
    sys.exit(main())
