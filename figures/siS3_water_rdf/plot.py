from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from matplotlib.ticker import MaxNLocator, MultipleLocator

import sys

# data/ holds the mean and the five seeds of every potential
HERE = Path(__file__).resolve().parent
CURVES = HERE / "data"
FIGURES = Path(sys.argv[1]).parent if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures"
NAME = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "si_rdf_water_20_50ps"
CONDITIONS = (("baseline", "PET-MAD-1.6-M", "C0"), ("lda_corrected", "LDA-corrected", "C1"), ("pbe_corrected", "PBE-corrected", "C2"))
PANELS = (("rdf_oo_20_50ps.tsv", "O–O", (2.2, 4.9)), ("rdf_oh_20_50ps.tsv", "O–H", (0.0, 4.9)), ("rdf_hh_20_50ps.tsv", "H–H", (1.0, 4.9)))
SEEDS = range(5)
OH_INSET = (1.3, 4.9)

# sizes in inches as in the learning curve figure
WIDTH = 7.0
LEFT = 0.62
RIGHT = 0.1
GAP = 0.45
TOP = 0.04 + 0.26 + 0.36
BOTTOM = 0.55
PANEL = (WIDTH - LEFT - RIGHT - 2 * GAP) / 3
PANEL_HEIGHT = PANEL / 1.35
DPI = 600
SIDE_PAD = 0.09
END_PAD = 0.03
FONT = 1.5
INSET_FONT = 1.0
LINE = 0.8
MEAN_LW = 1.6
RUN_LW = 0.5

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": LINE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def draw(ax, data, xlim, on_top=None):
    shown = (data["r_A"] >= xlim[0]) & (data["r_A"] <= xlim[1])
    top = 0.0

    for condition, label, color in CONDITIONS:
        for seed in SEEDS:
            run = data[f"{condition}_seed{seed}"]
            ax.plot(data["r_A"], run, color=color, lw=RUN_LW, alpha=0.35)
            top = max(top, run[shown].max())

        ax.plot(
            data["r_A"],
            data[f"{condition}_mean"],
            color=color,
            lw=MEAN_LW,
            label=label,
            zorder=3 if condition == on_top else 2,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.05 * top)


def style(ax, font):
    ax.tick_params(which="major", length=3.5 * font, width=LINE, labelsize=7.5 * font, pad=1.5)
    ax.tick_params(which="minor", length=2 * font, width=0.6)


def main():
    height = TOP + PANEL_HEIGHT + BOTTOM
    fig = plt.figure(figsize=(WIDTH, height))

    axes = []

    for c, (name, title, xlim) in enumerate(PANELS):
        left = LEFT + c * (PANEL + GAP)
        ax = fig.add_axes([left / WIDTH, BOTTOM / height, PANEL / WIDTH, PANEL_HEIGHT / height])
        axes.append(ax)

        data = np.genfromtxt(CURVES / name, names=True, delimiter="\t")

        # in the O-H panel the LDA corrected mean goes on top
        on_top = "lda_corrected" if title == "O–H" else None
        draw(ax, data, xlim, on_top)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        style(ax, FONT)
        ax.set_title(title, fontsize=9 * FONT)

        # zoom on the hydrogen bond region below the covalent peak
        if title == "O–H":
            inset = ax.inset_axes([0.40, 0.36, 0.56, 0.58])
            draw(inset, data, OH_INSET, on_top)
            inset.xaxis.set_major_locator(MaxNLocator(3, integer=True))
            inset.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            style(inset, FONT * INSET_FONT)

    axes[0].set_ylabel("g(r)", fontsize=8 * FONT, labelpad=2.0)
    fig.text((LEFT + 1.5 * PANEL + GAP) / WIDTH, 0.06 / height, "r (Å)", fontsize=8 * FONT, ha="center", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.04 / height),
        ncol=3,
        fontsize=8 * FONT,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.2,
    )

    # trimmed to the drawn content
    fig.canvas.draw()
    ink = fig.get_tightbbox(fig.canvas.get_renderer())
    box = Bbox.from_extents(ink.x0 - SIDE_PAD, ink.y0 - END_PAD, ink.x1 + SIDE_PAD, ink.y1 + END_PAD)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{NAME}.png", dpi=DPI, bbox_inches=box)
    fig.savefig(FIGURES / f"{NAME}.pdf", bbox_inches=box)

    smallest = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    print(f"{box.width:.2f} x {box.height:.2f} in, smallest font {smallest:.1f} pt")


if __name__ == "__main__":
    main()
