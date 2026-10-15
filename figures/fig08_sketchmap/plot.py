"""Sketch-map figure of MAD-CORE (fig:smap), drawn from data/smap_points.npz.

The data file is written by prepare_data.py from the released MAD-CORE files.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
NAME = "voronoi_groups24_d8_p20d8"

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(HERE / "data" / "smap_points.npz"))
parser.add_argument("output", nargs="?", default=str(HERE.parent.parent / "build" / "figures" / f"{NAME}.pdf"))
parser.add_argument("--columns", type=int, default=3)
parser.add_argument("--width", type=float, default=7.0)
parser.add_argument("--height-scale", type=float, default=0.88)
parser.add_argument("--reference", type=float, nargs="*",
                    default=[1e2, 1e4, 1e6, 1e8])
parser.add_argument("--alpha", type=float, default=0.45)
parser.add_argument("--edge-width", type=float, default=0.12)
args = parser.parse_args()

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
data = np.load(args.data)
panels = json.loads(str(data["panels"]))
colours = data["colours"].astype(str)
background = data["background_xy"]
low, high = data["low"], data["high"]
population_max = float(data["population_max"])
smin, smax, size_power = float(data["smin"]), float(data["smax"]), float(data["size_power"])

n_panels = len(panels)
rows = int(np.ceil(n_panels / args.columns))

MARGIN = 0.10
figure, axes = plt.subplots(rows, args.columns,
                            figsize=(args.width, args.width / args.columns * rows * args.height_scale),
                            sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()

for index, panel in enumerate(panels):
    axis = axes[index]
    axis.scatter(background[:, 0], background[:, 1], s=0.7,
                 c="0.80", linewidths=0, rasterized=True)

    xy = data[f"panel{index}_xy"]
    axis.scatter(xy[:, 0], xy[:, 1],
                 s=data[f"panel{index}_size"], color=colours[data[f"panel{index}_colour"]].astype(object),
                 alpha=args.alpha, edgecolors="black", linewidths=args.edge_width,
                 rasterized=True)

    axis.text(0.04, 0.96, panel["title"], transform=axis.transAxes, ha="left", va="top",
              fontsize=10.0, fontweight="normal")
    dataset_legend_loc = "center right" if index == 0 else "lower right"
    axis.legend(handles=[Patch(facecolor=colour, edgecolor="0.15", linewidth=0.4, label=label)
                         for label, colour in panel["entries"]],
                loc=dataset_legend_loc, frameon=False, fontsize=9.0, handlelength=0.9,
                handletextpad=0.35, labelspacing=0.35, borderpad=0.2,
                borderaxespad=0.3)

    if index == 0:
        reference_size = smin + (smax - smin) * (
            np.log10(args.reference) / np.log10(population_max)) ** size_power
        keys_legend = [Line2D([], [], marker="o", ls="", color="0.35",
                              ms=np.sqrt(size),
                              label=f"$10^{{{int(np.log10(value))}}}$")
                       for value, size in zip(args.reference, reference_size)]
        size_legend = axis.legend(handles=keys_legend, loc="lower left",
                                  frameon=False, fontsize=8.5, ncol=2,
                                  columnspacing=0.6, handletextpad=0.3,
                                  borderaxespad=0.4, labelspacing=0.5,
                                  title="structures represented")
        size_legend.get_title().set_fontsize(8.5)
        axis.add_artist(size_legend)

    span = high - low
    axis.set_xlim(low[0] - MARGIN * span[0], high[0] + MARGIN * span[0])
    axis.set_ylim(low[1] - MARGIN * span[1], high[1] + 0.22 * span[1])
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("black")

for axis in axes[n_panels:]:
    axis.set_visible(False)

for column in range(args.columns):
    lowest = [i for i in range(n_panels) if i % args.columns == column]
    if lowest:
        axes[lowest[-1]].set_xlabel("SMAP 1", fontsize=10.0, labelpad=2.5)
for index in range(0, n_panels, args.columns):
    axes[index].set_ylabel("SMAP 2", fontsize=10.0, labelpad=2.5)

figure.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.2)
figure.subplots_adjust(wspace=0.015, hspace=0.025)
figure.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", args.output)
