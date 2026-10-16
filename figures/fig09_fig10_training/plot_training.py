import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import label

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
PREFIX = "scratchv3-s"
POPULATION = "balanced_4000"
SIZES = ("5k", "10k", "20k")
X = [int(s[:-1]) * 1000 for s in SIZES]
ARMS = {"madcore": ("MAD-CORE", "#D55E00", "s"), "random": ("RANDOM", "#0072B2", "o")}
MAP_ARMS = ("random", "madcore")
KINDS = {"energy": ("meV/atom", 1.0), "force": ("eV/Å", 1e-3)}
METRICS = ("MAE", "RMSE", "p8", "SUP")
DRAWINGS = ("curves", "force_map", "separate_parts")
PART_TESTS = {"test_random": ("-", True, "RANDOM test"), "test_madcore": ("--", False, "MAD-CORE test")}

# page geometry in inches
WIDTH = 7.0
SPLIT_WIDTH = 6.5
BAR_TICKS = {"energy": (50, 100, 200, 500, 1000, 2000, 5000), "force": (0.2, 0.5, 1, 2, 5, 10)}
RIGHT = 0.1
MAP_LEFT = 0.42
PAIR_GAP = 0.05
QUANTITY_GAP = 0.2
MAP_WIDTH = (WIDTH - MAP_LEFT - RIGHT - 2 * PAIR_GAP - QUANTITY_GAP) / 4
LINE = 0.8
CURVE_FONT = 1.5
CURVE_MARK = 1.0

# the per-part supplementary figure spans both columns of the page
PARTS_FONT = 1.1
PARTS_DPI = 600

# space between the points and the frame of the learning-curve panels
CURVE_Y_PAD = 0.22
CURVE_X_MARGIN = 0.16

# map conventions
DENSITY_CMAP = "Greys"
ERROR_CMAP = "viridis"
SHARE_CMAP = "Blues"
CONTOUR_LEVELS = (3, 5, 7, 9)
CONTOUR_GREY = "black"
MIN_PATCH = 6

plt.rcParams.update({
    "font.size": 8,
    "axes.linewidth": LINE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def model_of(arm, size):
    return f"{PREFIX}--{arm}_{size}--s42"


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def log_padding(low, high, fraction=0.12):
    factor = (high / low) ** fraction

    return low / factor, high * factor

# the bins of the map that hold evaluated structures


def occupied_window(keep, x_edges, y_edges, margin=1):
    i, j = np.nonzero(keep)
    lo_i, hi_i = max(i.min() - margin, 0), min(i.max() + 1 + margin, len(x_edges) - 1)
    lo_j, hi_j = max(j.min() - margin, 0), min(j.max() + 1 + margin, len(y_edges) - 1)

    return (slice(lo_i, hi_i), slice(lo_j, hi_j),
            (x_edges[lo_i], x_edges[hi_i], y_edges[lo_j], y_edges[hi_j]))


def drop_specks(mask, min_size=MIN_PATCH):
    tagged, found = label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(tagged.ravel())
    keep = np.zeros(len(sizes), dtype=bool)
    keep[1:] = sizes[1:] >= min_size

    return keep[tagged]


def bare(axis, extent):
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xticks([])
    axis.set_yticks([])

# log y axis with labelled powers of ten


def log_y_ticks(ax, limits, x_margin=0.09, every_decade=False):
    ax.margins(x=x_margin)
    ax.autoscale_view()
    ax.set_ylim(*limits)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12) if every_decade else LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    low, high = limits
    decades_inside = math.floor(math.log10(high)) - math.ceil(math.log10(low)) + 1

    if decades_inside >= 1:
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** math.floor(math.log10(v)), 6) in (2.0, 3.0, 5.0) else ""))

# row labels of the first column on one vertical line


def align_row_labels(fig, pad_pt=4.0):
    axes = [ax for ax in fig.axes if ax.get_ylabel()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(label.get_window_extent(renderer).x0 for ax in axes for label in ax.yaxis.get_ticklabels(which="both") if label.get_visible() and label.get_text())

    for ax in axes:
        x = ax.transAxes.inverted().transform((left - pad_pt * fig.dpi / 72, 0))[0]
        ax.yaxis.set_label_coords(x, 0.5)

# an axes box placed in inches


def axes_at(fig, left, top, width, height):
    fig_width, fig_height = fig.get_size_inches()

    return fig.add_axes([left / fig_width, (fig_height - top - height) / fig_height, width / fig_width, height / fig_height])


def save(fig, out_dir, name, dpi=400):
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=dpi)

    fig_width, fig_height = fig.get_size_inches()
    smallest = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    plt.close(fig)
    print(f"wrote {out_dir / name}.{{png,pdf}}, {fig_width:.2f} x {fig_height:.2f} in, smallest font {smallest:.1f} pt", flush=True)


def style_curve_axis(ax, limits, show_x, font=1.0, tick_pad=3.5, x_margin=0.09, every_decade=False):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels(SIZES if show_x else [])
    log_y_ticks(ax, limits, x_margin, every_decade)
    ax.tick_params(which="major", length=3.5 * font, width=LINE, labelsize=7.5 * font, pad=tick_pad)
    ax.tick_params(which="minor", length=2 * font, width=0.6)

# rows energy and force


def draw_curves(fig, value, limits, left, top, width, height, gap, font=1.0, shared_x=False, tick_pad=3.5, label_pad=4.0, x_margin=0.09, every_decade=False, mark=None):
    for r, kind in enumerate(KINDS):
        unit, _ = KINDS[kind]

        for c, metric in enumerate(METRICS):
            ax = axes_at(fig, left + c * (width + gap), top + r * (height + 0.1), width, height)

            for arm, (name, colour, marker) in ARMS.items():
                ax.plot(X, [value[(kind, metric, name, s)] for s in SIZES], "-", color=colour, lw=1.4 * (mark or font), marker=marker, ms=4.5 * (mark or font))
            style_curve_axis(ax, limits[(kind, metric)], r == 1, font, tick_pad, x_margin, every_decade)

            if r == 0:
                ax.set_title(metric, fontsize=9 * font)

            if c == 0:
                ax.set_ylabel(f"{kind} error,\n{unit}", fontsize=8 * font, labelpad=label_pad)

            if r == 1 and not shared_x:
                ax.set_xlabel("training structures", fontsize=8 * font)

    # one x label centred under the bottom row
    if shared_x:
        fig_width, fig_height = fig.get_size_inches()
        centre = left + (4 * width + 3 * gap) / 2
        fig.text(centre / fig_width, 0.06 / fig_height, "training structures", fontsize=8 * font, ha="center", va="bottom")


def legend(fig, handles, title, centre, top, handlelength=3.0, columnspacing=2.0, font=1.0, ncol=2):
    fig_width, fig_height = fig.get_size_inches()
    fig.legend(
        handles=handles,
        title=title,
        loc="upper center",
        bbox_to_anchor=(centre, 1 - top / fig_height),
        ncol=ncol,
        fontsize=8 * font,
        title_fontsize=8 * font,
        frameon=False,
        handlelength=handlelength,
        columnspacing=columnspacing,
    )


def arm_handles(font=1.0, mark=None):
    return [plt.Line2D([], [], color=colour, marker=marker, lw=1.6 * (mark or font), ms=5 * (mark or font), label=name) for name, colour, marker in ARMS.values()]

# plotted values and the frozen y range of each panel.tsv


def curve_values(data, fraction=0.12):
    source = read_tsv(data / f"learning_curves_source_{POPULATION}.tsv")
    value = {(r["quantity"], r["metric"], r["training_set"], r["size"]): float(r["value"]) for r in source}
    limits = {(r["quantity"], r["metric"]): log_padding(float(r["data_min"]), float(r["data_max"]), fraction) for r in read_tsv(data / "learning_curve_ylimits.tsv")}

    return value, limits

# bins


def map_context(data, window):
    panels = np.load(data / f"sketchmap_panels_{POPULATION}.npz")
    info = json.loads((data / "MAPS.json").read_text())
    x_edges, y_edges = panels["x_edges"], panels["y_edges"]

    if window == "union":
        union = np.load(data / "sketchmap_panels_union_7847_round3.npz")
        assert np.array_equal(union["x_edges"], x_edges) and np.array_equal(union["y_edges"], y_edges)
        keep = union["keep"]
    else:
        keep = panels["keep"]

    rows, columns, extent = occupied_window(keep, x_edges, y_edges)
    density = panels["density"]
    body = drop_specks(np.isfinite(density) & (density > 0))
    cropped = np.where(body, density, np.nan)[rows, columns]
    contours = (
        0.5 * (x_edges[rows.start:rows.stop] + x_edges[rows.start + 1:rows.stop + 1]),
        0.5 * (y_edges[columns.start:columns.stop] + y_edges[columns.start + 1:columns.stop + 1]),
    )
    norms = {
        kind: LogNorm(vmin=info["colour_limits_meV"][kind][0] * scale, vmax=info["colour_limits_meV"][kind][1] * scale)

        for kind, (_, scale) in KINDS.items()
    }
    norms["count"] = LogNorm(vmin=info["count_limits"][0], vmax=info["count_limits"][1])

    return {"panels": panels, "rows": rows, "columns": columns, "extent": extent, "cropped": cropped, "contours": contours, "norms": norms}


def colour_map(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")

    return cmap


def finish_map(ax, ctx):
    ax.contour(*ctx["contours"], ctx["cropped"].T, levels=CONTOUR_LEVELS, colors=CONTOUR_GREY, linewidths=LINE, alpha=0.6)
    bare(ax, ctx["extent"])


def draw_error(ax, ctx, kind, arm, size):
    _, scale = KINDS[kind]
    panel = ctx["panels"][f"{model_of(arm, size)}__{kind}_mean"] * scale
    image = ax.imshow(np.ma.masked_invalid(panel).T[ctx["columns"], ctx["rows"]], origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(ERROR_CMAP), norm=ctx["norms"][kind])
    finish_map(ax, ctx)

    return image

# library structures per bin on a log scale


def draw_library(ax, ctx):
    counts = 10.0 ** ctx["cropped"]
    finite = counts[np.isfinite(counts)]
    image = ax.imshow(np.ma.masked_invalid(counts).T, origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(DENSITY_CMAP), norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
    finish_map(ax, ctx)

    return image


def draw_evaluated(ax, ctx):
    coverage = ctx["panels"]["coverage"]
    image = ax.imshow(np.ma.masked_invalid(coverage).T[ctx["columns"], ctx["rows"]], origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(SHARE_CMAP), norm=ctx["norms"]["count"])
    finish_map(ax, ctx)

    return image


def colour_bar(fig, image, left, top, width, height, label, orientation, ticks=None, pad=3.5, labelpad=4.0, font=1.0):
    bar = fig.colorbar(image, cax=axes_at(fig, left, top, width, height), orientation=orientation)
    bar.set_label(label, fontsize=8 * font, labelpad=labelpad)
    bar.ax.tick_params(labelsize=7.5 * font, width=LINE, pad=pad)

    # plain number labels inside the colour range
    if ticks is not None:
        inside = [t for t in ticks if image.norm.vmin <= t <= image.norm.vmax]
        bar.ax.minorticks_off()
        bar.set_ticks(inside)
        bar.set_ticklabels([f"{t:g}" for t in inside])

    return bar

# right edge of an axes with its tick and axis labels


def right_edge(fig, ax):
    return ax.get_tightbbox(fig.canvas.get_renderer()).x1 / fig.dpi

# room a vertical colour bar takes right of its panel


def bar_extent(ctx, bars):
    fig = plt.figure(figsize=(SPLIT_WIDTH, 3))
    ax = axes_at(fig, 0.3, 0.3, 2.0, 2.0)
    bar = colour_bar(fig, draw_library(ax, ctx), 2.36, 0.3, 0.1, 2.0, "structures / bin", "vertical", **bars)
    fig.canvas.draw()
    extent = right_edge(fig, bar.ax) - 2.3
    plt.close(fig)

    return extent

# main text figure one


def curves_figure(value, limits, out_dir, name):
    font = CURVE_FONT
    left, gap = 1.0, 0.44
    width = (SPLIT_WIDTH - left - RIGHT - 3 * gap) / 4
    height = width
    top = 0.06 + 0.68 + 0.31
    fig = plt.figure(figsize=(SPLIT_WIDTH, top + 2 * height + 0.1 + 0.55))
    draw_curves(fig, value, limits, left, top, width, height, gap, font, shared_x=True, tick_pad=1.5, label_pad=1.5, x_margin=CURVE_X_MARGIN, every_decade=True, mark=CURVE_MARK)

    # the energy and force y labels on one vertical line
    align_row_labels(fig)
    legend(fig, arm_handles(font, CURVE_MARK), "training set", 0.5, 0.04, font=font)
    save(fig, out_dir, name)

# the force maps


def map_figure(kind, ctx, out_dir, name):
    unit, _ = KINDS[kind]
    font = 1.5
    left, gap, bar_room = 0.32, 0.05, 0.8
    width = (SPLIT_WIDTH - left - RIGHT - 2 * gap - bar_room) / 3
    height = width / 1.1
    bars = {"pad": 1.5, "labelpad": 2.0, "font": font}

    # the first row spans the same width as the map columns
    columns_right = left + 3 * width + 2 * gap
    label_gap = 0.028
    context_width = (columns_right - left - bar_extent(ctx, bars) - label_gap) / 2
    context_height = context_width / 1.1
    context_top = 0.36
    maps_top = context_top + context_height + 0.34
    fig = plt.figure(figsize=(SPLIT_WIDTH, maps_top + 2 * height + gap + 0.08))

    library = axes_at(fig, left, context_top, context_width, context_height)
    density = draw_library(library, ctx)
    library.set_title("library density", fontsize=9 * font)
    colour_bar(fig, density, left + context_width + 0.06, context_top, 0.1, context_height, "structures / bin", "vertical", **bars)

    evaluated = axes_at(fig, columns_right - context_width, context_top, context_width, context_height)
    coverage = draw_evaluated(evaluated, ctx)
    evaluated.set_title("evaluated structures", fontsize=9 * font)
    colour_bar(fig, coverage, columns_right + 0.06, context_top, 0.1, context_height, "structures / bin", "vertical", **bars)

    image = None

    for r, arm in enumerate(MAP_ARMS):
        for c, size in enumerate(SIZES):
            ax = axes_at(fig, left + c * (width + gap), maps_top + r * (height + gap), width, height)
            image = draw_error(ax, ctx, kind, arm, size)

            if r == 0:
                ax.set_title(size, fontsize=9 * font)

            if c == 0:
                ax.set_ylabel(ARMS[arm][0], fontsize=8 * font, labelpad=2.0)

    colour_bar(fig, image, columns_right + 0.06, maps_top, 0.1, 2 * height + gap, f"mean {kind} error, {unit}", "vertical", BAR_TICKS[kind], **bars)
    save(fig, out_dir, name)

# learning curves on two test sets drawn separately


def separate_figure(value, tests, out_dir, name, font=1.0, mark=None, dpi=400):
    fig_width, left, gap, top, bottom, tick_pad, label_pad = WIDTH, 0.8, 0.42, 0.06 + 0.48 + 0.24, 0.48, 1.5, 1.5
    y_pad, x_margin, row_gap = CURVE_Y_PAD, CURVE_X_MARGIN, 0.16
    width = (fig_width - left - RIGHT - 3 * gap) / 4
    height = width
    fig = plt.figure(figsize=(fig_width, top + 2 * height + row_gap + bottom))
    rows = []

    for r, (kind, (unit, scale)) in enumerate(KINDS.items()):
        for c, metric in enumerate(METRICS):
            ax = axes_at(fig, left + c * (width + gap), top + r * (height + row_gap), width, height)
            shown = []

            for test, (style, filled, _) in tests.items():
                for arm, (name_arm, colour, marker) in ARMS.items():
                    y = [value(test, arm, kind, metric, s) for s in SIZES]
                    shown.extend(y)
                    rows.extend({"test": test, "training_set": name_arm, "size": s, "quantity": kind, "metric": metric, "value": v, "unit": unit}

                                for s, v in zip(SIZES, y))
                    ax.plot(X, y, style, color=colour, lw=1.4 * (mark or font), marker=marker, ms=4.5 * (mark or font), mfc=colour if filled else "white", mew=1.1 * (mark or font))
            style_curve_axis(ax, log_padding(min(shown), max(shown), y_pad), r == 1, font, tick_pad, x_margin, every_decade=True)

            if r == 0:
                ax.set_title(metric, fontsize=9 * font)

            if c == 0:
                ax.set_ylabel(f"{kind} error,\n{unit}", fontsize=8 * font, labelpad=label_pad)
    test_handles = [plt.Line2D([], [], color="0.25", ls=style, lw=1.6 * (mark or font), marker="o", ms=5 * (mark or font), mfc="0.25" if filled else "white", mew=1.1 * (mark or font), label=label)

                    for style, filled, label in tests.values()]
    align_row_labels(fig)
    fig_width, fig_height = fig.get_size_inches()
    centre = left + (4 * width + 3 * gap) / 2
    fig.text(centre / fig_width, 0.06 / fig_height, "training structures", fontsize=8 * font, ha="center", va="bottom")
    legend(fig, arm_handles(font, mark), "training set", 0.3, 0.04, handlelength=2.2, columnspacing=1.2, font=font, ncol=2)
    legend(fig, test_handles, "test set", 0.72, 0.04, handlelength=2.2, columnspacing=1.2, font=font, ncol=2)
    save(fig, out_dir, name, dpi)

    return rows

# the two 2


def separate_parts_figure(data, out_dir, name):
    table = {(r["part"], r["training_set"], r["size"]): r for r in read_tsv(data / f"metrics_by_part_{POPULATION}.tsv") if r["population"] == POPULATION}
    column = {"energy": "E_{}_meV_per_atom", "force": "F_{}_eV_per_A"}

    def value(test, arm, kind, metric, size):
        return float(table[(test, ARMS[arm][0], size)][column[kind].format(metric.lower())])

    rows = separate_figure(value, PART_TESTS, out_dir, name, font=PARTS_FONT, mark=CURVE_MARK, dpi=PARTS_DPI)

    with open(out_dir / f"{name}.tsv", "w") as fh:
        fh.write("\t".join(rows[0]) + "\n")

        for r in rows:
            fh.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    print("wrote", out_dir / f"{name}.tsv")


def main():
    parser = argparse.ArgumentParser(description="draw the training figures of the paper from data/")
    parser.add_argument("drawing", choices=DRAWINGS, help="curves (Fig. 9), force_map (Fig. 10) or separate_parts (SI)")
    parser.add_argument("output", nargs="?", help="output PDF; a PNG is written next to it (default build/figures/<standard name>.pdf)")
    parser.add_argument("--data", default=str(HERE / "data"), help="folder with the TSV, JSON and npz inputs")
    parser.add_argument("--window", choices=("union", "balanced"), default="union", help="crop window of the maps, the submitted figure uses the union of the two full tests")
    args = parser.parse_args()
    data = Path(args.data)
    standard = {"curves": f"train_curves_{POPULATION}", "force_map": f"train_force_map_{POPULATION}", "separate_parts": "si_train_curves_balanced_parts"}

    if args.output:
        out_dir, name = Path(args.output).parent, Path(args.output).stem
    else:
        out_dir, name = HERE.parent.parent / "build" / "figures", standard[args.drawing]

    if args.drawing == "curves":
        value, _ = curve_values(data)
        curves_figure(value, curve_values(data, CURVE_Y_PAD)[1], out_dir, name)

    if args.drawing == "force_map":
        map_figure("force", map_context(data, args.window), out_dir, name)

    if args.drawing == "separate_parts":
        separate_parts_figure(data, out_dir, name)


if __name__ == "__main__":
    main()
