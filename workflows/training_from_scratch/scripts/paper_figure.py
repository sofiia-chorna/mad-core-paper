import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator  # noqa: E402

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/scripts")
from maps_common import P  # noqa: E402

S = Path(__file__).resolve().parent.parent
OUT = S / "reports" / "populations"
PAPER = OUT / "paper"
SEPARATE = S / "reports" / "selected_checkpoints" / "metrics_mtt.tsv"
PARTS = OUT / "balanced_4000" / "tables" / "metrics_by_part_balanced_4000.tsv"
LIMITS = PAPER / "learning_curve_ylimits.tsv"
PREFIX = "scratchv3-s"
POPULATIONS = ("union_7847", "balanced_4000")
SIZES = ("5k", "10k", "20k")
X = [int(s[:-1]) * 1000 for s in SIZES]
ARMS = {"madcore": ("MAD-CORE", "#D55E00", "s"), "random": ("RANDOM", "#0072B2", "o")}
MAP_ARMS = ("random", "madcore")
MAP_COLUMNS = (("energy", "random"), ("energy", "madcore"), ("force", "random"), ("force", "madcore"))
KINDS = {"energy": ("meV/atom", 1.0), "force": ("eV/Å", 1e-3)}
METRICS = ("MAE", "RMSE", "p8", "SUP")
DRAWINGS = ("combined", "curves", "force_map", "energy_map", "separate_tests", "separate_parts")
TESTS = {"test_random": ("-", True, "RANDOM test set"), "test_madcore": ("--", False, "MAD-CORE test set")}
PART_TESTS = {"test_random": ("-", True, "RANDOM test"), "test_madcore": ("--", False, "MAD-CORE test")}

# page geometry in inches, the combined figure is 7 in wide and the split figures 6.5 in
WIDTH = 7.0
SPLIT_WIDTH = 6.5
TEXT_WIDTHS = (6.38, 7.0)
BAR_TICKS = {"energy": (50, 100, 200, 500, 1000, 2000, 5000), "force": (0.2, 0.5, 1, 2, 5, 10)}
RIGHT = 0.1
CURVE_LEFT = 0.62
CURVE_GAP = 0.42
CURVE_WIDTH = (WIDTH - CURVE_LEFT - RIGHT - 3 * CURVE_GAP) / 4
CURVE_HEIGHT = 0.95
MAP_LEFT = 0.42
PAIR_GAP = 0.05
QUANTITY_GAP = 0.2
MAP_WIDTH = (WIDTH - MAP_LEFT - RIGHT - 2 * PAIR_GAP - QUANTITY_GAP) / 4
MAP_ASPECT = 1.256
MAP_HEIGHT = MAP_WIDTH / MAP_ASPECT
LINE = 0.8
CURVE_FONT = 1.5
CURVE_MARK = 1.0

# the per-part supplementary figure spans both columns of the page
PARTS_FONT = 1.1
PARTS_DPI = 600

# space between the points and the frame of the learning-curve panels, as fractions of the log ranges
CURVE_Y_PAD = 0.22
CURVE_X_MARGIN = 0.16

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


# log y axis with labelled powers of ten, a panel without a power of ten gets plain labels at 2, 3 and 5 times a power of ten
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


# row labels of the first column on one vertical line, just left of the widest tick label of that column
def align_row_labels(fig, pad_pt=4.0):
    axes = [ax for ax in fig.axes if ax.get_ylabel()]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(label.get_window_extent(renderer).x0 for ax in axes for label in ax.yaxis.get_ticklabels(which="both") if label.get_visible() and label.get_text())
    for ax in axes:
        x = ax.transAxes.inverted().transform((left - pad_pt * fig.dpi / 72, 0))[0]
        ax.yaxis.set_label_coords(x, 0.5)


# an axes box placed in inches, measured from the top left corner of the figure
def axes_at(fig, left, top, width, height):
    fig_width, fig_height = fig.get_size_inches()
    return fig.add_axes([left / fig_width, (fig_height - top - height) / fig_height, width / fig_width, height / fig_height])


def letter(fig, left, top, text):
    fig_width, fig_height = fig.get_size_inches()
    fig.text(left / fig_width, 1 - top / fig_height, text, fontsize=10, fontweight="bold", ha="left", va="top")


def save(fig, name, sizes_log, dpi=400):
    for suffix in ("png", "pdf"):
        fig.savefig(PAPER / f"{name}.{suffix}", dpi=dpi)

    # smallest text of the figure as printed at the candidate text widths
    fig_width, fig_height = fig.get_size_inches()
    smallest = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())
    printed = {w: smallest * w / fig_width for w in TEXT_WIDTHS}
    sizes_log.append({"figure": name, "width_in": f"{fig_width:.2f}", "height_in": f"{fig_height:.2f}", "smallest_font_pt": f"{smallest:.1f}",
                      **{f"printed_pt_at_{w}in": f"{v:.1f}" for w, v in printed.items()}})
    plt.close(fig)
    print(f"wrote {PAPER / name}.{{png,pdf}}, {fig_width:.2f} x {fig_height:.2f} in, smallest font {smallest:.1f} pt", flush=True)


def style_curve_axis(ax, limits, show_x, font=1.0, tick_pad=3.5, x_margin=0.09, every_decade=False):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels(SIZES if show_x else [])
    log_y_ticks(ax, limits, x_margin, every_decade)
    ax.tick_params(which="major", length=3.5 * font, width=LINE, labelsize=7.5 * font, pad=tick_pad)
    ax.tick_params(which="minor", length=2 * font, width=0.6)


# rows energy and force, columns the four metrics, one line per training set, font scales text, lines and markers
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


# plotted values of one version, and the y range of each panel from the data range of both versions frozen in LIMITS
def curve_values(population, fraction=0.12):
    source = read_tsv(OUT / population / "tables" / f"learning_curves_source_{population}.tsv")
    value = {(r["quantity"], r["metric"], r["training_set"], r["size"]): float(r["value"]) for r in source}
    limits = {(r["quantity"], r["metric"]): log_padding(float(r["data_min"]), float(r["data_max"]), fraction) for r in read_tsv(LIMITS)}
    return value, limits


# bins, crop window, density contours and colour scales shared by every map
def map_context(population):
    panels = np.load(OUT / population / "tables" / f"sketchmap_panels_{population}.npz")
    union = np.load(OUT / "union_7847" / "tables" / "sketchmap_panels_union_7847.npz")
    info = json.loads((OUT / "MAPS.json").read_text())
    x_edges, y_edges = panels["x_edges"], panels["y_edges"]
    rows, columns, extent = P.occupied_window(union["keep"], x_edges, y_edges)
    density = panels["density"]
    body = P.drop_specks(np.isfinite(density) & (density > 0))
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
    ax.contour(*ctx["contours"], ctx["cropped"].T, levels=P.CONTOUR_LEVELS, colors=P.CONTOUR_GREY, linewidths=LINE, alpha=0.6)
    P.bare(ax, ctx["extent"])


def draw_error(ax, ctx, kind, arm, size):
    _, scale = KINDS[kind]
    panel = ctx["panels"][f"{model_of(arm, size)}__{kind}_mean"] * scale
    image = ax.imshow(np.ma.masked_invalid(panel).T[ctx["columns"], ctx["rows"]], origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(P.ERROR_CMAP), norm=ctx["norms"][kind])
    finish_map(ax, ctx)
    return image


# library structures per bin on a log scale, the same image as the log10 density on a linear scale
def draw_library(ax, ctx):
    counts = 10.0 ** ctx["cropped"]
    finite = counts[np.isfinite(counts)]
    image = ax.imshow(np.ma.masked_invalid(counts).T, origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(P.DENSITY_CMAP), norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
    finish_map(ax, ctx)
    return image


def draw_evaluated(ax, ctx):
    coverage = ctx["panels"]["coverage"]
    image = ax.imshow(np.ma.masked_invalid(coverage).T[ctx["columns"], ctx["rows"]], origin="lower", extent=ctx["extent"], aspect="auto", cmap=colour_map(P.SHARE_CMAP), norm=ctx["norms"]["count"])
    finish_map(ax, ctx)
    return image


def colour_bar(fig, image, left, top, width, height, label, orientation, ticks=None, pad=3.5, labelpad=4.0, font=1.0):
    bar = fig.colorbar(image, cax=axes_at(fig, left, top, width, height), orientation=orientation)
    bar.set_label(label, fontsize=8 * font, labelpad=labelpad)
    bar.ax.tick_params(labelsize=7.5 * font, width=LINE, pad=pad)

    # plain number labels inside the colour range instead of powers of ten only
    if ticks is not None:
        inside = [t for t in ticks if image.norm.vmin <= t <= image.norm.vmax]
        bar.ax.minorticks_off()
        bar.set_ticks(inside)
        bar.set_ticklabels([f"{t:g}" for t in inside])
    return bar


# right edge of an axes with its tick and axis labels, in inches
def right_edge(fig, ax):
    return ax.get_tightbbox(fig.canvas.get_renderer()).x1 / fig.dpi


# room a vertical library colour bar with its tick and axis labels takes right of its panel, measured on a scratch figure
def bar_extent(ctx, bars):
    fig = plt.figure(figsize=(SPLIT_WIDTH, 3))
    ax = axes_at(fig, 0.3, 0.3, 2.0, 2.0)
    bar = colour_bar(fig, draw_library(ax, ctx), 2.36, 0.3, 0.1, 2.0, "structures / bin", "vertical", **bars)
    fig.canvas.draw()
    extent = right_edge(fig, bar.ax) - 2.3
    plt.close(fig)
    return extent


def map_left(column):
    pair, inside = divmod(column, 2)
    return MAP_LEFT + column * MAP_WIDTH + inside * PAIR_GAP + pair * (PAIR_GAP + QUANTITY_GAP)


# the earlier single page version, kept as an alternative
def combined_figure(population, value, limits, ctx, sizes_log):
    curve_top = 0.06 + 0.52 + 0.2
    curve_bottom = curve_top + 2 * CURVE_HEIGHT + 0.1
    context_top = curve_bottom + 0.42 + 0.12 + 0.2
    maps_top = context_top + MAP_HEIGHT + 0.15 + 0.42
    bar_top = maps_top + 3 * MAP_HEIGHT + 2 * PAIR_GAP + 0.08
    fig = plt.figure(figsize=(WIDTH, bar_top + 0.09 + 0.48))

    draw_curves(fig, value, limits, CURVE_LEFT, curve_top, CURVE_WIDTH, CURVE_HEIGHT, CURVE_GAP)
    legend(fig, arm_handles(), "training set", 0.5, 0.04)
    letter(fig, 0.05, 0.08, "a")

    library = axes_at(fig, map_left(1), context_top, MAP_WIDTH, MAP_HEIGHT)
    draw_library(library, ctx)
    library.set_title("library density", fontsize=8)
    evaluated = axes_at(fig, map_left(2), context_top, MAP_WIDTH, MAP_HEIGHT)
    coverage = draw_evaluated(evaluated, ctx)
    evaluated.set_title("evaluated structures", fontsize=8)
    colour_bar(fig, coverage, map_left(2) + MAP_WIDTH + 0.08, context_top, 0.08, MAP_HEIGHT, "structures / bin", "vertical")
    letter(fig, 0.05, context_top - 0.22, "b")

    images = {}
    for c, (kind, arm) in enumerate(MAP_COLUMNS):
        for r, size in enumerate(SIZES):
            ax = axes_at(fig, map_left(c), maps_top + r * (MAP_HEIGHT + PAIR_GAP), MAP_WIDTH, MAP_HEIGHT)
            images[kind] = draw_error(ax, ctx, kind, arm, size)
            if r == 0:
                ax.set_title(ARMS[arm][0], fontsize=8)
            if c == 0:
                ax.set_ylabel(size, fontsize=8)
    fig_width, fig_height = fig.get_size_inches()
    for pair, kind in enumerate(KINDS):
        unit, _ = KINDS[kind]
        centre = (map_left(2 * pair) + map_left(2 * pair + 1) + MAP_WIDTH) / 2
        fig.text(centre / fig_width, 1 - (maps_top - 0.24) / fig_height, f"{kind} error", fontsize=9, ha="center", va="bottom")
        colour_bar(fig, images[kind], map_left(2 * pair), bar_top, 2 * MAP_WIDTH + PAIR_GAP, 0.09, f"mean {kind} error, {unit}", "horizontal")
    letter(fig, 0.05, maps_top - 0.38, "c")
    save(fig, f"mad-core_train_{population}", sizes_log)


# main text figure one, the pooled learning curves at the split width
def curves_figure(population, value, limits, sizes_log):
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
    save(fig, f"train_curves_{population}", sizes_log)


# main text figure two (forces) and its supplementary twin (energy): a first row with the library density and the
# evaluated structures, each with its colour bar, then rows training sets and columns training sizes with one error
# colour bar for both rows, fonts as in the learning curves
def map_figure(population, kind, ctx, name, sizes_log):
    unit, _ = KINDS[kind]
    font = 1.5
    left, gap, bar_room = 0.32, 0.05, 0.8
    width = (SPLIT_WIDTH - left - RIGHT - 2 * gap - bar_room) / 3
    height = width / 1.1
    bars = {"pad": 1.5, "labelpad": 2.0, "font": font}

    # the first row spans the same width as the map columns, the gap after the first colour bar label is measured from
    # the label box, which has some white space beyond the ink
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
    save(fig, name, sizes_log)


# supplementary learning curves on each frozen test separately, both tests in every panel
# learning curves on two test sets drawn separately, value(test, arm, kind, metric, size) in the units of KINDS,
# font scales text, lines and markers, shared_x puts one x label under the bottom row, puts each legend in one row and uses the two-column width
def separate_figure(value, tests, name, sizes_log, font=1.0, shared_x=False, mark=None, dpi=400):
    if shared_x:
        fig_width, left, gap, height, top, bottom, tick_pad, label_pad = WIDTH, 0.8, 0.42, None, 0.06 + 0.48 + 0.24, 0.48, 1.5, 1.5
        y_pad, x_margin, row_gap = CURVE_Y_PAD, CURVE_X_MARGIN, 0.16
    else:
        fig_width, left, gap, height, top, bottom, tick_pad, label_pad = SPLIT_WIDTH, 0.75, 0.4, 1.0, 0.06 + 0.46 + 0.2, 0.45, 3.5, 4.0
        y_pad, x_margin, row_gap = 0.12, 0.09, 0.1
    width = (fig_width - left - RIGHT - 3 * gap) / 4

    # the shared-x layout uses square panels
    if height is None:
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
            style_curve_axis(ax, log_padding(min(shown), max(shown), y_pad), r == 1, font, tick_pad, x_margin, every_decade=shared_x)
            if r == 0:
                ax.set_title(metric, fontsize=9 * font)
            if c == 0:
                ax.set_ylabel(f"{kind} error,\n{unit}", fontsize=8 * font, labelpad=label_pad)
            if r == 1 and not shared_x:
                ax.set_xlabel("training structures", fontsize=8 * font)
    test_handles = [plt.Line2D([], [], color="0.25", ls=style, lw=1.6 * (mark or font), marker="o", ms=5 * (mark or font), mfc="0.25" if filled else "white", mew=1.1 * (mark or font), label=label)
                    for style, filled, label in tests.values()]
    if shared_x:
        align_row_labels(fig)
        fig_width, fig_height = fig.get_size_inches()
        centre = left + (4 * width + 3 * gap) / 2
        fig.text(centre / fig_width, 0.06 / fig_height, "training structures", fontsize=8 * font, ha="center", va="bottom")
        legend(fig, arm_handles(font, mark), "training set", 0.3, 0.04, handlelength=2.2, columnspacing=1.2, font=font, ncol=2)
        legend(fig, test_handles, "test set", 0.72, 0.04, handlelength=2.2, columnspacing=1.2, font=font, ncol=2)
    else:
        legend(fig, arm_handles(), "training set", 0.27, 0.04, handlelength=2.2, columnspacing=1.2)
        legend(fig, test_handles, "test set", 0.72, 0.04, handlelength=2.2, columnspacing=1.2)
    save(fig, name, sizes_log, dpi)
    return rows


# the full RANDOM test (5,847) and the MAD-CORE test (2,000), from metrics_mtt.tsv in meV/atom and meV/A
def separate_tests_figure(sizes_log):
    table = {(r["test"], r["model"]): r for r in read_tsv(SEPARATE)}
    column = {"energy": "E", "force": "F"}

    def value(test, arm, kind, metric, size):
        return float(table[(test, model_of(arm, size))][f"{column[kind]}_{metric.lower()}"]) * KINDS[kind][1]

    separate_figure(value, TESTS, "si_train_curves_separate_tests", sizes_log)


# the two 2,000-structure parts of balanced_4000, from the per-part table in meV/atom and eV/A
def separate_parts_figure(sizes_log):
    table = {(r["part"], r["training_set"], r["size"]): r for r in read_tsv(PARTS) if r["population"] == "balanced_4000"}
    column = {"energy": "E_{}_meV_per_atom", "force": "F_{}_eV_per_A"}

    def value(test, arm, kind, metric, size):
        return float(table[(test, ARMS[arm][0], size)][column[kind].format(metric.lower())])

    rows = separate_figure(value, PART_TESTS, "si_train_curves_balanced_parts", sizes_log, font=PARTS_FONT, shared_x=True, mark=CURVE_MARK, dpi=PARTS_DPI)
    with open(PAPER / "si_train_curves_balanced_parts.tsv", "w") as fh:
        fh.write("\t".join(rows[0]) + "\n")
        for r in rows:
            fh.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
    print("wrote", PAPER / "si_train_curves_balanced_parts.tsv")


# usage: paper_figure.py <population> [combined curves force_map energy_map separate_tests separate_parts], all figures by default
def main():
    population = sys.argv[1]
    chosen = sys.argv[2:] or list(DRAWINGS)
    assert population in POPULATIONS and set(chosen) <= set(DRAWINGS), sys.argv
    PAPER.mkdir(parents=True, exist_ok=True)
    sizes_log = []

    # curve and map inputs are read only for the drawings that use them
    if {"combined", "curves"} & set(chosen):
        value, limits = curve_values(population)
    if {"combined", "force_map", "energy_map"} & set(chosen):
        ctx = map_context(population)

    if "combined" in chosen:
        combined_figure(population, value, limits, ctx, sizes_log)
    if "curves" in chosen:
        curves_figure(population, value, curve_values(population, CURVE_Y_PAD)[1], sizes_log)
    if "force_map" in chosen:
        map_figure(population, "force", ctx, f"train_force_map_{population}", sizes_log)
    if "energy_map" in chosen:
        map_figure(population, "energy", ctx, f"si_train_energy_map_{population}", sizes_log)
    if "separate_tests" in chosen:
        separate_tests_figure(sizes_log)
    if "separate_parts" in chosen:
        separate_parts_figure(sizes_log)

    # rows of the figures drawn now replace their earlier rows
    path = PAPER / f"readability_{population}.tsv"
    rows = {r["figure"]: r for r in read_tsv(path)} if path.exists() else {}
    rows.update({r["figure"]: r for r in sizes_log})
    with open(path, "w") as fh:
        columns = list(sizes_log[0].keys())
        fh.write("\t".join(columns) + "\n")
        for row in rows.values():
            fh.write("\t".join(row[c] for c in columns) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
