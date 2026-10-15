import json

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter, label

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from common_smap import FIGURES, LOGS, MAIN_MODELS, REPORTS, make_logger  # noqa: E402

DENSITY_CMAP = "Greys"
ERROR_CMAP = "viridis"
SHARE_CMAP = "Blues"
CONTOUR_LEVELS = (3, 5, 7, 9)
CONTOUR_GREY = "black"
HEIGHT_SCALE = 1.25
SIGMA_BINS = 0.9
SUPPORT = 1.5
MIN_PATCH = 6

AXIS_LABELS = {"energy": "mean energy\nerror, meV/atom",
               "force": "mean force\nerror, meV/\u00c5"}

MAIN_LABELS = {"random_10k": "random", "madcore_10k": "mad-core", "voronoi_10k": "voronoi"}
AUG_LABELS = {"random_addition": "random", "madcore_top": "mad-core top",
              "voronoi_addition": "voronoi"}
UNITS = {"energy": "meV/atom", "force": "meV/A"}


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


def raw_panels(panels, models, min_count):
    count = panels["count"]
    keep = drop_specks(count >= min_count)
    values = {f"{m}__{k}_mean": np.where(keep, panels[f"{m}__{k}_mean"], np.nan)
              for m in models for k in ("energy", "force")}

    return values, keep, np.where(count > 0, count.astype(float), np.nan)


def smoothed_panels(panels, models, sigma=SIGMA_BINS, support=SUPPORT):
    smooth_count = gaussian_filter(panels["count"].astype(float), sigma, mode="constant")
    keep = drop_specks(smooth_count >= support)
    density = smooth_count

    values = {}
    for model in models:
        for kind in ("energy", "force"):
            total = gaussian_filter(panels[f"{model}__{kind}_sum"].astype(float), sigma, mode="constant")
            values[f"{model}__{kind}_mean"] = np.where(keep, total / np.maximum(density, 1e-12), np.nan)

    return values, keep, np.where(keep, density, np.nan)


def bare(axis, extent):
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xticks([])
    axis.set_yticks([])


def small_bar(figure, image, box, label, fig_size):
    axis = figure.add_axes([box[0] / fig_size[0], box[1] / fig_size[1],
                            box[2] / fig_size[0], box[3] / fig_size[1]])
    bar = figure.colorbar(image, cax=axis)
    bar.set_label(label, fontsize=10)
    bar.ax.tick_params(labelsize=9, length=2.5, pad=2)

    return bar


def map_figure(values, keep, coverage, models, labels, edges, density, window, path, say):
    rows, columns, extent = window
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])

    panel_w = 2.60
    panel_h = panel_w * aspect * HEIGHT_SCALE
    gap, bar_w, bar_gap, title_h = 0.09, 0.16, 0.07, 0.46
    left, right, bottom = 0.08, 1.11, 0.10

    block_w = 3 * panel_w + 2 * gap + bar_gap + bar_w
    label_room = 1.02
    context_w = (block_w - 2 * (bar_gap + bar_w) - label_room) / 2
    context_h = context_w * aspect * HEIGHT_SCALE

    y_force = bottom
    y_energy = y_force + panel_h + gap
    y_context = y_energy + panel_h + title_h + 0.10

    fig_w = left + block_w + right
    fig_h = y_context + context_h + title_h
    figure = plt.figure(figsize=(fig_w, fig_h))
    x_of = [left + index * (panel_w + gap) for index in range(3)]

    x_context = [left, left + context_w + bar_gap + bar_w + label_room]

    def place(x, y, w, h):
        return figure.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

    def side_bar(image, x, y, h, label, size):
        bar = figure.colorbar(image, cax=place(x, y, bar_w, h))
        bar.set_label(label, fontsize=size, labelpad=2)
        bar.ax.tick_params(labelsize=size - 2, length=3, pad=2)

    body = drop_specks(np.isfinite(density) & (density > 0))
    cropped = np.where(body, density, np.nan)[rows, columns]
    contours = (0.5 * (edges[0][rows.start:rows.stop] + edges[0][rows.start + 1:rows.stop + 1]),
                0.5 * (edges[1][columns.start:columns.stop] + edges[1][columns.start + 1:columns.stop + 1]),
                np.nan_to_num(cropped).T, CONTOUR_LEVELS)

    grey = plt.get_cmap(DENSITY_CMAP).copy()
    grey.set_bad("white")
    evaluated = plt.get_cmap(SHARE_CMAP).copy()
    evaluated.set_bad("white")
    context = [
        (np.ma.masked_invalid(cropped).T, grey, None, "library density", "log10 structures / bin"),
        (np.ma.masked_invalid(coverage).T[columns, rows], evaluated,
         LogNorm(vmin=max(np.nanmin(coverage), 0.5), vmax=np.nanmax(coverage)),
         "evaluated structures", "structures / bin"),
    ]
    axes = []
    for index, (panel, cmap, norm, title, label) in enumerate(context):
        axis = place(x_context[index], y_context, context_w, context_h)
        image = axis.imshow(panel, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
        axis.set_title(title, fontsize=21)
        side_bar(image, x_context[index] + context_w + bar_gap, y_context, context_h, label, 17)
        axes.append(axis)

    cmap = plt.get_cmap(ERROR_CMAP).copy()
    cmap.set_bad("white")
    for kind, y_row in (("energy", y_energy), ("force", y_force)):
        pool = np.concatenate([values[f"{m}__{kind}_mean"][keep] for m in models])
        pool = pool[np.isfinite(pool) & (pool > 0)]
        norm = LogNorm(vmin=np.percentile(pool, 1), vmax=np.percentile(pool, 99))

        for index, model in enumerate(models):
            axis = place(x_of[index], y_row, panel_w, panel_h)
            panel = np.ma.masked_invalid(values[f"{model}__{kind}_mean"]).T[columns, rows]
            image = axis.imshow(panel, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
            if kind == "energy":
                axis.set_title(labels[model], fontsize=21)
            axes.append(axis)

        side_bar(image, x_of[2] + panel_w + bar_gap, y_row, panel_h, AXIS_LABELS[kind], 17)
        say(f"  {kind}: log scale {norm.vmin:.0f}-{norm.vmax:.0f} {UNITS[kind]}")

    for axis in axes:
        axis.contour(*contours, colors=CONTOUR_GREY, linewidths=1.7, alpha=0.65)
        bare(axis, extent)

    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"{path}.{suffix}", dpi=400)
    plt.close(figure)
    say(f"wrote {path}.pdf/.png  ({fig_w:.1f} x {fig_h:.1f} in, {int(keep.sum())} supported bins)")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    say = make_logger(LOGS / "plot_maps.log")

    summary = json.load(open(REPORTS / "grid.json"))
    min_count = summary["min_count"]
    x_edges, y_edges = np.array(summary["grid"]["x_edges"]), np.array(summary["grid"]["y_edges"])
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])

    library = np.load(REPORTS / "cache" / "panels_library.npz")
    density = np.log10(np.where(library["population"] > 0, library["population"], np.nan))
    background = (x_centres, y_centres, np.nan_to_num(density).T, CONTOUR_LEVELS)

    main_panels = np.load(REPORTS / "cache" / "panels_main.npz")
    count = main_panels["count"]
    models = list(MAIN_MODELS)

    aug_panels = np.load(REPORTS / "cache" / "panels_aug.npz")
    aug_count = aug_panels["count"]
    aug_models = summary["panels"]["aug"]["models"]
    aug_min = max(5, min_count // 2)

    random_panels = np.load(REPORTS / "cache" / "panels_random_only.npz")
    madcore_panels = np.load(REPORTS / "cache" / "panels_madcore_only.npz")
    edges = (x_edges, y_edges)

    random_panels = np.load(REPORTS / "cache" / "panels_random_only.npz")
    madcore_panels = np.load(REPORTS / "cache" / "panels_madcore_only.npz")

    jobs = [
        ("sketchmap_error_smooth", main_panels, models, MAIN_LABELS, min_count, True),
        ("sketchmap_error_random_smooth", random_panels, models, MAIN_LABELS, min_count, True),
        ("sketchmap_error_madcore_smooth", madcore_panels, models, MAIN_LABELS, min_count, True),
    ]

    windows = {}
    for path, panels, arms, labels, cut, smooth in jobs:
        key = id(panels)
        if key not in windows:
            footprint = drop_specks(gaussian_filter(panels["count"].astype(float), SIGMA_BINS,
                                                     mode="constant") >= SUPPORT)
            windows[key] = occupied_window(footprint, x_edges, y_edges, margin=2)

        values, keep, coverage = (smoothed_panels(panels, arms) if smooth
                                  else raw_panels(panels, arms, cut))
        say(f"\n{path}: {'smoothed sigma %.1f bins, support %.1f' % (SIGMA_BINS, SUPPORT) if smooth else 'raw, min_count %d' % cut}")
        map_figure(values, keep, coverage, arms, labels, edges, density, windows[key], path, say)


if __name__ == "__main__":
    main()
