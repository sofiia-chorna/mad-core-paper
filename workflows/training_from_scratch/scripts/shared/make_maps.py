import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from maps_common import ARM_LABEL, B, FIGURE_SETS, LIBRARY_XY, MANIFESTS, MAPS, P, POPULATION_FILE, S, SET_LABEL, SETS, SIZES, make_logger, read_tsv  # noqa: E402

# best validation checkpoints only
parser = argparse.ArgumentParser()
parser.add_argument("figure_set", nargs="?", default="scaling")
parser.add_argument("--round", default=str(S))
parser.add_argument("--prefix", default="scratch-s")
parser.add_argument("--sizes", default=",".join(SIZES))
parser.add_argument("--scale-from", default=None, help="panels npz of another round whose pooled colour scale is reused, so rounds are comparable")
ARGS = parser.parse_args()
FIGURE_SET = ARGS.figure_set
ROUND = Path(ARGS.round).resolve()
SIZES = tuple(ARGS.sizes.split(","))
ARMS = FIGURE_SETS[FIGURE_SET]


def model_of(arm, size):
    return f"{ARGS.prefix}--{arm}_{size}--s42"


MODELS = [model_of(arm, size) for arm in ARMS for size in SIZES]
SUFFIX = "scaling" if FIGURE_SET == "scaling" else f"scaling_{FIGURE_SET}"
REPORTS = ROUND / "reports"
FIGURES = REPORTS / "figures"
LOGS = ROUND / "logs"
ROUND_MAPS = REPORTS / "maps"
TRAIN_SETS = json.load(open(S / "manifests" / "SETS.json"))["sets"]
CACHE = REPORTS / "selected_checkpoints" / "cache_mtt"
AVAILABLE = [m for m in MODELS if all((CACHE / f"{m}@{name}.npz").exists() for name in SETS)]
UNITS = {"energy": "meV/atom", "force": "meV/A"}
BORDER_WIDTH = plt.rcParams["axes.linewidth"]
TITLE = {"energy": "energy error, meV/atom", "force": "force error, meV/A"}


def load_errors(say):
    coords = np.load(MAPS / "coords_eval.npz", allow_pickle=True)
    coordinate_of = {cid: coords["embedding"][i, :2] for i, cid in enumerate(coords["configuration_id"].astype(str))}
    identifier, x, y, set_name, errors = [], [], [], [], {m: {"energy": [], "force": []} for m in AVAILABLE}
    counts = {}
    if not AVAILABLE:
        say("no evaluated model yet")
        return None, None
    for m in MODELS:
        if m not in AVAILABLE:
            say(f"{m}: evaluation not available yet, panel left blank")
    for name in SETS:
        cids = None
        for m in AVAILABLE:
            path = CACHE / (f"{m}@{name}.npz")
            d = np.load(path, allow_pickle=True)
            if cids is None:
                cids = d["configuration_id"].astype(str)
            assert (d["configuration_id"].astype(str) == cids).all(), "model caches disagree on the structure order"
            errors[m]["energy"].append(np.abs(d["de"]))
            errors[m]["force"].append(d["df_comp"])
        identifier.extend(cids.tolist())
        set_name.extend([name] * len(cids))
        x.extend(coordinate_of[c][0] for c in cids)
        y.extend(coordinate_of[c][1] for c in cids)
        counts[name] = len(cids)
    records = {"configuration_id": np.array(identifier), "x": np.array(x), "y": np.array(y), "set": np.array(set_name), "group": np.array([c.split(":")[0] for c in identifier]),
               "errors": {m: {k: np.concatenate(v) for k, v in errors[m].items()} for m in AVAILABLE}}
    assert len(set(identifier)) == len(identifier), "the two tests overlap"
    return records, counts


def map_figure(kind, values, keep, coverage, density, grid, say):
    x_edges, y_edges = np.asarray(grid["x_edges"]), np.asarray(grid["y_edges"])
    rows, columns, extent = P.occupied_window(keep, x_edges, y_edges)
    pool = np.concatenate([values[f"{m}__{kind}_mean"][keep] for m in AVAILABLE])
    if ARGS.scale_from:
        ref = np.load(ARGS.scale_from)
        ref_keep = ref["keep"]
        pool = np.concatenate([ref[k][ref_keep] for k in ref.files if k.endswith(f"__{kind}_mean")])
        say(f"colour scale taken from {ARGS.scale_from}")
    norm = LogNorm(vmin=np.percentile(pool, 1), vmax=np.percentile(pool, 99))
    cmap = plt.get_cmap(P.ERROR_CMAP).copy()
    cmap.set_bad("white")
    body = P.drop_specks(np.isfinite(density) & (density > 0))
    cropped = np.where(body, density, np.nan)[rows, columns]
    contours = (0.5 * (x_edges[rows.start:rows.stop] + x_edges[rows.start + 1:rows.stop + 1]), 0.5 * (y_edges[columns.start:columns.stop] + y_edges[columns.start + 1:columns.stop + 1]))
    n_col = len(SIZES) + 1
    fig, axes = plt.subplots(len(ARMS), n_col, figsize=(3.375 * n_col, 3.1 * len(ARMS)))
    image = None
    for r, arm in enumerate(ARMS):
        for c, size in enumerate(SIZES):
            ax = axes[r, c]
            model = model_of(arm, size)
            if model in AVAILABLE:
                panel = np.ma.masked_invalid(values[f"{model}__{kind}_mean"]).T[columns, rows]
                image = ax.imshow(panel, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
            else:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="grey")
            ax.contour(*contours, cropped.T, levels=P.CONTOUR_LEVELS, colors=P.CONTOUR_GREY, linewidths=BORDER_WIDTH, alpha=0.6)
            P.bare(ax, extent)
            if r == 0:
                ax.set_title(size if FIGURE_SET != "atoms" else f"atom budget of MAD-CORE {size}\n({TRAIN_SETS[f'madcore_{size}']['atoms']:,} atoms)", fontsize=16)
            if c == 0:
                ax.set_ylabel(ARM_LABEL[arm].replace(" training", "").upper(), fontsize=16)
    # context column library density and evaluation coverage
    grey = plt.get_cmap(P.DENSITY_CMAP).copy()
    grey.set_bad("white")
    last = len(SIZES)
    axes[0, last].imshow(np.ma.masked_invalid(cropped).T, origin="lower", extent=extent, aspect="auto", cmap=grey)
    axes[0, last].set_title("library density", fontsize=14)
    blues = plt.get_cmap(P.SHARE_CMAP).copy()
    blues.set_bad("white")
    cov = axes[1, last].imshow(np.ma.masked_invalid(coverage).T[columns, rows], origin="lower", extent=extent, aspect="auto", cmap=blues, norm=LogNorm(vmin=max(np.nanmin(coverage), 0.5), vmax=np.nanmax(coverage)))
    axes[1, last].text(0.97, 0.03, "evaluated structures", transform=axes[1, last].transAxes, ha="right", va="bottom", fontsize=14)
    for ax in axes[2:, last]:
        ax.set_axis_off()
    # density contours in the context column too, with the same line width as the panel borders
    for ax in axes[:2, last]:
        ax.contour(*contours, cropped.T, levels=P.CONTOUR_LEVELS, colors=P.CONTOUR_GREY, linewidths=BORDER_WIDTH, alpha=0.6)
        P.bare(ax, extent)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.03, wspace=0.02, hspace=0.03)
    fig.suptitle(f"{TITLE[kind]}, PET-S", fontsize=18)

    # one colour bar per row with the height of that row
    row_bars = (
        (axes[0, last], image, f"mean error, {UNITS[kind]}"),
        (axes[1, last], cov, "structures / bin"),
    )
    for row_axis, mappable, label in row_bars:
        box = row_axis.get_position()
        bar_axis = fig.add_axes([box.x1 + 0.01, box.y0, 0.012, box.height])
        bar = fig.colorbar(mappable, cax=bar_axis)
        bar.set_label(label, fontsize=14)
        bar.ax.tick_params(labelsize=13)

    out = FIGURES / f"sketchmap_{kind}_{SUFFIX}_selected.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.05)
    say(f"wrote {out}")
    plt.close(fig)


MARKER = {"random": "o", "madcore": "s", "voronoi": "^", "random_atoms": "o", "voronoi_atoms": "^"}


def learning_curves(say, arms, axis, name):
    rows = {(r["test"], r["model"]): r for r in read_tsv(REPORTS / "selected_checkpoints" / "metrics_mtt.tsv")}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for c, test in enumerate(SETS):
        for r, kind in enumerate(("E", "F")):
            ax = axes[r, c]
            for arm in arms:
                x = [TRAIN_SETS[f"{arm}_{s}"][axis] if f"{arm}_{s}" in TRAIN_SETS and axis in TRAIN_SETS[f"{arm}_{s}"] else int(s[:-1]) * 1000 for s in SIZES]
                for metric, ls in (("mae", "-"), ("rmse", "--")):
                    vals = [float(rows[(test, model_of(arm, s))][f"{kind}_{metric}"]) if (test, model_of(arm, s)) in rows else np.nan for s in SIZES]
                    ax.plot(x, vals, MARKER[arm] + ls, label=f"{ARM_LABEL[arm].replace(chr(10), ' ')}, {metric.upper()}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if axis == "structures":
                ax.set_xticks([int(s[:-1]) * 1000 for s in SIZES])
                ax.set_xticklabels(list(SIZES))
            ax.set_xlabel(f"training {axis}")
            ax.set_ylabel("energy error (meV/atom)" if kind == "E" else "force error (meV/A)")
            if r == 0:
                ax.set_title(SET_LABEL[test])
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("PET-S, best checkpoints", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    say(f"wrote {name}")


def main():
    LOGS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ROUND_MAPS.mkdir(parents=True, exist_ok=True)
    say = make_logger(LOGS / f"make_maps_{SUFFIX}_selected.log")
    records, counts = load_errors(say)
    if records is None:
        return
    library_xy = np.load(LIBRARY_XY)
    population = np.load(POPULATION_FILE, mmap_mode="r")
    grid_path = MAPS / "grid.json"
    if grid_path.exists():
        grid = json.load(open(grid_path))
    else:
        grid = B.make_grid(library_xy, records)
        json.dump(grid, open(grid_path, "w"))
    _, _, inside = B.digitise(records["x"], records["y"], grid)
    say(f"{len(records['configuration_id']):,} structures ({counts}), inside grid {int(inside.sum()):,}")
    count, composition, statistics = B.bin_statistics(records, AVAILABLE, list(SETS), grid)
    panels = {"count": count, **statistics}
    values, keep, coverage = P.smoothed_panels(panels, AVAILABLE)
    centres, weighted, inside_library = B.library_background(library_xy, population, grid)
    density = np.log10(np.where(weighted > 0, weighted, np.nan))
    np.savez(ROUND_MAPS / f"panels_{SUFFIX}_selected.npz", count=count, keep=keep, coverage=coverage, density=density, x_edges=np.asarray(grid["x_edges"]), y_edges=np.asarray(grid["y_edges"]),
             **{k: v for k, v in values.items()}, **{f"set__{k}": v for k, v in composition.items()})
    np.savez(ROUND_MAPS / f"records_{SUFFIX}_selected.npz", configuration_id=records["configuration_id"], x=records["x"], y=records["y"], set=records["set"],
             **{f"{m}__{k}": records["errors"][m][k] for m in AVAILABLE for k in ("energy", "force")})
    say(f"support bins {int(keep.sum())}; structures in supported bins {int(count[keep].sum()):,} of {int(count.sum()):,}")
    for kind in ("energy", "force"):
        map_figure(kind, values, keep, coverage, density, grid, say)
    if FIGURE_SET == "scaling":
        learning_curves(say, ("random", "madcore"), "structures", "learning_curves")
    if FIGURE_SET == "voronoi":
        learning_curves(say, ("random", "voronoi", "madcore"), "structures", "learning_curves_voronoi")
    if FIGURE_SET == "atoms":
        learning_curves(say, ("random", "voronoi", "madcore", "random_atoms", "voronoi_atoms"), "atoms", "learning_curves_atoms")


if __name__ == "__main__":
    main()
