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
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator  # noqa: E402

sys.path.insert(0, "${DEV_ROOT}/phace-atomic/mad-core-valid/experiments/scratch_scaling/scripts")
from maps_common import B, LIBRARY_XY, MAPS, P, POPULATION_FILE  # noqa: E402

S = Path(__file__).resolve().parent.parent
OUT = S / "reports" / "populations"
STATS = S / "reports" / "selected_checkpoints" / "cache_stats"
PROVENANCE = S / "reports" / "eval_provenance.tsv"
ROUND3_PANELS = S / "reports" / "maps" / "panels_scaling_selected.npz"
SCALE_FROM = MAPS / "panels_scaling_selected.npz"
PREFIX = "scratchv3-s"
POPULATIONS = ("union_7847", "balanced_4000")
TESTS = ("test_random", "test_madcore")
SIZES = ("2k", "5k", "10k", "20k")
PLOT_SIZES = ("5k", "10k", "20k")
X = [int(s[:-1]) * 1000 for s in PLOT_SIZES]
ARMS = {"madcore": ("MAD-CORE", "#D55E00", "s"), "random": ("RANDOM", "#0072B2", "o")}
MAP_ARMS = ("random", "madcore")
KINDS = {"energy": ("E", "meV/atom", 1.0), "force": ("F", "eV/Å", 1e-3)}
METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("p8", "p8"), ("sup", "SUP"))
BORDER_WIDTH = plt.rcParams["axes.linewidth"]


def model_of(arm, size):
    return f"{PREFIX}--{arm}_{size}--s42"


MODELS = [model_of(arm, size) for arm in ARMS for size in SIZES]
PLOT_MODELS = [model_of(arm, size) for arm in ARMS for size in PLOT_SIZES]


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path, rows, columns):
    with open(path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in columns) + "\n")


# per structure errors of every model, in the order of the population manifest
def population_errors(manifest):
    caches = {(m, t): np.load(STATS / f"{m}@{t}.npz", allow_pickle=True) for m in MODELS for t in TESTS}
    position = {t: {c: i for i, c in enumerate(caches[(MODELS[0], t)]["configuration_id"].astype(str))} for t in TESTS}
    for m in MODELS:
        for t in TESTS:
            assert (caches[(m, t)]["configuration_id"].astype(str) == caches[(MODELS[0], t)]["configuration_id"].astype(str)).all()

    index = {t: np.array([position[t][r["configuration_id"]] for r in manifest if r["source_test"] == t]) for t in TESTS}
    errors = {}
    for m in MODELS:
        parts = [caches[(m, t)] for t in TESTS]
        errors[m] = {
            key: np.concatenate([part[key][index[t]] for part, t in zip(parts, TESTS)])
            for key in ("natoms", "de", "n_components", "f_sum_abs", "f_sum_sq", "f_sum_e8", "f_max_abs")
        }
    return errors


# pooled metrics in meV, energy over structures, forces over all cartesian components of the population
def pooled(e):
    a = np.abs(e["de"])
    n = e["n_components"].sum()
    return {
        "E_mae": a.mean(),
        "E_rmse": np.sqrt((a ** 2).mean()),
        "E_p8": ((a ** 8).mean()) ** (1 / 8),
        "E_sup": a.max(),
        "F_mae": e["f_sum_abs"].sum() / n,
        "F_rmse": np.sqrt(e["f_sum_sq"].sum() / n),
        "F_p8": (e["f_sum_e8"].sum() / n) ** (1 / 8),
        "F_sup": e["f_max_abs"].max(),
    }


def provenance():
    rows = [r for r in read_tsv(PROVENANCE) if r["checkpoint"] == "selected"]
    found = {}
    for r in rows:
        entry = (r["best_epoch"], r["exported_sha256"])
        assert found.setdefault(r["run"], entry) == entry, r["run"]
    return found


def log_padding(low, high, fraction=0.12):
    factor = (high / low) ** fraction
    return low / factor, high * factor


# log y axis with labelled powers of ten, a panel without a power of ten gets plain labels at 2, 3 and 5 times a power of ten
def log_y_ticks(ax, limits):
    ax.margins(x=0.09)
    ax.autoscale_view()
    ax.set_ylim(*limits)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    low, high = limits
    decades_inside = math.floor(math.log10(high)) - math.ceil(math.log10(low)) + 1
    if decades_inside >= 1:
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** math.floor(math.log10(v)), 6) in (2.0, 3.0, 5.0) else ""))


def learning_curve_figure(population, metrics, limits):
    fig, axes = plt.subplots(2, len(METRICS), figsize=(22, 10.5), sharex=True)
    for r, (kind, (column, unit, scale)) in enumerate(KINDS.items()):
        for c, (metric, metric_label) in enumerate(METRICS):
            ax = axes[r, c]
            for arm, (name, colour, marker) in ARMS.items():
                y = [metrics[population][model_of(arm, size)][f"{column}_{metric}"] * scale for size in PLOT_SIZES]
                ax.plot(X, y, "-", color=colour, lw=2.8, marker=marker, ms=11)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(X)
            ax.set_xticklabels(PLOT_SIZES, fontsize=22)
            log_y_ticks(ax, limits[(kind, metric)])
            ax.tick_params(axis="y", which="both", labelsize=21)
            ax.tick_params(which="major", length=9, width=1.6)
            ax.tick_params(which="minor", length=5, width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.6)

            if r == 0:
                ax.set_title(metric_label, fontsize=26)
            if c == 0:
                ax.set_ylabel(f"{kind} error, {unit}", fontsize=24)
            if r == 1:
                ax.set_xlabel("training structures", fontsize=22)

    handles = [plt.Line2D([], [], color=colour, marker=marker, lw=3, ms=12, label=name) for name, colour, marker in ARMS.values()]
    fig.legend(
        handles=handles,
        title="training set",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=22,
        title_fontsize=22,
        frameon=False,
        handlelength=4.0,
        columnspacing=2.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.885))

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / population / "figures" / f"learning_curves_{population}.{suffix}", dpi=220)
    plt.close(fig)


def records_of(manifest, errors, coordinate_of):
    cids = [r["configuration_id"] for r in manifest]
    return {
        "configuration_id": np.array(cids),
        "x": np.array([coordinate_of[c][0] for c in cids]),
        "y": np.array([coordinate_of[c][1] for c in cids]),
        "set": np.array([r["source_test"] for r in manifest]),
        "errors": {m: {"energy": np.abs(errors[m]["de"]), "force": errors[m]["f_sum_abs"] / errors[m]["n_components"]} for m in MODELS},
    }


def map_figure(population, kind, values, coverage, window, background, norms):
    rows, columns, extent = window
    cropped, contours = background
    column, unit, scale = KINDS[kind]
    norm = norms[kind]
    cmap = plt.get_cmap(P.ERROR_CMAP).copy()
    cmap.set_bad("white")

    n_col = len(PLOT_SIZES) + 1
    fig, axes = plt.subplots(len(MAP_ARMS), n_col, figsize=(3.375 * n_col, 3.1 * len(MAP_ARMS)))
    image = None
    shown = []
    for r, arm in enumerate(MAP_ARMS):
        for c, size in enumerate(PLOT_SIZES):
            ax = axes[r, c]
            panel = values[f"{model_of(arm, size)}__{kind}_mean"] * scale
            shown.append(panel[np.isfinite(panel)])
            image = ax.imshow(np.ma.masked_invalid(panel).T[columns, rows], origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
            ax.contour(*contours, cropped.T, levels=P.CONTOUR_LEVELS, colors=P.CONTOUR_GREY, linewidths=BORDER_WIDTH, alpha=0.6)
            P.bare(ax, extent)
            if r == 0:
                ax.set_title(size, fontsize=16)
            if c == 0:
                ax.set_ylabel(ARMS[arm][0], fontsize=16)

    # context column with the library density and the evaluated structures per bin
    last = len(PLOT_SIZES)
    grey = plt.get_cmap(P.DENSITY_CMAP).copy()
    grey.set_bad("white")
    axes[0, last].imshow(np.ma.masked_invalid(cropped).T, origin="lower", extent=extent, aspect="auto", cmap=grey)
    axes[0, last].set_title("library density", fontsize=14)

    blues = plt.get_cmap(P.SHARE_CMAP).copy()
    blues.set_bad("white")
    cov = axes[1, last].imshow(np.ma.masked_invalid(coverage).T[columns, rows], origin="lower", extent=extent, aspect="auto", cmap=blues, norm=norms["count"])
    axes[1, last].text(0.97, 0.03, "evaluated structures", transform=axes[1, last].transAxes, ha="right", va="bottom", fontsize=14)

    for ax in axes[:2, last]:
        ax.contour(*contours, cropped.T, levels=P.CONTOUR_LEVELS, colors=P.CONTOUR_GREY, linewidths=BORDER_WIDTH, alpha=0.6)
        P.bare(ax, extent)

    fig.subplots_adjust(left=0.04, right=0.90, top=0.88, bottom=0.03, wspace=0.02, hspace=0.03)
    fig.suptitle(f"{kind} error, {unit}, PET-S", fontsize=18)

    row_bars = (
        (axes[0, last], image, f"mean error, {unit}"),
        (axes[1, last], cov, "structures / bin"),
    )
    for row_axis, mappable, label in row_bars:
        box = row_axis.get_position()
        bar_axis = fig.add_axes([box.x1 + 0.01, box.y0, 0.012, box.height])
        bar = fig.colorbar(mappable, cax=bar_axis)
        bar.set_label(label, fontsize=14)
        bar.ax.tick_params(labelsize=13)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / population / "figures" / f"sketchmap_{kind}_{population}.{suffix}", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # bins outside the fixed colour limits are drawn in the end colours
    shown = np.concatenate(shown)
    return {"bins_below": int((shown < norm.vmin).sum()), "bins_above": int((shown > norm.vmax).sum()), "bins_shown": int(shown.size)}


def compact_markdown(population, metrics):
    lines = [
        "| training set | size | E MAE / RMSE, meV/atom | F MAE / RMSE, eV/Å | E p8, meV/atom | F p8, eV/Å | E SUP, meV/atom | F SUP, eV/Å |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, (name, _, _) in ARMS.items():
        for size in SIZES:
            v = metrics[population][model_of(arm, size)]
            lines.append(
                f"| {name} | {size} | {v['E_mae']:.1f} / {v['E_rmse']:.1f} | {v['F_mae'] / 1000:.3f} / {v['F_rmse'] / 1000:.3f} | "
                f"{v['E_p8']:.1f} | {v['F_p8'] / 1000:.2f} | {v['E_sup']:.1f} | {v['F_sup'] / 1000:.2f} |"
            )
    return "\n".join(lines)


def main():
    summary = json.loads((OUT / "POPULATIONS.json").read_text())
    manifests = {p: read_tsv(OUT / "manifests" / f"{p}.tsv") for p in POPULATIONS}
    for p in POPULATIONS:
        (OUT / p / "figures").mkdir(parents=True, exist_ok=True)
        (OUT / p / "tables").mkdir(parents=True, exist_ok=True)
    checkpoints = provenance()

    # pooled metrics for every population and model
    errors = {p: population_errors(manifests[p]) for p in POPULATIONS}
    metrics = {p: {m: pooled(errors[p][m]) for m in MODELS} for p in POPULATIONS}
    for p in POPULATIONS:
        total = summary["populations"][p]["total"]
        for m in MODELS:
            assert len(errors[p][m]["de"]) == total["structures"]
            assert int(errors[p][m]["n_components"].sum()) == total["force_components"]

    # full and compact tables, and the plotted learning curve values
    full_columns = ["population", "training_set", "size", "model", "best_epoch", "model_best_sha256", "structures", "atoms", "force_components",
                    "E_mae_meV_per_atom", "E_rmse_meV_per_atom", "E_p8_meV_per_atom", "E_sup_meV_per_atom",
                    "F_mae_eV_per_A", "F_rmse_eV_per_A", "F_p8_eV_per_A", "F_sup_eV_per_A"]
    compact_columns = ["training_set", "size", "E_MAE", "E_RMSE", "F_MAE", "F_RMSE", "E_p8", "F_p8", "E_SUP", "F_SUP"]
    source_columns = ["population", "training_set", "size", "training_structures", "model", "quantity", "metric", "value", "unit"]
    for p in POPULATIONS:
        total = summary["populations"][p]["total"]
        full, compact, source = [], [], []
        for arm, (name, _, _) in ARMS.items():
            for size in SIZES:
                m = model_of(arm, size)
                v = metrics[p][m]
                epoch, digest = checkpoints[m]
                full.append({
                    "population": p, "training_set": name, "size": size, "model": m, "best_epoch": epoch, "model_best_sha256": digest,
                    "structures": total["structures"], "atoms": total["atoms"], "force_components": total["force_components"],
                    **{f"E_{k}_meV_per_atom": f"{v[f'E_{k}']:.6f}" for k in ("mae", "rmse", "p8", "sup")},
                    **{f"F_{k}_eV_per_A": f"{v[f'F_{k}'] / 1000:.6f}" for k in ("mae", "rmse", "p8", "sup")},
                })
                compact.append({
                    "training_set": name, "size": size,
                    "E_MAE": f"{v['E_mae']:.1f}", "E_RMSE": f"{v['E_rmse']:.1f}", "F_MAE": f"{v['F_mae'] / 1000:.3f}", "F_RMSE": f"{v['F_rmse'] / 1000:.3f}",
                    "E_p8": f"{v['E_p8']:.1f}", "F_p8": f"{v['F_p8'] / 1000:.2f}", "E_SUP": f"{v['E_sup']:.1f}", "F_SUP": f"{v['F_sup'] / 1000:.2f}",
                })
                for kind, (column, unit, scale) in KINDS.items():
                    for metric, metric_label in METRICS:
                        if size not in PLOT_SIZES:
                            continue
                        source.append({
                            "population": p, "training_set": name, "size": size, "training_structures": int(size[:-1]) * 1000, "model": m,
                            "quantity": kind, "metric": metric_label, "value": f"{v[f'{column}_{metric}'] * scale:.6f}", "unit": unit.replace("Å", "A"),
                        })
        write_tsv(OUT / p / "tables" / f"metrics_{p}.tsv", full, full_columns)
        write_tsv(OUT / p / "tables" / f"metrics_compact_{p}.tsv", compact, compact_columns)

        # each part of the population on its own, so the sampled random part can be compared with the full random test
        by_part = []
        source_test = np.array([r["source_test"] for r in manifests[p]])
        for t in TESTS:
            mask = source_test == t
            for arm, (name, _, _) in ARMS.items():
                for size in SIZES:
                    e = {k: v[mask] for k, v in errors[p][model_of(arm, size)].items()}
                    v = pooled(e)
                    by_part.append({
                        "population": p, "part": t, "training_set": name, "size": size, "structures": int(mask.sum()), "force_components": int(e["n_components"].sum()),
                        **{f"E_{k}_meV_per_atom": f"{v[f'E_{k}']:.6f}" for k in ("mae", "rmse", "p8", "sup")},
                        **{f"F_{k}_eV_per_A": f"{v[f'F_{k}'] / 1000:.6f}" for k in ("mae", "rmse", "p8", "sup")},
                    })
        write_tsv(OUT / p / "tables" / f"metrics_by_part_{p}.tsv", by_part, list(by_part[0].keys()))
        write_tsv(OUT / p / "tables" / f"learning_curves_source_{p}.tsv", source, source_columns)
        (OUT / p / "tables" / f"metrics_compact_{p}.md").write_text(compact_markdown(p, metrics) + "\n")

    # learning curves with the same y range per panel in both versions
    limits = {}
    for kind, (column, unit, scale) in KINDS.items():
        for metric, _ in METRICS:
            values = [metrics[p][m][f"{column}_{metric}"] * scale for p in POPULATIONS for m in PLOT_MODELS]
            limits[(kind, metric)] = log_padding(min(values), max(values))
    for p in POPULATIONS:
        learning_curve_figure(p, metrics, limits)
        print("wrote learning curves", p)

    # maps on the round 1 projection, grid and colour limits, binned again for each population
    coords = np.load(MAPS / "coords_eval.npz", allow_pickle=True)
    coordinate_of = {cid: coords["embedding"][i, :2] for i, cid in enumerate(coords["configuration_id"].astype(str))}
    grid = json.load(open(MAPS / "grid.json"))
    x_edges, y_edges = np.asarray(grid["x_edges"]), np.asarray(grid["y_edges"])
    library_xy = np.load(LIBRARY_XY)
    library_population = np.load(POPULATION_FILE, mmap_mode="r")
    _, weighted, _ = B.library_background(library_xy, library_population, grid)
    density = np.log10(np.where(weighted > 0, weighted, np.nan))

    reference = np.load(SCALE_FROM)
    error_limits = {}
    for kind in KINDS:
        pool = np.concatenate([reference[k][reference["keep"]] for k in reference.files if k.endswith(f"__{kind}_mean")])
        error_limits[kind] = (float(np.percentile(pool, 1)), float(np.percentile(pool, 99)))

    panels = {}
    for p in POPULATIONS:
        records = records_of(manifests[p], errors[p], coordinate_of)
        count, composition, statistics = B.bin_statistics(records, MODELS, list(TESTS), grid)
        values, keep, coverage = P.smoothed_panels({"count": count, **statistics}, MODELS)
        panels[p] = {"count": count, "composition": composition, "values": values, "keep": keep, "coverage": coverage}
        np.savez(
            OUT / p / "tables" / f"sketchmap_panels_{p}.npz",
            count=count, keep=keep, coverage=coverage, density=density, x_edges=x_edges, y_edges=y_edges,
            **values, **{f"set__{k}": v for k, v in composition.items()},
        )

    # the full union must reproduce the delivered round 3 maps for the sizes they show
    delivered = np.load(ROUND3_PANELS)
    union = panels["union_7847"]
    assert (delivered["keep"] == union["keep"]).all() and (delivered["count"] == union["count"]).all()
    for key in delivered.files:
        if key.endswith("_mean"):
            assert np.allclose(delivered[key], union["values"][key], equal_nan=True, rtol=1e-12, atol=1e-9), key
    print("union panels reproduce the delivered round 3 maps")

    window = P.occupied_window(union["keep"], x_edges, y_edges)
    rows, columns, _ = window
    body = P.drop_specks(np.isfinite(density) & (density > 0))
    cropped = np.where(body, density, np.nan)[rows, columns]
    contours = (
        0.5 * (x_edges[rows.start:rows.stop] + x_edges[rows.start + 1:rows.stop + 1]),
        0.5 * (y_edges[columns.start:columns.stop] + y_edges[columns.start + 1:columns.stop + 1]),
    )
    coverage_all = np.concatenate([panels[p]["coverage"][np.isfinite(panels[p]["coverage"])] for p in POPULATIONS])
    norms = {
        "energy": LogNorm(vmin=error_limits["energy"][0], vmax=error_limits["energy"][1]),
        "force": LogNorm(vmin=error_limits["force"][0] * KINDS["force"][2], vmax=error_limits["force"][1] * KINDS["force"][2]),
        "count": LogNorm(vmin=max(coverage_all.min(), 0.5), vmax=coverage_all.max()),
    }

    map_info = {"colour_limits_meV": error_limits, "colour_limits_from": str(SCALE_FROM), "count_limits": [norms["count"].vmin, norms["count"].vmax], "populations": {}}
    for p in POPULATIONS:
        panel = panels[p]
        info = {
            "supported_bins": int(panel["keep"].sum()),
            "structures_inside_grid": int(panel["count"].sum()),
            "structures_in_supported_bins": int(panel["count"][panel["keep"]].sum()),
        }
        for kind in KINDS:
            info[kind] = map_figure(p, kind, panel["values"], panel["coverage"], window, (cropped, contours), norms)
        map_info["populations"][p] = info
        print("wrote maps", p, info)

        # plotted bin values of the supported bins
        i_index, j_index = np.nonzero(panel["keep"])
        bins = []
        for i, j in zip(i_index, j_index):
            row = {
                "i": i, "j": j,
                "x_lo": f"{x_edges[i]:.6f}", "x_hi": f"{x_edges[i + 1]:.6f}", "y_lo": f"{y_edges[j]:.6f}", "y_hi": f"{y_edges[j + 1]:.6f}",
                "structures": int(panel["count"][i, j]), "smoothed_structures": f"{panel['coverage'][i, j]:.4f}",
                **{f"{t}_structures": int(panel["composition"][t][i, j]) for t in TESTS},
            }
            for m in PLOT_MODELS:
                arm_size = m.split("--")[1]
                row[f"{arm_size}_energy_meV_per_atom"] = f"{panel['values'][f'{m}__energy_mean'][i, j]:.6f}"
                row[f"{arm_size}_force_eV_per_A"] = f"{panel['values'][f'{m}__force_mean'][i, j] / 1000:.6f}"
            bins.append(row)
        columns_out = list(bins[0].keys())
        write_tsv(OUT / p / "tables" / f"sketchmap_bins_{p}.tsv", bins, columns_out)

    (OUT / "MAPS.json").write_text(json.dumps(map_info, indent=1) + "\n")

    # the two versions side by side, the arm ratio and which arm is lower
    comparison = []
    for kind, (column, unit, scale) in KINDS.items():
        for metric, metric_label in METRICS:
            for size in SIZES:
                row = {"quantity": kind, "metric": metric_label, "size": size, "unit": unit.replace("Å", "A")}
                for p in POPULATIONS:
                    mad = metrics[p][model_of("madcore", size)][f"{column}_{metric}"] * scale
                    ran = metrics[p][model_of("random", size)][f"{column}_{metric}"] * scale
                    row[f"madcore_{p}"] = f"{mad:.4g}"
                    row[f"random_{p}"] = f"{ran:.4g}"
                    row[f"random_over_madcore_{p}"] = f"{ran / mad:.3g}"
                    row[f"lower_{p}"] = "MAD-CORE" if mad < ran else "RANDOM"
                comparison.append(row)
    write_tsv(OUT / "comparison.tsv", comparison, list(comparison[0].keys()))

    parts = []
    for p in POPULATIONS:
        parts.append(f"## {p}\n\n{compact_markdown(p, metrics)}\n")
    (OUT / "METRICS.md").write_text("# Pooled test metrics, round 3, best checkpoints\n\n" + "\n".join(parts))
    print("POPULATION FIGURES DONE")


if __name__ == "__main__":
    main()
