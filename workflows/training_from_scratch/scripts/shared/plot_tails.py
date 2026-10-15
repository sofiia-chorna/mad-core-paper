import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator  # noqa: E402

# mae, rmse, p8 and sup against training set size, mad-core versus random training, best validation checkpoints; two
# figures per round (energy and forces with the tests in separate columns, and with both tests in one panel),
# metrics taken verbatim from the verified evaluation tables; captions go to CAPTIONS.md, not into the image; png only
# usage: plot_tails.py --round DIR --prefix scratch-s --sizes 5k,10k,20k --label "round 1" --validation "..."
parser = argparse.ArgumentParser()
parser.add_argument("--round", required=True)
parser.add_argument("--prefix", required=True)
parser.add_argument("--sizes", default="5k,10k,20k")
parser.add_argument("--label", required=True)
parser.add_argument("--validation", default="the common 1,000-structure random validation set")
parser.add_argument("--xlabel", default="training structures")
parser.add_argument("--extra", default="")
ARGS = parser.parse_args()
ROUND = Path(ARGS.round).resolve()
SIZES = ARGS.sizes.split(",")
TABLE = ROUND / "reports" / "selected_checkpoints" / "metrics_mtt.tsv"
FIGURES = ROUND / "reports" / "figures"
TESTS = {"test_random": "RANDOM test set, 5,847 structures", "test_madcore": "MAD-CORE test set, 2,000 structures"}
ARMS = {"madcore": ("MAD-CORE", "#D55E00", "s"), "random": ("RANDOM", "#0072B2", "o")}
PROTOCOL = ("{label}: PET-S trained from scratch, seed 42, best checkpoints selected by native mae_prod on {validation}. All structures of each test are included, no clipping. "
            "Metrics as in metatrain: energy error per structure = |dE| / atoms in meV/atom, statistics over structures; force errors pooled over all force components "
            "(x, y, z of every atom) of the test set, in eV/A; p8 = (mean e^8)^(1/8) and SUP = max use the same pooling.{extra}")
TEST_TITLE = {"test_random": "RANDOM test set", "test_madcore": "MAD-CORE test set"}
KINDS = {"energy": ("E", "meV/atom", 1.0), "force": ("F", "eV/A", 1e-3)}
METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("p8", "p8"), ("sup", "SUP"))


# padding between the points and the panel border, then a standard log y axis: labelled powers of ten and unlabelled minor ticks; a panel that contains no power of ten gets
# plain number labels at 2, 3 and 5 times a power of ten instead, so every panel shows values; called after plotting
def log_y_ticks(ax):
    ax.margins(x=0.09, y=0.12)
    ax.autoscale_view()
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    low, high = ax.get_ylim()
    decades_inside = math.floor(math.log10(high)) - math.ceil(math.log10(low)) + 1
    if decades_inside >= 1:
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** math.floor(math.log10(v)), 6) in (2.0, 3.0, 5.0) else ""))


def model_of(arm, size):
    return f"{ARGS.prefix}--{arm}_{size}--s42"


def trend(values):
    if all(b < a for a, b in zip(values, values[1:])):
        return "decreases monotonically"
    if all(b > a for a, b in zip(values, values[1:])):
        return "increases monotonically"
    return "is not monotonic"


# energy and forces in one figure: rows rmse, p8 and sup, columns energy and force on each test
def combined_figure(rows, captions):
    columns = [(kind, test) for kind in KINDS for test in TESTS]
    fig, axes = plt.subplots(len(METRICS), len(columns), figsize=(20, 15), sharex=True)
    missing = set()
    for c, (kind, test) in enumerate(columns):
        column, unit, scale = KINDS[kind]
        for r, (metric, metric_label) in enumerate(METRICS):
            ax = axes[r, c]
            for arm, (name, colour, marker) in ARMS.items():
                x, y = [], []
                for size in SIZES:
                    row = rows.get((test, model_of(arm, size)))
                    if row is None:
                        missing.add((name, size))
                        continue
                    x.append(int(size[:-1]) * 1000)
                    y.append(float(row[f"{column}_{metric}"]) * scale)
                ax.plot(x, y, "-", color=colour, lw=2.8, marker=marker, ms=11, label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([int(s[:-1]) * 1000 for s in SIZES])
            ax.set_xticklabels(SIZES, fontsize=22)
            log_y_ticks(ax)
            ax.tick_params(axis="y", which="both", labelsize=21)
            ax.tick_params(which="major", length=9, width=1.6)
            ax.tick_params(which="minor", length=5, width=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(1.6)
            if r == 0:
                ax.set_title(f"{kind}, {unit}\n{TEST_TITLE[test]}", fontsize=23)
            if c == 0:
                ax.set_ylabel(metric_label, fontsize=25)
            if r == len(METRICS) - 1:
                ax.set_xlabel(ARGS.xlabel, fontsize=22)
    axes[0, 0].legend(fontsize=21, frameon=False)
    incomplete = " (incomplete)" if missing else ""
    fig.tight_layout()
    fig.savefig(FIGURES / "scaling_energy_force.png", dpi=220)
    plt.close(fig)
    pending = f" Incomplete: pending {', '.join(f'{n} {s}' for n, s in sorted(missing))}." if missing else ""
    captions.append(f"**`scaling_energy_force.png`** (energy and force errors together). Rows: MAE, RMSE, p8 = (mean e^8)^(1/8) and SUP = max over structures; columns: energy (meV/atom) on the RANDOM test set (5,847 structures) and on the MAD-CORE test set (2,000 structures), "
                    f"then force (eV/A) on the same two tests; the tests are evaluated separately, never pooled. "
                    + PROTOCOL.format(label=ARGS.label, validation=ARGS.validation, extra=ARGS.extra) + pending)
    print("wrote scaling_energy_force", incomplete or "(complete)")


TEST_STYLE = {"test_random": ("-", True, "RANDOM test set"), "test_madcore": ("--", False, "MAD-CORE test set")}


# both tests in the same panel: colour is the training set, line style the test (solid filled markers for the random
# test, dashed open markers for the mad-core holdout); rows energy and force, columns mae, rmse, p8 and sup
def both_tests_figure(rows, captions):
    fig, axes = plt.subplots(2, len(METRICS), figsize=(22, 11), sharex=True)
    missing = set()
    for r, (kind, (column, unit, scale)) in enumerate(KINDS.items()):
        for c, (metric, metric_label) in enumerate(METRICS):
            ax = axes[r, c]
            for test, (style, filled, _) in TEST_STYLE.items():
                for arm, (name, colour, marker) in ARMS.items():
                    x, y = [], []
                    for size in SIZES:
                        row = rows.get((test, model_of(arm, size)))
                        if row is None:
                            missing.add((name, size))
                            continue
                        x.append(int(size[:-1]) * 1000)
                        y.append(float(row[f"{column}_{metric}"]) * scale)
                    ax.plot(x, y, style, color=colour, lw=2.8, marker=marker, ms=11, mfc=colour if filled else "white", mew=2.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([int(s[:-1]) * 1000 for s in SIZES])
            ax.set_xticklabels(SIZES, fontsize=22)
            log_y_ticks(ax)
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
                ax.set_xlabel(ARGS.xlabel, fontsize=22)
    arm_handles = [plt.Line2D([], [], color=colour, marker=marker, lw=3, ms=12, label=name) for name, colour, marker in ARMS.values()]
    test_handles = [plt.Line2D([], [], color="0.25", ls=style, lw=3, marker="o", ms=12, mfc="0.25" if filled else "white", mew=2.5, label=label) for style, filled, label in TEST_STYLE.values()]
    legend_style = {"loc": "upper center", "ncol": 2, "fontsize": 22, "title_fontsize": 22, "frameon": False, "handlelength": 4.0, "columnspacing": 2.0}
    fig.legend(handles=arm_handles, title="training set", bbox_to_anchor=(0.31, 1.0), **legend_style)
    fig.legend(handles=test_handles, title="test set", bbox_to_anchor=(0.69, 1.0), **legend_style)
    incomplete = " (incomplete)" if missing else ""
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(FIGURES / "scaling_both_tests.png", dpi=220)
    plt.close(fig)
    pending = f" Incomplete: pending {', '.join(f'{n} {s}' for n, s in sorted(missing))}." if missing else ""
    captions.append(f"**`scaling_both_tests.png`** (both tests in each panel). Rows: energy (meV/atom) and force (eV/A) errors; columns: MAE, RMSE, p8 = (mean e^8)^(1/8) and SUP = max over structures. "
                    f"Colour is the training set (MAD-CORE orange, RANDOM blue); line style is the test set: solid lines with filled markers for the RANDOM test set (5,847 structures), dashed lines with open markers for the MAD-CORE test set (2,000 structures). "
                    f"Each test is evaluated separately, never pooled. " + PROTOCOL.format(label=ARGS.label, validation=ARGS.validation, extra=ARGS.extra) + pending)
    print("wrote scaling_both_tests", incomplete or "(complete)")



def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = {(r["test"], r["model"]): r for r in csv.DictReader(open(TABLE), delimiter="\t")}
    used, description, captions = [], [], []
    # source values and monotonicity statements for every test, quantity, metric and training set
    for kind, (column, unit, scale) in KINDS.items():
        for test, test_label in TESTS.items():
            for metric, metric_label in METRICS:
                for arm, (name, colour, marker) in ARMS.items():
                    y = []
                    for size in SIZES:
                        row = rows.get((test, model_of(arm, size)))
                        if row is None:
                            continue
                        y.append(float(row[f"{column}_{metric}"]) * scale)
                        used.append({"test": test, "arm": arm, "size": size, "model": model_of(arm, size), "metric": f"{column}_{metric}", "value": row[f"{column}_{metric}"], "unit_in_table": "meV/atom" if column == "E" else "meV/A"})
                    if len(y) == len(SIZES):
                        description.append(f"{ARGS.label}, {test_label.split(',')[0]}, {name}: {kind} {metric_label} {trend(y)} with size ({', '.join(f'{v:.3g}' for v in y)} {unit})")
    combined_figure(rows, captions)
    both_tests_figure(rows, captions)
    with open(FIGURES / "scaling_source.tsv", "w") as fh:
        cols = ["test", "arm", "size", "model", "metric", "value", "unit_in_table"]
        fh.write("\t".join(cols) + "\n")
        seen = set()
        for u in used:
            key = (u["test"], u["model"], u["metric"])
            if key not in seen:
                seen.add(key)
                fh.write("\t".join(str(u[c]) for c in cols) + "\n")
    sketch = ("**`sketchmap_energy_scaling_selected.png`, `sketchmap_force_scaling_selected.png`.** Every panel evaluates the same combined 7,847 test structures (random test 5,847 plus MAD-CORE holdout 2,000) placed on the sketch map of the library; "
              "rows identify the training distribution (random or MAD-CORE), columns the training-set size, colours the local mean error per bin (smoothed error sums divided by smoothed counts, shared logarithmic colour scale), "
              f"grey contours the library density, the last column the library density and the number of evaluated structures per bin. Best validation checkpoints ({ARGS.label}), single seed.")
    (FIGURES / "CAPTIONS.md").write_text(f"# Figure captions, {ARGS.label}\n\nSource of all values: `{TABLE}` (verified selected-checkpoint metrics; extract in `scaling_source.tsv`).\n\n" + "\n\n".join(captions) + "\n\n" + sketch + "\n")
    (FIGURES / "SCALING_DESCRIPTION.md").write_text(f"# {ARGS.label}: how the metrics change with training-set size (best validation checkpoints, single seed)\n\n" + "\n".join(f"- {d}" for d in description) + "\n")
    print("\n".join(description))


if __name__ == "__main__":
    main()
