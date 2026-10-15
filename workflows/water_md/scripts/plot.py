import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import BLOCKS, DISCARDED, LATE_WINDOW_START_PS, WINDOW_START_PS, block_key  # noqa: E402
from observables import NEAREST_OH_EDGES, RDF_BIN, RDF_CENTRES, RDF_RMAX, SECOND_OH_EDGES  # noqa: E402
from water import COLORS, CONDITIONS, D, LABELS, TEMPERATURE_K  # noqa: E402


RUN_STYLE = {"lw": 0.7, "alpha": 0.25}
MEAN_STYLE = {"lw": 2.0}
PANELS = (("oo", "O–O", 2.2), ("oh", "O–H", 0.6), ("hh", "H–H", 1.0))
BLOCK_STYLES = (":", "-.", "--", "-")


def load(analysis):
    records = json.loads((analysis / "production_runs.json").read_text())
    curves = np.load(analysis / "production_curves.npy", allow_pickle=True).item()

    grouped = {}
    for condition in CONDITIONS:
        names = [r["run"] for r in records if r["condition"] == condition and curves.get(r["run"])]
        grouped[condition] = [curves[n] for n in names]

    return records, grouped


def draw(ax, x, condition, per_run, get):
    values = [get(c) for c in per_run]
    values = [v for v in values if v is not None]

    for v in values:
        ax.plot(x, v, color=COLORS[condition], **RUN_STYLE)

    if values:
        ax.plot(x, np.nanmean(values, axis=0), color=COLORS[condition], label=LABELS[condition], **MEAN_STYLE)


def save(fig, figures, name):
    for suffix in ("pdf", "png"):
        fig.savefig(figures / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def window_curve(window, key):
    return lambda c: c[window][key] if window in c else None


def rdf_figure(grouped, figures, window, name):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="white")

    for ax, (key, title, x_min) in zip(axes, PANELS):
        for condition in CONDITIONS:
            draw(ax, RDF_CENTRES, condition, grouped[condition], window_curve(window, key))

        ax.set_xlim(x_min, RDF_RMAX)
        ax.set_xlabel("r (Å)")
        ax.set_title(title)
        ax.axhline(1.0, color="0.6", lw=0.6)
        ax.set_ylim(bottom=0.0)

        # the intramolecular O-H peak is far above the intermolecular structure, so the axis is cut and the peak named
        if key == "oh":
            means = [np.mean([c[window][key] for c in runs if window in c], axis=0) for runs in grouped.values() if runs]
            top = max(m[RDF_CENTRES > 1.3].max() for m in means)
            peak = max(m.max() for m in means)
            ax.set_ylim(0.0, 1.35 * top)
            ax.text(0.97, 0.95, f"intramolecular pairs included\npeak up to g = {peak:.0f}, cut", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="0.3")

        if key == "hh":
            ax.text(0.97, 0.95, "intramolecular pairs included", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.3")

    axes[0].set_ylabel("g(r)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, figures, name)


def distance_figure(grouped, figures, window, name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), facecolor="white")
    nearest_x = 0.5 * (NEAREST_OH_EDGES[1:] + NEAREST_OH_EDGES[:-1])
    second_x = 0.5 * (SECOND_OH_EDGES[1:] + SECOND_OH_EDGES[:-1])

    for condition in CONDITIONS:
        draw(axes[0], nearest_x, condition, grouped[condition], window_curve(window, "nearest_oh"))
        draw(axes[1], second_x, condition, grouped[condition], window_curve(window, "second_oh"))

    axes[0].set_xlim(0.85, 1.35)
    axes[0].set_title("nearest O–H distances")
    axes[1].set_xlim(1.0, 3.8)
    axes[1].set_title("second-nearest O–H distances")

    for ax in axes:
        ax.set_xlabel("distance (Å)")
        ax.set_ylim(bottom=0.0)

    axes[0].set_ylabel("probability density (1/Å)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, figures, name)


# total, H and O kinetic temperatures in one-picosecond bins, the discarded first 5 ps marked
def temperature_figure(grouped, figures):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="white", sharey=True)

    for ax, (key, title) in zip(axes, (("all", "all atoms"), ("H", "H"), ("O", "O"))):
        for condition in CONDITIONS:
            runs = [c for c in grouped[condition] if "temperature_bins" in c]
            if runs:
                draw(ax, runs[0]["temperature_bins"]["centres"], condition, runs, lambda c, k=key: c["temperature_bins"][k])

        ax.axhline(TEMPERATURE_K, color="0.6", lw=0.6)
        ax.axvline(WINDOW_START_PS, color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("time (ps)")
        ax.set_title(title)

    axes[0].set_ylabel("kinetic temperature (K)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, figures, "temperature_water")


# O-O g of r of each time block, mean over runs, one panel per condition
def block_figure(grouped, figures):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor="white")

    for ax, condition in zip(axes, CONDITIONS):
        for (lo, hi), style in zip((DISCARDED,) + BLOCKS, BLOCK_STYLES):
            key = block_key(lo, hi)
            curves = [c[key]["oo"] for c in grouped[condition] if key in c]
            if curves:
                label = f"{lo:g}–{hi:g} ps" + (" (discarded)" if (lo, hi) == DISCARDED else "")
                ax.plot(RDF_CENTRES, np.mean(curves, axis=0), color=COLORS[condition], ls=style, lw=1.5, label=label)

        ax.set_xlim(2.2, RDF_RMAX)
        ax.set_ylim(bottom=0.0)
        ax.axhline(1.0, color="0.6", lw=0.6)
        ax.set_xlabel("r (Å)")
        ax.set_title(f"O–O, {LABELS[condition]}")
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("g(r)")
    fig.tight_layout()
    save(fig, figures, "rdf_oo_blocks_water")


def write_captions(records, grouped, figures):
    densities = sorted({round(r["density_g_cm3"], 4) for r in records if "density_g_cm3" in r})
    density = " and ".join(f"{d:.4f}" for d in densities)
    cell = " and ".join(f"{c:.4f}" for c in sorted({round(r["cell_A"], 6) for r in records if "cell_A" in r}))
    runs = ", ".join(f"{LABELS[c]} {len(grouped[c])}" for c in CONDITIONS)
    system = (f"32 H2O in a fixed periodic {cell} Å cube, the Atomistic Cookbook thermostat start with its molecular centres scaled to "
              f"density {density} g/cm³, NVT at 300 K with an SVR thermostat (τ = 10 fs) and a 0.25 fs timestep.")
    runs_text = (f"Runs per condition: {runs}. Faint lines are individual velocity-seeded runs, thick lines their arithmetic mean. "
                 "PET-MAD-1.6-M is PET-MAD-M v1.6.0 alone (the corrected-attention-mask export). The approximate LDA and PBE potentials are PET-MAD-1.6-M minus the learned "
                 "r²SCAN−LDA or r²SCAN−PBE correction trained on 18,000 MAD-CORE structures (validation 1,000), evaluated on the same coordinates at every step; "
                 "they are machine-learned approximations, not DFT trajectories.")

    captions = [
        "# Figure captions",
        "",
        "## rdf_water",
        "",
        f"O–O, O–H and H–H radial distribution functions, {WINDOW_START_PS:g}–50 ps of each run. {system} {runs_text} "
        f"Bins of {RDF_BIN} Å up to {RDF_RMAX} Å, below half the cell width, normalised to the ideal gas at the cell density. "
        "The O–H and H–H functions include intramolecular pairs, and the O–H axis is cut below the intramolecular peak.",
        "",
        "## oh_distance_water",
        "",
        f"Nearest and second-nearest O–H distances, {WINDOW_START_PS:g}–50 ps. For every H in every frame, the distance to its nearest O "
        f"and to its second-nearest O under the minimum image, as probability densities. {system} {runs_text}",
        "",
        "## rdf_water_20_50ps and oh_distance_water_20_50ps",
        "",
        f"The same as rdf_water and oh_distance_water over {LATE_WINDOW_START_PS:g}–50 ps of each run, a supplementary window added after "
        "the block analysis showed the LDA-corrected structure still relaxing through 5–20 ps. "
        f"Density {density} g/cm³. The O–H and H–H functions include intramolecular pairs.",
        "",
        "## temperature_water",
        "",
        "Kinetic temperature of all atoms, of H and of O in one-picosecond bins, from the saved velocities, each subset over its share of "
        f"the 3N−3 degrees of freedom. The dashed line marks the end of the discarded first {WINDOW_START_PS:g} ps, the grey line the 300 K bath. "
        f"{system} {runs_text}",
        "",
        "## rdf_oo_blocks_water",
        "",
        "O–O radial distribution function of each time block, the mean over the runs of a condition: the discarded first 5 ps and the "
        f"production blocks {', '.join(f'{lo:g}–{hi:g} ps' for lo, hi in BLOCKS)}. {system}",
    ]
    (figures / "CAPTIONS.md").write_text("\n".join(captions) + "\n")


# usage plot.py, reads the analysis folder and writes the figures and their captions next to it
def main():
    analysis = Path(sys.argv[1]) if len(sys.argv) > 1 else D / "analysis"
    figures = analysis / "figures"
    figures.mkdir(exist_ok=True)
    records, grouped = load(analysis)

    rdf_figure(grouped, figures, "window", "rdf_water")
    distance_figure(grouped, figures, "window", "oh_distance_water")
    rdf_figure(grouped, figures, "late", "rdf_water_20_50ps")
    distance_figure(grouped, figures, "late", "oh_distance_water_20_50ps")
    temperature_figure(grouped, figures)
    block_figure(grouped, figures)
    write_captions(records, grouped, figures)

    print(figures)


if __name__ == "__main__":
    main()
