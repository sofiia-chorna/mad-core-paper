import argparse
import collections

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

analysis = "${DEV_ROOT}/phace-atomic/voronoi/out/analysis"
run_dir = "${DEV_ROOT}/phace-atomic/skmatter-run/sketchmap"
coreset = ("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92"
           "/coreset224/out/coreset_224_light.parquet")
N_CORESET = 2 ** 24

MAD_PARTS = ("mad-1.5", "mad-next", "mad_union")
MAD_LABEL = "MAD-NEXT"
CATALYST_PREFIXES = ("OC20", "OC22", "OC25")
ANI_PREFIXES = ("ANI-1", "ANI-2")
RENAME = {"Open_Molecular_Crystals_2025_OMC25": "OMC25",
          "Materials_Project": "Materials Project",
          "mcloudarchive__jk-9v": "Li-ion conductors",
          "Carolina_Materials": "Carolina",
          "MatPES-2025.1": "MatPES",
          "Finding_new_crystal_compounds_using_chemical_similarity": "Chem. similarity"}

# contrasting colors reused in every panel, drawn from the paper's palette (the red
PALETTE5 = ["#C94E3F", "#00A79F", "#F0C93B", "#4B4A86", "#403C3A"]
OTHER_COLOUR = "#7E7368"

# members listed in the color order of PALETTE5 (first member rosy red, ...)
GROUPS = [
    ("MAD-NEXT", [MAD_LABEL]),
    ("Bulk materials", ["OMat24", "Alexandria", "JARVIS"]),
    ("Repositories", ["nomad", "Materials Project", "Materials Cloud"]),
    ("Molecular", ["OMol25", "ANI", "gdb13"]),
    ("Catalysis and interfaces", ["Open Catalyst", "ODAC"]),
]
# datasets drawn in a second pass above the panel's population order, so their many
PANEL_TOP = {"Bulk materials": ["OMat24"]}
# few of its landmark cells stay visible above the lifted dataset
PANEL_TOP_BIGGEST = {"Bulk materials": {"JARVIS": 10000}}

LABEL = {"HEA25_high_entropy_transition-metal_alloys": "HEA25",
         "HEA25S_high_entropy_alloys": "HEA25S",
         "UNEP_v1_2023": "UNEP-v1",
         "MatPES-2025.1": "MatPES",
         "Carolina_Materials": "Carolina",
         "Finding_new_crystal_compounds_using_chemical_similarity": "Chem. similarity",
         "Open_Molecular_Crystals_2025_OMC25": "OMC25",
         "Si_Al_Ti_Seko_PRB_2019": "Si-Al-Ti",
         "AlNiTi_CMS_2019": "AlNiTi",
         "WBe_PRB2019": "W-Be",
         "nomad": "NOMAD",
         "ANI": "ANI-1/2x"}

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="p20d8")
parser.add_argument("--d", type=int, default=8)
parser.add_argument("--out-dir", default=f"{run_dir}/figures")
parser.add_argument("--columns", type=int, default=3)
parser.add_argument("--width", type=float, default=7.0)
parser.add_argument("--height-scale", type=float, default=0.88)
parser.add_argument("--n-show", type=int, default=2 ** 21)
parser.add_argument("--gamma", type=float, default=0.5)
parser.add_argument("--always-top", type=int, default=512)
parser.add_argument("--background", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--reference", type=float, nargs="*",
                    default=[1e2, 1e4, 1e6, 1e8])
parser.add_argument("--smin", type=float, default=2.0)
parser.add_argument("--smax", type=float, default=60.0)
parser.add_argument("--size-power", type=float, default=2.5)
parser.add_argument("--alpha-min", type=float, default=0.35)
parser.add_argument("--alpha-max", type=float, default=0.85)
parser.add_argument("--edge-width", type=float, default=0.12)
args = parser.parse_args()

projection = np.load(f"{run_dir}/projector/projection_2p24_d{args.d}_{args.tag}.npy")
assert projection.shape == (N_CORESET, args.d), projection.shape
coordinates = projection[:, :2].astype(np.float32)

blob = np.load(f"{analysis}/populations.npz")
offsets = blob["level_offsets"]
population = blob["populations"][offsets[24]:offsets[25]].astype(np.float64)
assert population.shape == (N_CORESET,), population.shape
print(f"level-24 populations: sum {population.sum():,.0f}, max {population.max():,.0f}")

source = np.asarray(pq.read_table(coreset, columns=["family"]).column("family")
                    .to_pylist(), dtype=object)
assert source.shape == (N_CORESET,), source.shape


def label_of(name):
    if name in MAD_PARTS or name == "mad-union":
        return MAD_LABEL
    if name.upper().startswith(CATALYST_PREFIXES):
        return "Open Catalyst"
    if name.upper().startswith(ANI_PREFIXES):
        return "ANI"
    if name.startswith("JARVIS"):
        return "JARVIS"
    if name.startswith(("mcloudarchive", "mcloud__")):
        return "Materials Cloud"
    return RENAME.get(name, name)


family = np.asarray([label_of(name) for name in source], dtype=object)

# population-weighted sample without replacement (Efraimidis-Spirakis smallest
generator = np.random.default_rng(args.seed)
weight = population ** args.gamma
keys = -np.log(generator.random(N_CORESET)) / weight
shown = np.argsort(keys)[:args.n_show]
top = np.argsort(population)[::-1][:args.always_top]
shown = np.union1d(shown, top)
print(f"showing {shown.size:,} of {N_CORESET:,} cells "
      f"(gamma {args.gamma}, forced top {args.always_top}); shown cells hold "
      f"{population[shown].sum() / population.sum():.1%} of the collection")

background = np.sort(generator.choice(N_CORESET, args.background, replace=False))

counts = collections.Counter(family[shown].tolist())
full_counts = collections.Counter(family.tolist())
size_fraction = (np.log10(population) / np.log10(population.max())) ** args.size_power
marker_size = args.smin + (args.smax - args.smin) * size_fraction
# opacity grows with cell size, so the big cells on top keep their true color
point_alpha = args.alpha_min + (args.alpha_max - args.alpha_min) * size_fraction
# edge stroke also grows with cell size the mass of tiny points carries almost no
point_edge = args.edge_width * (0.15 + 0.85 * size_fraction)

named = {name for _, members in GROUPS for name in members}
panels = GROUPS + [("Other datasets", ["__other__"])]

n_panels = len(panels)
rows = int(np.ceil(n_panels / args.columns))

MARGIN = 0.10
figure, axes = plt.subplots(rows, args.columns,
                            figsize=(args.width, args.width / args.columns * rows * args.height_scale),
                            sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()

shown_family = family[shown]
for index, (title, members) in enumerate(panels):
    axis = axes[index]
    axis.scatter(coordinates[background, 0], coordinates[background, 1], s=0.7,
                 c="0.80", linewidths=0, rasterized=True)

    colour_of = {name: (OTHER_COLOUR if name == "__other__"
                        else PALETTE5[i % len(PALETTE5)])
                 for i, name in enumerate(members)}
    if members == ["__other__"]:
        in_panel = shown[~np.isin(shown_family, list(named))]
        point_colour = np.full(in_panel.size, OTHER_COLOUR, dtype=object)
    else:
        mask = np.isin(shown_family, members)
        in_panel = shown[mask]
        point_colour = np.asarray([colour_of[f] for f in shown_family[mask]],
                                  dtype=object)
    # the biggest cells land on top PANEL_TOP members get a second pass above
    priority = PANEL_TOP.get(title, [])
    panel_fam = shown_family[np.searchsorted(shown, in_panel)]
    lifted = np.full(in_panel.size, -1)
    for rank, name in enumerate(priority):
        lifted[panel_fam == name] = rank
    for name, n_biggest in PANEL_TOP_BIGGEST.get(title, {}).items():
        member_pos = np.flatnonzero(panel_fam == name)
        lift = member_pos[np.argsort(population[in_panel[member_pos]])[::-1][:n_biggest]]
        lifted[lift] = len(priority)
    for rank in sorted(set(lifted.tolist())):
        mask = lifted == rank
        subset = in_panel[mask]
        if subset.size == 0:
            continue
        sub_colour = point_colour[mask]
        order = np.argsort(population[subset])
        drawn = subset[order]
        rgba = np.asarray([matplotlib.colors.to_rgba(c) for c in sub_colour[order]])
        rgba[:, 3] = point_alpha[drawn]
        axis.scatter(coordinates[drawn, 0], coordinates[drawn, 1],
                     s=marker_size[drawn], color=rgba,
                     edgecolors="black", linewidths=point_edge[drawn],
                     rasterized=True)

    axis.text(0.04, 0.96, title, transform=axis.transAxes, ha="left", va="top",
              fontsize=10.0, fontweight="normal")
    dataset_legend_loc = "center right" if index == 0 else "lower right"
    axis.legend(handles=[Patch(facecolor=colour_of[name], edgecolor="0.15",
                               linewidth=0.4,
                               label=(f"{len(full_counts) - len(named)} datasets"
                                      if name == "__other__" else LABEL.get(name, name)))
                         for name in members],
                loc=dataset_legend_loc, frameon=False, fontsize=9.0, handlelength=0.9,
                handletextpad=0.35, labelspacing=0.35, borderpad=0.2,
                borderaxespad=0.3)

    if index == 0:
        reference_size = args.smin + (args.smax - args.smin) * (
            np.log10(args.reference) / np.log10(population.max())) ** args.size_power
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

    low = coordinates.min(0)
    high = coordinates.max(0)
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
out = f"{args.out_dir}/voronoi_groups24_d{args.d}_{args.tag}.png"
figure.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
figure.savefig(out.replace(".png", ".pdf"), dpi=400, bbox_inches="tight",
               pad_inches=0.02)
print("wrote", out.replace(".png", ".pdf"))

for title, members in panels:
    print(f"  {title}")
    for name in members:
        shown_n = (int((~np.isin(shown_family, list(named))).sum())
                   if name == "__other__" else counts.get(name, 0))
        total_n = (sum(v for k, v in full_counts.items() if k not in named)
                   if name == "__other__" else full_counts.get(name, 0))
        print(f"      {LABEL.get(name, name):<26} {shown_n:>9,} shown "
              f"of {total_n:>10,} cells")
print("ten biggest cells shown:")
for i in np.argsort(population)[::-1][:10]:
    print(f"  rank {i:>9,} pop {population[i]:>12,.0f} {family[i]}")
