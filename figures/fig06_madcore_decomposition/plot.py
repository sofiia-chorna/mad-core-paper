import os
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "plotting"))
import epfl

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures" / "coreset224-kset.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

N_TOP = 15
OTHER_COLOR = "#B8B2AB"

# size-ordered stack: strong hues for the big families, no two similar hues adjacent
FAMILY_COLOR_LIST = [
    "#2E6DB4",  # OMat24            blue
    "#DE7E20",  # Alexandria        burnt orange
    "#178F82",  # JARVIS-QETB       teal
    "#7A3FC0",  # MP-ALOE           purple
    "#F5C542",  # OQMD              gold
    "#4E6B78",  # NOMAD             slate
    "#4CAF6E",  # OMol25            green
    "#C94E3F",  # MAD-NEXT          rosy red
    "#C2A233",  # OQMD-CFID         dark mustard
    "#6BA6E0",  # Materials Project sky blue
    "#5E3C99",  # MatPES            dark violet
    "#9BD69A",  # Carolina          light green
    "#C72E82",  # ANI-1             magenta
    "#2F2C2A",  # OC25              near-black
    "#EE9AC6",  # Alex-MP-20        pink
]

CAT_ORDER = ["3D bulk", "molecules", "clusters", "surfaces/2D", "molecular crystals",
             "solvated", "atom"]
CAT_COLOR = {
    "3D bulk": epfl.ACCRETE["petrol"], "molecules": epfl.ACCRETE["apricot"],
    "clusters": epfl.ACCRETE["indigo"], "surfaces/2D": epfl.ACCRETE["brick"],
    "molecular crystals": "#63C3B9", "solvated": "#C72E82",
    "atom": epfl.ACCRETE["charcoal"],
}
CAT_MARKER = {"3D bulk": "o", "molecules": "s", "clusters": "D", "surfaces/2D": "^",
              "molecular crystals": "v", "solvated": "P", "atom": "X"}

ELFAM_ORDER = ["hydrogen", "s-block metals", "p-block elements", "transition metals",
               "lanthanides/actinides"]
ELFAM_COLOR = {
    "hydrogen": epfl.ACCRETE["charcoal"],
    "s-block metals": epfl.ACCRETE["apricot"],
    "p-block elements": epfl.ACCRETE["petrol"],
    "transition metals": epfl.ACCRETE["brick"],
    "lanthanides/actinides": epfl.ACCRETE["indigo"],
}

DISPLAY = {
    "OMat24": "OMat24", "Alexandria": "Alexandria", "JARVIS_QE_TB": "JARVIS-QETB",
    "MP-ALOE": "MP-ALOE", "JARVIS_OQMD_no_CFID": "OQMD", "nomad": "NOMAD",
    "OMol25": "OMol25", "mad-union": "MAD-NEXT", "mad-next": "MAD-NEXT",
    "JARVIS_CFID_OQMD": "OQMD-CFID", "Materials_Project": "Materials Project",
    "MatPES-2025.1": "MatPES", "Carolina_Materials": "Carolina Materials",
    "ANI-1": "ANI-1", "OC25": "OC25", "Alex_MP-20": "Alex-MP-20",
    "Finding_new_crystal_compounds_using_chemical_similarity": "Chem-sim crystals",
    "JARVIS_ALIGNN_FF": "JARVIS-ALIGNN", "UNEP_v1_2023": "UNEP-v1",
    "OC20_S2EF_train_all": "OC20-S2EF", "QM-22": "QM-22",
}

FS_TICK = 15.0
FS_LABEL = 19.5
FS_LEGEND = 15.5
FS_PANEL = 20.0


def read_tsv(name):
    lines = (HERE / "data" / name).read_text().splitlines()
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


def by_level(name):
    """columns of a k_set_size x quantity table, the values at k = 2^1 .. 2^24"""
    header, rows = read_tsv(name)
    return {h: np.asarray([float(r[i]) if h == "epsilon" else int(r[i]) for r in rows])
            for i, h in enumerate(header)}


def bars(ax, keys, cum, colors, ks, xlab=True):
    x = np.arange(len(ks), dtype=float)
    bottom = np.zeros(len(ks))
    for k in keys:
        row = np.asarray([cum[k][i] / kk for i, kk in enumerate(ks)])
        ax.bar(x, row, bottom=bottom, width=0.90, color=colors[k], linewidth=0)
        bottom += row
    ax.set_xticks(x)
    ax.set_xticklabels([f"$2^{{{e + 1}}}$" for e in range(len(ks))], fontsize=FS_TICK)
    ax.tick_params(axis="x", pad=5, length=0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    if xlab:
        ax.set_xlabel("$k$-set size", fontsize=FS_LABEL)
    ax.set_ylim(0, 1)
    ax.margins(x=0.0)


def main():
    ks = [1 << e for e in range(1, 25)]

    # structures of each dataset group in every k-set, the largest N_TOP drawn by name
    _, rows = read_tsv("group_counts.tsv")
    gcum = {}
    for k, group, count in rows:
        gcum.setdefault(group, {})[int(k)] = int(count)
    gcum = {g: np.asarray([v[k] for k in ks]) for g, v in gcum.items()}
    total = {g: v[-1] for g, v in sorted(gcum.items())}
    top = sorted(total, key=lambda g: -total[g])[:N_TOP]
    n_groups = len(total)
    fkeys = top + ["other"]
    fcum = {g: gcum[g] for g in top}
    fcum["other"] = sum(v for g, v in gcum.items() if g not in top)
    fcol = {k: (FAMILY_COLOR_LIST[i] if k != "other" else OTHER_COLOR)
            for i, k in enumerate(fkeys)}

    ccum = by_level("category_counts.tsv")
    ckeys = [k for k in CAT_ORDER if k in ccum]
    elfam = by_level("element_family_counts.tsv")
    eps = by_level("coverage_radius.tsv")["epsilon"]

    fig = plt.figure(figsize=(14.4, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.56, wspace=0.155,
                          left=0.065, right=0.985, top=0.87, bottom=0.07)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])

    # ---- a: dataset families
    bars(axa, fkeys, fcum, fcol, ks)
    axa.set_xticklabels([f"$2^{{{e + 1}}}$" for e in range(len(ks))], fontsize=17.5)
    axa.tick_params(axis="x", pad=10, length=0)
    axa.set_ylabel("fraction of $k$-set", fontsize=FS_LABEL)
    la = [DISPLAY.get(k, k) if k != "other" else f"other ({n_groups - N_TOP} datasets)"
          for k in fkeys]
    h = [Patch(facecolor=fcol[k], edgecolor="0.15", linewidth=0.4) for k in fkeys]
    axa.legend(h, la, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
               mode="expand", ncol=6, fontsize=FS_LEGEND, frameon=False,
               handlelength=1.0, handletextpad=0.35, labelspacing=0.34,
               columnspacing=0.8)

    # ---- b: chemical families of the elements
    etot = sum(elfam[f] for f in ELFAM_ORDER)
    x = np.arange(len(ks), dtype=float)
    bottom = np.zeros(len(ks))
    for f in ELFAM_ORDER:
        row = np.asarray([elfam[f][i] / etot[i] for i in range(len(ks))])
        axb.bar(x, row, bottom=bottom, width=0.90, color=ELFAM_COLOR[f], linewidth=0)
        bottom += row
    axb.set_xticks(x[1::2])
    axb.set_xticklabels([f"$2^{{{e + 1}}}$" for e in range(len(ks))][1::2],
                        fontsize=FS_TICK)
    axb.tick_params(axis="x", pad=5, length=0)
    axb.tick_params(axis="y", labelsize=FS_TICK)
    axb.set_xlabel("$k$-set size", fontsize=FS_LABEL)
    axb.set_ylabel("fraction of element counts", fontsize=FS_LABEL)
    axb.set_ylim(0, 1)
    axb.margins(x=0.0)
    h = [Patch(facecolor=ELFAM_COLOR[f], edgecolor="0.15", linewidth=0.4)
         for f in ELFAM_ORDER]
    axb.legend(h, ELFAM_ORDER, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0),
               mode="expand", ncol=2, fontsize=FS_LEGEND, frameon=False,
               handlelength=1.0, handletextpad=0.35, labelspacing=0.34,
               columnspacing=0.8)

    # ---- c: category growth, log-log
    for k in ckeys:
        y = list(ccum[k])
        axc.plot(ks, y, marker=CAT_MARKER[k], ms=6.0, lw=2.8, color=CAT_COLOR[k],
                 label=k, markeredgewidth=0)
    axc.set_xscale("log", base=2)
    axc.set_yscale("log")
    axc.set_xticks([1 << e for e in (1, 4, 8, 12, 16, 20, 24)])
    axc.set_xticklabels([f"$2^{{{e}}}$" for e in (1, 4, 8, 12, 16, 20, 24)],
                        fontsize=FS_TICK)
    axc.tick_params(axis="y", labelsize=FS_TICK)
    axc.minorticks_off()
    axc.set_xlabel("$k$-set size", fontsize=FS_LABEL)
    axc.set_ylabel("selections in category", fontsize=FS_LABEL)
    axc.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.0), mode="expand",
               ncol=3, fontsize=FS_LEGEND, frameon=False, handlelength=1.4,
               handletextpad=0.35, labelspacing=0.34, columnspacing=0.7)

    # coverage-radius inset in the empty upper-left corner of c
    axi = axc.inset_axes([0.085, 0.60, 0.33, 0.34])
    axi.set_facecolor("white")
    axi.plot(ks, list(eps), lw=1.8, marker="o", ms=2.8,
             color=epfl.ACCRETE["charcoal"])
    axi.set_xscale("log", base=2)
    axi.set_yscale("log")
    axi.set_xticks([1 << e for e in (1, 12, 24)])
    axi.set_xticklabels([f"$2^{{{e}}}$" for e in (1, 12, 24)], fontsize=11.5)
    axi.set_yticks([10, 1])
    axi.set_yticklabels(["10", "1"], fontsize=11.5)
    axi.minorticks_off()
    axi.tick_params(length=2, pad=2)
    axi.set_xlabel("$k$-set size", fontsize=12.0, labelpad=1)
    axi.set_ylabel("$\\epsilon(k)$", fontsize=12.0, labelpad=1)
    axi.set_title("coverage radius", fontsize=12.5, pad=3)
    for s in axi.spines.values():
        s.set_linewidth(0.7)

    for pnl, ax, xoff in (("a", axa, -0.048), ("b", axb, -0.115), ("c", axc, -0.115)):
        ax.text(xoff, 1.28 if pnl != "a" else 1.24, pnl, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="top", ha="left")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
