from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


D = Path(__file__).resolve().parent.parent
CURVES = D / "analysis" / "curves"
FIGURES = D / "analysis" / "figures"
BASELINE = ("baseline", "PET-MAD-1.6-M", "C0")
PANELS = (("lda_corrected", "approx. LDA (PET-MAD-1.6-M minus delta_LDA, 18k)", "C1"), ("pbe_corrected", "approx. PBE (PET-MAD-1.6-M minus delta_PBE, 18k)", "C2"))
SEEDS = range(5)
XLIM = (2.2, 4.9)


# O-O radial distribution over 20 to 50 ps, one panel per approximate functional next to the baseline, faint lines are the five replicas
def main():
    data = np.genfromtxt(CURVES / "rdf_oo_20_50ps.tsv", names=True, delimiter="\t")
    shown = (data["r_A"] >= XLIM[0]) & (data["r_A"] <= XLIM[1])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    top = 0.0

    for ax, (condition, label, color) in zip(axes, PANELS):
        for cond, lab, col in (BASELINE, (condition, label, color)):
            for seed in SEEDS:
                ax.plot(data["r_A"], data[f"{cond}_seed{seed}"], color=col, alpha=0.25, linewidth=0.7)
            ax.plot(data["r_A"], data[f"{cond}_mean"], color=col, label=lab, linewidth=1.8)
            top = max(top, data[f"{cond}_mean"][shown].max(), max(data[f"{cond}_seed{s}"][shown].max() for s in SEEDS))

        ax.set_xlim(*XLIM)
        ax.set_xlabel("r (Å)")
        ax.set_title("g(r) O–O, 20 to 50 ps")
        ax.legend(fontsize=8)

    axes[0].set_ylim(0.0, 1.05 * top)
    axes[0].set_ylabel("g(r)")
    fig.tight_layout()
    fig.savefig(FIGURES / "rdf_oo_lda_pbe_panels_20_50ps.png", dpi=200)


if __name__ == "__main__":
    main()
