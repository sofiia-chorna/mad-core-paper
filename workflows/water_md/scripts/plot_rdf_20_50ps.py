from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


D = Path(__file__).resolve().parent.parent
CURVES = D / "analysis" / "curves"
FIGURES = D / "analysis" / "figures"
CONDITIONS = (("baseline", "PET-MAD-1.6-M", "C0"), ("lda_corrected", "LDA-corrected", "C1"), ("pbe_corrected", "PBE-corrected", "C2"))
PANELS = (("rdf_oo_20_50ps.tsv", "O–O", (2.2, 4.9)), ("rdf_oh_20_50ps.tsv", "O–H", (1.3, 4.9)), ("rdf_hh_20_50ps.tsv", "H–H", (1.0, 4.9)))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    for ax, (name, title, xlim) in zip(axes, PANELS):
        data = np.genfromtxt(CURVES / name, names=True, delimiter="\t")
        shown = (data["r_A"] >= xlim[0]) & (data["r_A"] <= xlim[1])

        for condition, label, color in CONDITIONS:
            ax.plot(data["r_A"], data[f"{condition}_mean"], color=color, label=label)

        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.05 * max(data[f"{c}_mean"][shown].max() for c, _, _ in CONDITIONS))
        ax.set_title(title)
        ax.set_xlabel("r (Å)")
        ax.set_ylabel("g(r)")

    axes[0].legend()
    fig.tight_layout()

    fig.savefig(FIGURES / "rdf_oo_oh_hh_20_50ps.png", dpi=200)


if __name__ == "__main__":
    main()
