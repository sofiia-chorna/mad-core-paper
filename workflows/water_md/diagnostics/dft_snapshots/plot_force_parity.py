from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare import parse_aims

HERE = Path(__file__).resolve().parent
AIMS = HERE / "aims"
TABLES = HERE / "tables"
FIGURES = HERE / "figures"

PANELS = (("pw-lda", "delta_lda", "approx. LDA vs LDA"), ("pbe", "delta_pbe", "approx. PBE vs PBE"))


def main():
    arrays = np.load(TABLES / "model_predictions.npz")
    names = sorted({k.split("/")[1] for k in arrays.files if k.startswith("snapshots/")})

    # one limit for both panels
    lim = 0.0
    for functional, delta, _ in PANELS:
        for name in names:
            dft = parse_aims(AIMS / name / functional / "aims.out")["forces"]
            model = arrays[f"snapshots/{name}/baseline"] - arrays[f"snapshots/{name}/{delta}"]
            lim = max(lim, 1.05 * np.abs(dft).max(), 1.05 * np.abs(model).max())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
    for ax, (functional, delta, title) in zip(axes, PANELS):
        dft = np.concatenate([parse_aims(AIMS / name / functional / "aims.out")["forces"].ravel() for name in names])
        model = np.concatenate(
            [(arrays[f"snapshots/{name}/baseline"] - arrays[f"snapshots/{name}/{delta}"]).ravel() for name in names]
        )
        mae = np.abs(model - dft).mean()

        ax.scatter(dft, model, s=4, alpha=0.5, color="C0", rasterized=True)
        ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("DFT force (eV/Å)")
        ax.set_ylabel("model force (eV/Å)")
        ax.text(0.04, 0.95, f"MAE {mae * 1000:.0f} meV/Å", transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(FIGURES / "force_parity_snapshots.png", dpi=200)
    fig.savefig(FIGURES / "force_parity_snapshots.pdf")


if __name__ == "__main__":
    main()
