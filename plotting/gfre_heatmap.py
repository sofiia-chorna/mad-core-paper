import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Rectangle

HEATMAP_CMAP = plt.colormaps["viridis"]

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.6,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

DISPLAY_ORDER = [
    "pet-mad",
    "mace-omat",
    "uma-omat",
    "dpa-omat",
    "dpa-omol",
    "orb-v3-omat",
    "phace-atomic",
]

PRETTY_LABELS = {
    "pet-mad":      "PET-MAD-1.5",
    "mace-omat":    "MACE-OMat",
    "uma-omat":     "UMA-OMat",
    "dpa-omat":     "DPA3.2-OMat",
    "dpa-omol":     "DPA3-OMol",
    "orb-v3-omat":  "ORB-OMat",
    "phace-atomic": "ecumetric",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def build_matrix(results):
    labels = DISPLAY_ORDER
    err = pd.DataFrame(np.nan, index=labels, columns=labels)
    ann = pd.DataFrame('',    index=labels, columns=labels, dtype="U20")
    for entry in results.values():
        a, b = entry["A"], entry["B"]
        if a not in labels or b not in labels:
            continue
        err.loc[a, b] = entry["A_to_B"]
        ann.loc[a, b] = f"{entry['A_to_B']:.2f}"
        if a != b:
            err.loc[b, a] = entry["B_to_A"]
            ann.loc[b, a] = f"{entry['B_to_A']:.2f}"
    present = [m for m in labels if not err.loc[m].isna().all()]
    return err.loc[present, present], ann.loc[present, present]


def _draw(ax, matrix, annot, cbar_ax=None, vmax=1.0, show_yticks=True,
          annot_fontsize=16, show_borders=True):
    values = matrix.to_numpy(dtype=float)
    hm = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]), labels=matrix.index if show_yticks else [])
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    if show_borders:
        ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rgba = HEATMAP_CMAP(values[i, j] / vmax)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.52 else "black"
            ax.text(j, i, annot.iat[i, j], ha="center", va="center",
                    fontsize=annot_fontsize, color=text_color)
    if show_borders:
        ax.add_patch(Rectangle((-0.5, -0.5), matrix.shape[1], matrix.shape[0],
                               fill=False, edgecolor='black', linewidth=1.0))
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, pad=8, labelsize=24)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=40, fontsize=24)
    return hm

