from __future__ import annotations

import colorsys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MAIN = {
    "silver":        "#CAC5C5",
    "mid_gray":      "#8A8483",
    "rosy_copper":   "#C94E3F",
    "stormy_teal":   "#31717E",
    "tropical_teal": "#49A3A0",
    "apricot":       "#F1994F",
    "petrol":        "#4D6F7C",
    "brick":         "#B05C4C",
    "indigo":        "#4B4A86",
    "charcoal":      "#403C3A",
}

EPFL = {
    "rouge":     "#FF0000",
    "groseille": "#B51F1F",
    "canard":    "#007480",
    "leman":     "#00A79F",
    "ardoise":   "#413D3A",
    "perle":     "#CAC7C7",
}

# Palette used by figures/accrete.pdf (recovered from its fill operators). Close to
# MAIN but not identical -- keep the two apart so figures stay internally consistent.
ACCRETE = {
    "charcoal": "#413C3A",
    "petrol":   "#42707D",
    "brick":    "#BC5647",
    "apricot":  "#FF923B",
    "indigo":   "#4B4A8B",
}

RAMPS = {
    # multi-hue sequential ramp over the whole ACCRETE palette, dark -> bright, which
    # is the order its luminances already fall in (~61, 82, 100, 115, 169). Runs
    # viridis-style (low = dark) so the light "absent" gray can't be mistaken for a
    # low count, and so skewed data still separates across the dark half.
    "accrete_seq": [ACCRETE["charcoal"], ACCRETE["indigo"], ACCRETE["petrol"],
                    ACCRETE["brick"], ACCRETE["apricot"]],
    # MAIN-palette families (dark -> base -> light), for shaded same-family categories
    "copper": ["#7f2c21", MAIN["rosy_copper"], "#e3a89e"],
    "teal":   ["#22545e", MAIN["stormy_teal"], MAIN["tropical_teal"], "#a6d4d1"],
    "gray":   ["#6c6665", MAIN["mid_gray"], MAIN["silver"], "#e9e7e7"],
    # classical EPFL
    "red":    ["#4a0c0c", EPFL["groseille"], EPFL["rouge"], "#ff8a80", "#ffd2cc"],
    "slate":  [EPFL["ardoise"], "#8a827d", EPFL["perle"]],
}


def cmap(name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"epfl_{name}", RAMPS[name])


def ramp(name: str, n: int, lo: float = 0.04, hi: float = 0.90) -> list:
    c = cmap(name)
    return [c(x) for x in np.linspace(lo, hi, n)]


QUAL_HUE = {"warm": (0.00, 0.15), "cool": (0.45, 0.58), "violet": (0.76, 0.93),
            "full": (0.00, 0.92)}   # 'full' spans the spectrum for many distinct single-family categories
_S = (0.90, 0.52, 0.74)
_V = (0.60, 0.94, 0.80)


def qualitative(family: str, n: int) -> list:
    h0, h1 = QUAL_HUE[family]
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        h = (h0 + t * (h1 - h0)) % 1.0
        out.append(colorsys.hsv_to_rgb(h, _S[i % 3], _V[i % 3]))
    return out
