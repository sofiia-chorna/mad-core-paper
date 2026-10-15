
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "plotting"))
from epfl import MAIN

LEVEL_COLORS = [MAIN["rosy_copper"], MAIN["stormy_teal"], MAIN["charcoal"],
                MAIN["tropical_teal"]]
POINT_GRAY = "#AAA4A4"   # between MAIN silver and mid_gray
CIRCLE_COLOR = MAIN["stormy_teal"]

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# %%
rng = np.random.default_rng(4)
t = rng.uniform(0, 1, 60)
filament = np.stack([5.6 - 3.2 * t,
                     0.35 + 2.6 * t**2], axis=1) + rng.normal(0, 0.09, (60, 2))
pts = np.vstack([
    rng.normal([1.2, 1.5], [0.42, 0.28], (90, 2)),
    rng.normal([4.7, 3.7], [0.55, 0.72], (55, 2)),
    rng.normal([2.6, 4.35], [0.33, 0.22], (45, 2)),
    filament,
    rng.uniform([0.1, -0.1], [6.5, 5.4], (45, 2)),
    np.array([[6.75, 5.15], [-0.15, 5.3]]),
])


def run_fps(X, n_select, seed_idx=0):
    dists = np.sqrt(((X[:, None] - X[None, :]) ** 2).sum(axis=2))
    selected = [seed_idx]
    min_dist = dists[seed_idx].copy()
    min_dist[seed_idx] = 0.0
    coverages = [min_dist.max()]
    separations = [np.inf]
    for _ in range(n_select - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, dists[next_idx])
        min_dist[selected] = 0.0
        coverages.append(min_dist.max())
        sel_d = dists[np.ix_(selected, selected)]
        np.fill_diagonal(sel_d, np.inf)
        separations.append(sel_d.min())
    return selected, np.array(coverages), np.array(separations)


max_k = 4
n_select = 2 ** max_k + 1          # one extra: the "next pick" of the last panel
seed_idx = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
fps_order, coverages, separations = run_fps(pts, n_select, seed_idx)
all_dists = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=2))


def entry_level(i):
    """Level at which ordering position i (0-based) enters the hierarchy."""
    return 0 if i < 2 else int(np.ceil(np.log2(i + 1))) - 1


# %%
show_ks = [1, 2, 3, 4]
n_panels = len(show_ks)

pad = 0.55
xlim = (pts[:, 0].min() - pad, pts[:, 0].max() + pad)
ylim = (pts[:, 1].min() - pad, pts[:, 1].max() + pad)
cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
half = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
xlim, ylim = (cx - half, cx + half), (cy - half, cy + half)


def label_offset(pt, avoid=None, mag=13):
    """Pick a label direction that stays inside the axes and points away
    from any arrows attached to this point (avoid = list of unit vectors)."""
    best, best_score = None, -1e9
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            # keep the label inside the panel
            if dx > 0 and pt[0] + 1.6 > xlim[1]:
                continue
            if dx < 0 and pt[0] - 1.6 < xlim[0]:
                continue
            if dx == 0 and (pt[0] - 0.8 < xlim[0] or pt[0] + 0.8 > xlim[1]):
                continue
            if dy > 0 and pt[1] + 1.0 > ylim[1]:
                continue
            if dy < 0 and pt[1] - 1.0 < ylim[0]:
                continue
            if dy == 0 and (pt[1] - 0.5 < ylim[0] or pt[1] + 0.5 > ylim[1]):
                continue
            v = np.array([dx, dy], float)
            v /= np.linalg.norm(v)
            score = 0.0
            if avoid:
                score = min(float(np.arccos(np.clip(v @ a, -1.0, 1.0)))
                            for a in avoid)
            if dx and dy:
                score += 0.05        # slight preference for diagonal placement
            if dx > 0 and dy > 0:
                score += 0.03        # tie-break: up-right
            if score > best_score:
                best_score, best = score, (dx, dy)
    dx, dy = best if best else (1, 1)
    ha = {1: 'left', 0: 'center', -1: 'right'}[dx]
    va = {1: 'bottom', 0: 'center', -1: 'top'}[dy]
    return (mag * dx, mag * dy), ha, va


# size the figure so each square (aspect='equal') panel fills its grid slot:
# slot width == top-row height, otherwise equal-aspect panels get centred in
# wider slots and phantom gaps appear between them no matter the wspace.
FIG_H = 5.0
HSPACE, WSPACE = 0.15, 0.015
top, bottom, left, right = 0.9, 0.04, 0.015, 0.985
row_units = 5.0 + 2.5 * HSPACE                      # 4 + 1 + hspace*mean
panel_h = (top - bottom) * FIG_H * 4.0 / row_units  # top-row height (inches)
FIG_W = panel_h * (n_panels + (n_panels - 1) * WSPACE) / (right - left)

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
fig.set_facecolor("white")
gs = GridSpec(2, n_panels, left=left, right=right, top=top, bottom=bottom,
              wspace=WSPACE, hspace=HSPACE, height_ratios=[4.0, 1.0])

for panel_idx, k in enumerate(show_ks):
    ax = fig.add_subplot(gs[0, panel_idx])
    size = 2 ** k
    selected = fps_order[:size]
    cov = coverages[size - 1]

    for si in selected:
        ax.add_patch(Circle(pts[si], cov, facecolor=CIRCLE_COLOR, alpha=0.10,
                            edgecolor='none', linewidth=0, zorder=1))

    ax.scatter(pts[:, 0], pts[:, 1], s=20, c=POINT_GRAY, edgecolors='none',
               zorder=3, alpha=0.9, rasterized=True)

    for level_k in range(k, 0, -1):
        level_pts = fps_order[2 ** (level_k - 1):2 ** level_k]
        color = LEVEL_COLORS[min(level_k - 1, 3)]
        ax.scatter(pts[level_pts, 0], pts[level_pts, 1], s=110, c=color,
                   edgecolors='none', zorder=5 + level_k)
    ax.scatter(*pts[fps_order[0]], s=110, c=LEVEL_COLORS[0], edgecolors='none',
               zorder=12)

    # the next pick = the point realizing the current coverage (open marker)
    nxt = fps_order[size]
    ax.scatter(*pts[nxt], s=130, facecolors='none', edgecolors='black',
               linewidths=2.0, zorder=14)

    # Delta_1 = separation (closest selected pair),
    # Delta_2 = coverage (next pick <-> its nearest selected point)
    sel = np.array(selected)
    sel_d = all_dists[np.ix_(sel, sel)].copy()
    np.fill_diagonal(sel_d, np.inf)
    i0, j0 = np.unravel_index(np.argmin(sel_d), sel_d.shape)
    near = sel[int(np.argmin(all_dists[nxt, sel]))]

    arrow_dirs = {}
    for (a, b), lab in [((int(sel[i0]), int(sel[j0])), '$\\Delta_1$'),
                        ((int(nxt), int(near)), '$\\Delta_2$')]:
        ax.annotate('', xy=pts[a], xytext=pts[b],
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.6,
                                    shrinkA=6, shrinkB=6), zorder=15)
        dv = pts[b] - pts[a]
        u = dv / np.linalg.norm(dv)
        arrow_dirs.setdefault(a, []).append(u)
        arrow_dirs.setdefault(b, []).append(-u)
        mid = (pts[a] + pts[b]) / 2
        perp = np.array([-u[1], u[0]])
        if np.dot(perp, np.array([cx, cy]) - mid) < 0:
            perp = -perp
        # short arrows (late panels) need the label pushed further out
        lab_off = 0.42 if np.linalg.norm(dv) > 1.5 else 0.68
        ax.text(*(mid + perp * lab_off), lab, fontsize=18, fontweight='bold',
                ha='center', va='center', color='black', zorder=16)

    off, ha, va = label_offset(pts[nxt], avoid=arrow_dirs.get(int(nxt)))
    ax.annotate(f'$p_{{{size + 1}}}$', xy=pts[nxt], xytext=off,
                textcoords='offset points', fontsize=16, fontweight='bold',
                color='black', ha=ha, va=va, zorder=20)

    if k <= 2:
        for idx, pt_idx in enumerate(selected):
            off, ha, va = label_offset(pts[pt_idx],
                                       avoid=arrow_dirs.get(int(pt_idx)))
            ax.annotate(f'$p_{{{idx + 1}}}$', xy=pts[pt_idx], xytext=off,
                        textcoords='offset points', fontsize=16,
                        fontweight='bold', color='black',
                        ha=ha, va=va, zorder=20)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)
        spine.set_visible(True)
    ax.set_title(f'$S_{{{k}}}$,  $|S_{{{k}}}| = {size}$', fontsize=17,
                 color='black', pad=8)

# %%
# Ordering strip: the single FPS ordering with nested brackets = the hierarchy.
ax = fig.add_subplot(gs[1, :])
n_dots = 2 ** max_k
y_dots = 1.0

for i in range(n_dots):
    color = LEVEL_COLORS[min(entry_level(i), 3)]
    ax.scatter(i + 1, y_dots, s=150, c=color, edgecolors='none', zorder=5,
               clip_on=False)
ax.text(0.25, y_dots, 'FPS order', fontsize=16, ha='right', va='center',
        color='black')

# nested brackets under the dots
tick = 0.28
for b, k in enumerate(range(1, max_k + 1)):
    size = 2 ** k
    y = 0.15 - 0.62 * b
    x0, x1 = 0.62, size + 0.38
    ax.plot([x0, x1], [y, y], color='black', lw=1.1, zorder=3)
    ax.plot([x0, x0], [y, y + tick], color='black', lw=1.1, zorder=3)
    ax.plot([x1, x1], [y, y + tick], color='black', lw=1.1, zorder=3)
    ax.text(x1 + 0.28, y, f'$S_{{{k}}}$', fontsize=15, ha='left',
            va='center', color='black')

ax.set_xlim(-5.0, n_dots + 5.0)
ax.set_ylim(0.15 - 0.62 * (max_k - 1) - 0.45, 1.25)
ax.axis('off')

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "build" / "figures" / "fig1.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"wrote {OUT}")
