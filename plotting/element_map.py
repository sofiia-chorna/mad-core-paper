from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from ase.data import chemical_symbols

MAX_Z = 118  # Og


# fmt: off
PT_LAYOUT = {
    "H": (0, 0), "He": (0, 17),
    "Li": (1, 0), "Be": (1, 1), "B": (1, 12), "C": (1, 13), "N": (1, 14), "O": (1, 15), "F": (1, 16), "Ne": (1, 17),
    "Na": (2, 0), "Mg": (2, 1), "Al": (2, 12), "Si": (2, 13), "P": (2, 14), "S": (2, 15), "Cl": (2, 16), "Ar": (2, 17),
    "K": (3, 0), "Ca": (3, 1), "Sc": (3, 2), "Ti": (3, 3), "V": (3, 4), "Cr": (3, 5), "Mn": (3, 6), "Fe": (3, 7), "Co": (3, 8), "Ni": (3, 9), "Cu": (3, 10), "Zn": (3, 11), "Ga": (3, 12), "Ge": (3, 13), "As": (3, 14), "Se": (3, 15), "Br": (3, 16), "Kr": (3, 17),
    "Rb": (4, 0), "Sr": (4, 1), "Y": (4, 2), "Zr": (4, 3), "Nb": (4, 4), "Mo": (4, 5), "Tc": (4, 6), "Ru": (4, 7), "Rh": (4, 8), "Pd": (4, 9), "Ag": (4, 10), "Cd": (4, 11), "In": (4, 12), "Sn": (4, 13), "Sb": (4, 14), "Te": (4, 15), "I": (4, 16), "Xe": (4, 17),
    "Cs": (5, 0), "Ba": (5, 1), "Hf": (5, 3), "Ta": (5, 4), "W": (5, 5), "Re": (5, 6), "Os": (5, 7), "Ir": (5, 8), "Pt": (5, 9), "Au": (5, 10), "Hg": (5, 11), "Tl": (5, 12), "Pb": (5, 13), "Bi": (5, 14), "Po": (5, 15), "At": (5, 16), "Rn": (5, 17),
    "Fr": (6, 0), "Ra": (6, 1), "Rf": (6, 3), "Db": (6, 4), "Sg": (6, 5), "Bh": (6, 6), "Hs": (6, 7), "Mt": (6, 8), "Ds": (6, 9), "Rg": (6, 10), "Cn": (6, 11), "Nh": (6, 12), "Fl": (6, 13), "Mc": (6, 14), "Lv": (6, 15), "Ts": (6, 16), "Og": (6, 17),
    "La": (8, 3), "Ce": (8, 4), "Pr": (8, 5), "Nd": (8, 6), "Pm": (8, 7), "Sm": (8, 8), "Eu": (8, 9), "Gd": (8, 10), "Tb": (8, 11), "Dy": (8, 12), "Ho": (8, 13), "Er": (8, 14), "Tm": (8, 15), "Yb": (8, 16), "Lu": (8, 17),
    "Ac": (9, 3), "Th": (9, 4), "Pa": (9, 5), "U": (9, 6), "Np": (9, 7), "Pu": (9, 8), "Am": (9, 9), "Cm": (9, 10), "Bk": (9, 11), "Cf": (9, 12), "Es": (9, 13), "Fm": (9, 14), "Md": (9, 15), "No": (9, 16), "Lr": (9, 17),
}
# fmt: on
N_ROWS, N_COLS = 10, 18


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fps", required=True, help="FPS output parquet")
    p.add_argument("--manifest", required=True, help="manifest used to run FPS")
    p.add_argument("--ks", type=int, nargs="+", required=True,
                   help="K cutoffs to plot (e.g. 65536 262144 1000000)")
    p.add_argument("--mode", choices=["counts", "percent"], default="counts")
    p.add_argument("--vmax-percent", type=float, nargs="+", default=[2.0],
                   help="max %% for percent mode colormap. One value for all panels, "
                        "or one per --ks panel (default: 2)")
    p.add_argument("--vmax-counts", type=int, default=None,
                   help="explicit colorbar max for counts mode (otherwise auto-fit "
                        "to the largest count across panels)")
    p.add_argument("--vmin-counts", type=int, default=None,
                   help="explicit colorbar min for counts mode (otherwise auto-fit "
                        "to the smallest non-zero count across panels)")
    p.add_argument("--per-panel-vmax", action="store_true",
                   help="counts mode: give each panel its OWN colorbar min and max (auto-fit to "
                        "that panel's own count range) instead of one shared scale across panels")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--enriched", default=None,
                   help="path to enriched parquet (default: <fps stem>.with_elements.parquet next to --fps)")
    return p.parse_args()


# ---------------------------- enrichment ----------------------------

def _bitmask_from_atomic_numbers(arr: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Vectorized (n, MAX_Z+1) bool mask from a flat atomic_numbers array + row offsets."""
    n = len(offsets) - 1
    mask = np.zeros((n, MAX_Z + 1), dtype=bool)
    for i in range(n):
        z = arr[offsets[i]:offsets[i + 1]]
        mask[i, z] = True
    return mask


def enrich(fps_path: str, manifest_path: str, enriched_path: Path,
           source_bitmask_path: Path):
    """Build:
      enriched_path        — (fps_order, element_mask) per pick, parquet
      source_bitmask_path  — (MAX_Z+1,) int64 counts of structures per element in
                              the FULL source (needed for ratio/percent modes), npz
    """
    print(f"reading picks from {fps_path} ...")
    picks = pq.read_table(fps_path, columns=["fps_order", "configuration_id"])
    fo = picks.column("fps_order").to_numpy()
    cids = picks.column("configuration_id").to_pylist()
    cid_to_order = {cid: int(fo[i]) for i, cid in enumerate(cids)}
    K = len(cid_to_order)
    print(f"  {K:,} picks")

    pick_mask = np.zeros((K, MAX_Z + 1), dtype=bool)
    found = np.zeros(K, dtype=bool)
    full_counts = np.zeros(MAX_Z + 1, dtype=np.int64)

    shards = [ln.strip() for ln in open(manifest_path) if ln.strip()]
    print(f"\nscanning {len(shards)} source shards for atomic_numbers ...")
    for j, sh in enumerate(shards):
        t = pq.read_table(sh, columns=["configuration_id", "atomic_numbers"])
        cids_sh = t.column("configuration_id").to_pylist()
        an = t.column("atomic_numbers").combine_chunks()
        values = np.asarray(an.values.to_numpy(zero_copy_only=False), dtype=np.int32)
        offsets = np.asarray(an.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        n = len(cids_sh)
        mask = _bitmask_from_atomic_numbers(values, offsets)  # (n, MAX_Z+1)
        full_counts += mask.sum(axis=0)
        # collect hits
        for i, cid in enumerate(cids_sh):
            o = cid_to_order.get(cid)
            if o is not None:
                pick_mask[o] = mask[i]
                found[o] = True
        if (j + 1) % 50 == 0 or j + 1 == len(shards):
            print(f"  shard {j+1:>4}/{len(shards)}  cum_hits={int(found.sum()):>10,}/{K:,}", flush=True)

    if not found.all():
        miss = int((~found).sum())
        print(f"WARN: {miss:,} picks not found by configuration_id (something is off)")

    # write outputs
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    out_tbl = pa.table({
        "fps_order": pa.array(np.arange(K, dtype=np.int64)),
        "element_mask": pa.array(pick_mask.tolist(), type=pa.list_(pa.bool_())),
    })
    pq.write_table(out_tbl, enriched_path, compression="zstd")
    np.savez(source_bitmask_path, full_counts=full_counts, K=K, n_total=int(sum(
        # n_total isn't strictly recoverable without extra scan; estimate from full_counts
        [full_counts.max()]
    )))
    # the "n_total" stored above is a hack; recompute exactly by reading shard sizes
    # cheap (parquet metadata, no row read):
    n_total = 0
    for sh in shards:
        n_total += pq.ParquetFile(sh).metadata.num_rows
    np.savez(source_bitmask_path, full_counts=full_counts, K=K, n_total=n_total)
    print(f"\nwrote {enriched_path}")
    print(f"wrote {source_bitmask_path}  (n_total={n_total:,})")


# ---------------------------- plotting ----------------------------

def _text_color(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"


def _fmt_count(v) -> str:
    """Plain round-number label: 200, 1k, 20k (matches the per-cell labels)."""
    v = int(round(v))
    return f"{v:,}" if v < 1000 else f"{v // 1000}k"


def _nice_log_ticks(vmin: float, vmax: float) -> list[int]:
    """Round 1/2/3/5/7 x 10^n values within [vmin, vmax], for a log colorbar."""
    if vmax <= vmin:
        return [int(round(vmin))]
    lo = int(math.floor(math.log10(vmin)))
    hi = int(math.ceil(math.log10(vmax)))
    span = hi - lo
    mantissas = (1,) if span >= 4 else (1, 3) if span >= 2 else (1, 2, 3, 5, 7)  # thin ticks on wide ranges
    return [m * 10 ** n for n in range(lo, hi + 1) for m in mantissas
            if vmin <= m * 10 ** n <= vmax]


def _add_colorbar(ax, cmap, norm, label, ticks=None, tick_labels=None):
    x_min, x_max = ax.get_xlim(); y_min, y_max = ax.get_ylim()
    ax_w = x_max - x_min; ax_h = y_max - y_min
    cb_data_w = 8.0
    cb_x0 = (7.0 - cb_data_w / 2 - x_min) / ax_w
    cb_y0 = (8.0 - y_min) / ax_h
    cb_w = cb_data_w / ax_w; cb_h = 0.35 / ax_h
    cax = ax.inset_axes([cb_x0, cb_y0, cb_w, cb_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    if ticks is not None:
        cbar.set_ticks(ticks)
        if tick_labels is not None:
            cbar.set_ticklabels(tick_labels)
        cbar.ax.minorticks_off()  # drop the unlabeled 4/6/8x10^n minor log ticks
    cbar.set_label(label, fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6)
    return cbar


def draw_pt(ax, el_counts: dict, title: str, mode: str, vmax_percent: float,
            vmax_counts: int | None = None, vmin_counts: int | None = None):
    if mode == "counts":
        # Use shared vmin/vmax across panels if provided (so colormaps are
        # comparable); otherwise auto-fit to this panel's data.
        if vmax_counts is None:
            vmax_counts = max(el_counts.values()) if el_counts else 1
        if vmin_counts is None:
            nz = [v for v in el_counts.values() if v > 0]
            vmin_counts = min(nz) if nz else 1
        vmin_counts = max(int(vmin_counts), 1)
        vmax_counts = max(int(vmax_counts), 1)
        cmap = plt.colormaps["viridis"]
        cbar_ticks = cbar_tick_labels = None
        if vmax_counts > vmin_counts:
            norm = mcolors.LogNorm(vmin=vmin_counts, vmax=vmax_counts)
            cbar_ticks = _nice_log_ticks(vmin_counts, vmax_counts)
            cbar_tick_labels = [_fmt_count(t) for t in cbar_ticks]
        else:
            norm = mcolors.Normalize(vmin_counts - 0.5, vmin_counts + 0.5)
        cbar_label = "# structures"
    else:  # percent
        cmap = plt.colormaps["viridis"]
        norm = mcolors.Normalize(vmin=0, vmax=vmax_percent)
        cbar_ticks = cbar_tick_labels = None
        cbar_label = "% of element structures picked"

    for el, (row, col) in PT_LAYOUT.items():
        x, y = col, N_ROWS - 1 - row
        v = el_counts.get(el, None)
        if v is None or (mode == "counts" and v == 0):
            rgba = (0.91, 0.91, 0.91, 1.0)
            label = None
        else:
            if mode == "percent":
                rgba = cmap(norm(v))
                label = f"{v:.1f}%"
            else:
                rgba = cmap(norm(v))
                label = _fmt_count(v)
        rect = plt.Rectangle((x, y), 0.95, 0.95, facecolor=rgba, edgecolor="white", linewidth=0.8)
        ax.add_patch(rect)
        fc = _text_color(rgba) if label is not None else "#aaaaaa"
        ax.text(x + 0.475, y + 0.63, el, ha="center", va="center",
                fontsize=11, fontweight="bold", color=fc)
        if label is not None:
            ax.text(x + 0.475, y + 0.22, label, ha="center", va="center",
                    fontsize=7, color=fc, alpha=0.9)

    ax.set_xlim(-0.3, N_COLS + 0.3); ax.set_ylim(-0.5, N_ROWS + 0.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=13, pad=2)
    _add_colorbar(ax, cmap, norm, cbar_label, ticks=cbar_ticks,
                  tick_labels=cbar_tick_labels)


# ---------------------------- main ----------------------------

def main():
    args = parse_args()
    fps = Path(args.fps)
    enriched = Path(args.enriched) if args.enriched else fps.with_name(fps.stem + ".with_elements.parquet")
    src_counts = enriched.with_suffix(".source_counts.npz")

    if not enriched.exists() or not src_counts.exists():
        print(f"=== ENRICH ===  ({enriched.name} missing)")
        enrich(args.fps, args.manifest, enriched, src_counts)
    else:
        print(f"using cached {enriched.name}")

    # load enriched
    t = pq.read_table(enriched)
    fo = t.column("fps_order").to_numpy()
    em = np.asarray(t.column("element_mask").to_pylist(), dtype=bool)  # (K, MAX_Z+1)
    order = np.argsort(fo)
    em = em[order]
    K_total = em.shape[0]
    print(f"loaded enriched: K={K_total:,}")

    src = np.load(src_counts)
    full_counts = src["full_counts"]
    n_total = int(src["n_total"])
    print(f"source: n_total={n_total:,}  elements present={int((full_counts>0).sum())}")

    ks = [k for k in args.ks if k <= K_total]
    if not ks:
        raise SystemExit(f"no K in --ks <= {K_total}")
    print(f"plotting K = {ks}")

    # broadcast --vmax-percent (1 → all panels, or one per panel)
    vps = list(args.vmax_percent)
    if len(vps) == 1:
        vps = vps * len(ks)
    elif len(vps) != len(ks):
        raise SystemExit(
            f"--vmax-percent must be 1 or {len(ks)} values; got {len(vps)}")

    # pre-compute per-K element counts (for shared counts vmax)
    per_k_counts = [em[:K].sum(axis=0) for K in ks]
    if args.per_panel_vmax:
        shared_counts_vmax = None          # draw_pt auto-fits each panel to its own max
        shared_counts_vmin = None          # ... and its own (non-zero) min
    else:
        if args.vmax_counts is not None:
            shared_counts_vmax = int(args.vmax_counts)
        else:
            shared_counts_vmax = int(max((c.max() for c in per_k_counts), default=1))
        if args.vmin_counts is not None:
            shared_counts_vmin = int(args.vmin_counts)
        else:
            nz_mins = [c[c > 0].min() for c in per_k_counts if (c > 0).any()]
            shared_counts_vmin = int(min(nz_mins)) if nz_mins else 1

    def panel_label(K):
        if K == 1_000_000:
            return "N=1M"
        log2k = math.log2(K)
        if abs(log2k - round(log2k)) < 1e-9:
            return f"k={int(round(log2k))}"
        if K % 1_000_000 == 0:
            return f"K={K // 1_000_000}M"
        return f"K={K:,}"

    fig, axes = plt.subplots(1, len(ks), figsize=(7 * len(ks), 5.5))
    if len(ks) == 1:
        axes = [axes]

    for ax, K, vmax_k, fps_counts in zip(axes, ks, vps, per_k_counts):
        if args.mode == "counts":
            el = {chemical_symbols[z]: int(fps_counts[z])
                  for z in range(MAX_Z + 1) if fps_counts[z] > 0}
            title = panel_label(K)
            draw_pt(ax, el, title, args.mode, vmax_k,
                    vmax_counts=shared_counts_vmax, vmin_counts=shared_counts_vmin)
        else:
            el = {}
            for z in range(MAX_Z + 1):
                if full_counts[z] > 0:
                    el[chemical_symbols[z]] = float(100.0 * fps_counts[z] / full_counts[z])
            if args.per_panel_vmax:                       # auto-fit this panel's own max %
                vmax_k = max(el.values()) if el else 1.0
            title = panel_label(K)
            draw_pt(ax, el, title, args.mode, vmax_k)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nsaved => {out}")


if __name__ == "__main__":
    main()
