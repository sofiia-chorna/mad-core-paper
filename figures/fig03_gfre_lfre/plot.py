import argparse
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "plotting"))
from gfre_heatmap import PRETTY_LABELS, _draw, build_matrix, load_results  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="see the README")
    p.add_argument("output", nargs="?",
                   default=os.path.join(HERE, os.pardir, os.pardir, "build", "figures", "gfre_lfre_heatmap_union.pdf"))
    p.add_argument("--gfre", default=os.path.join(HERE, "data", "union_gfre_seed0.json"),
                   help="GFRE results JSON")
    p.add_argument("--lfre", default=os.path.join(HERE, "data", "union_lfre_seed0.json"),
                   help="LFRE results JSON")
    p.add_argument("--no-borders", action="store_true",
                   help="Hide heatmap cell borders and outer heatmap frames")
    p.add_argument("--wspace", type=float, default=0.025,
                   help="Horizontal space between the GFRE and LFRE panels")
    p.add_argument("--cbar-pad", type=float, default=0.009,
                   help="Gap between the LFRE panel and the colorbar in figure coordinates")
    args = p.parse_args()

    err_g, ann_g = build_matrix(load_results(args.gfre))
    print("GFRE matrix:")
    print(err_g.to_string(float_format='%.3f'))
    err_l, ann_l = build_matrix(load_results(args.lfre))
    print("\nLFRE matrix:")
    print(err_l.to_string(float_format='%.3f'))

    rename = {k: PRETTY_LABELS.get(k, k) for k in err_g.index}
    err_g = err_g.rename(index=rename, columns=rename)
    ann_g = ann_g.rename(index=rename, columns=rename)
    err_l = err_l.rename(index=rename, columns=rename)
    ann_l = ann_l.rename(index=rename, columns=rename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9), dpi=200)
    plt.subplots_adjust(wspace=args.wspace)

    _draw(ax1, err_g, ann_g, cbar_ax=None, show_yticks=True, annot_fontsize=23,
          show_borders=not args.no_borders)
    ax1.set_xlabel("target", labelpad=8, fontsize=31)
    ax1.set_ylabel("source", fontsize=31)
    ax1.set_yticklabels(ax1.get_yticklabels(), ha='right', rotation=0, fontsize=24)
    ax1.set_title("GFRE", pad=10, fontsize=33)
    hm2 = _draw(ax2, err_l, ann_l, cbar_ax=None, show_yticks=False, annot_fontsize=23,
                show_borders=not args.no_borders)
    ax2.set_xlabel("target", labelpad=8, fontsize=31)
    ax2.set_title("LFRE", pad=10, fontsize=33)
    fig.canvas.draw()
    pos = ax2.get_position()
    cbar_ax = fig.add_axes([pos.x1 + args.cbar_pad, pos.y0, 0.022, pos.height])
    cbar = fig.colorbar(hm2, cax=cbar_ax)
    cbar.set_label("GFRE / LFRE", rotation=270, labelpad=30, fontsize=28)
    cbar.ax.tick_params(labelsize=22)

    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\nsaved => {out}")


if __name__ == "__main__":
    sys.exit(main())
