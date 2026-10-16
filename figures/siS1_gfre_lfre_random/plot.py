import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "plotting"))

# the helpers of the main figure keep model order, labels and colour scale identical
import gfre_heatmap as union_plot  # noqa: E402


def load_matrix(path):
    with open(path) as handle:
        results = json.load(handle)

    err, ann = union_plot.build_matrix(results)
    rename = {key: union_plot.PRETTY_LABELS.get(key, key) for key in err.index}

    return err.rename(index=rename, columns=rename), ann.rename(index=rename, columns=rename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?",
                        default=os.path.join(HERE, os.pardir, os.pardir, "build", "figures", "gfre_lfre_heatmap_random_test.pdf"))
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(15, 9), dpi=200)
    plt.subplots_adjust(wspace=0.025)

    heatmap = None
    for column, kind in enumerate(["gfre", "lfre"]):
        err, ann = load_matrix(os.path.join(args.data, f"random_{kind}_seed0.json"))
        print(f"{kind.upper()}\n{err.to_string(float_format='%.3f')}\n")

        ax = axes[column]
        heatmap = union_plot._draw(ax, err, ann, show_yticks=(column == 0), annot_fontsize=23)
        ax.set_title(kind.upper(), pad=10, fontsize=33)
        ax.set_xlabel("target", labelpad=8, fontsize=31)
        if column == 0:
            ax.set_ylabel("source", fontsize=31)
            ax.set_yticklabels(ax.get_yticklabels(), ha="right", rotation=0, fontsize=24)

    fig.canvas.draw()
    position = axes[1].get_position()
    cbar_ax = fig.add_axes([position.x1 + 0.009, position.y0, 0.022, position.height])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.set_label("GFRE / LFRE", rotation=270, labelpad=30, fontsize=28)
    cbar.ax.tick_params(labelsize=22)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved {args.output}")

if __name__ == "__main__":
    main()
