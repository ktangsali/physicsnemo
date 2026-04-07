#!/usr/bin/env python3
"""Re-plot KDE overlay from saved prediction_results.npz files.

Usage:
  python plot_kde_from_npz.py runs/geotransolver/surface/final_gp_head/prediction_results.npz
  python plot_kde_from_npz.py runs/geotransolver/surface/final_mlp_head_2/prediction_results.npz
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <prediction_results.npz>")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    data = np.load(npz_path)

    keys = list(data.keys())
    datasets: dict[str, dict[str, np.ndarray]] = {}
    for k in keys:
        tag, field = k.split("__", 1)
        datasets.setdefault(tag, {})[field] = data[k]

    has_std = any("pred_std_cd" in v and v["pred_std_cd"].max() > 0 for v in datasets.values())

    n_cols = 2 if has_std else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    ax_dis = axes[0]
    ax_std = axes[1] if has_std else None

    head_label = "GP" if has_std else "MLP"
    cmap = plt.cm.get_cmap("tab10", len(datasets))

    for idx, (tag, res) in enumerate(datasets.items()):
        color = cmap(idx)
        is_id = idx == 0
        lw = 2.5 if is_id else 1.5
        ls = "-" if is_id else "--"
        label = tag.replace("_", " ").title()

        disagree = res["abs_diff"]
        if len(disagree) > 2:
            xs = np.linspace(max(0, disagree.min() * 0.8), disagree.max() * 1.2, 500)
            kde = gaussian_kde(disagree)
            ax_dis.plot(xs, kde(xs), color=color, lw=lw, ls=ls, label=label)
            ax_dis.fill_between(xs, kde(xs), alpha=0.1 if is_id else 0.05, color=color)

        if ax_std is not None:
            std_dev = res["pred_std_cd"]
            if len(std_dev) > 2:
                xs = np.linspace(max(0, std_dev.min() * 0.8), std_dev.max() * 1.2, 500)
                kde = gaussian_kde(std_dev)
                ax_std.plot(xs, kde(xs), color=color, lw=lw, ls=ls, label=label)
                ax_std.fill_between(xs, kde(xs), alpha=0.1 if is_id else 0.05, color=color)

    ax_dis.set_xlabel(f"|Cd_{head_label} − Cd_GeoTransolver|")
    ax_dis.set_ylabel("Density")
    ax_dis.set_title(f"Disagreement ({head_label}): ID vs OOD")
    ax_dis.legend(loc="best", fontsize=8)
    ax_dis.grid(True, alpha=0.3)

    if ax_std is not None:
        ax_std.set_xlabel("GP Predictive Std Dev (Cd)")
        ax_std.set_ylabel("Density")
        ax_std.set_title("GP Std Dev: ID vs OOD")
        ax_std.set_xlim(2e-2, 2.1e-2)
        ax_std.legend(loc="best", fontsize=8)
        ax_std.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = npz_path.parent / "kde_id_vs_ood_zoomed.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
