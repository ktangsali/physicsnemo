#!/usr/bin/env python3
"""
Plot GP training metrics from active learning sbatch .out logs.

Three subplots:
  1. Avg Train MSE & Avg Val GP MSE vs epoch
  2. Avg Train GP Loss vs epoch (linear scale -- can be negative)
  3. GP hyperparameters vs epoch (lengthscale min/max/mean, outputscale, noise)

Accumulates results from all files matching the pattern.

Usage:
  python plot_gp_results.py [pattern]
  python plot_gp_results.py "coreai_modulus_cae-active-learning:drivaer-geotransolver-gp-stage-1-try-3_*.out"
  python plot_gp_results.py   # uses default pattern (see below)

Output: saves plot as gp_train_val_loss.png (accumulated across all matching logs).
"""

import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

JOB_ID_RE = re.compile(r"_(\d+)\.out$", re.IGNORECASE)

DEFAULT_PATTERN = "slurm_logs/drivaer-gp-v3_*.out"


def job_id_from_path(log_path: Path) -> Optional[str]:
    m = JOB_ID_RE.search(log_path.name)
    return m.group(1) if m else None


def find_log_files(pattern: str) -> List[Path]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    paths = [Path(f) for f in files]
    paths.sort(key=lambda p: p.stat().st_mtime)
    return paths


def parse_log(log_path: Path) -> Dict[str, list]:
    """Extract epoch-level averages and GP hyperparameters from a single log file."""
    avg_train_re = re.compile(
        r"Epoch \[(\d+)/\d+\] Avg Train (?:GP|Head) Loss: ([\d.eE+-]+)"
        r"(?:\s+Avg Train MSE: ([\d.eE+-]+))?"
    )
    avg_val_re = re.compile(r"Epoch \[(\d+)/\d+\] Avg Val (?:GP|Head) MSE: ([\d.eE+-]+)")
    hyper_re = re.compile(
        r"GP hypers .+ lengthscale: min=([\d.eE+-]+) max=([\d.eE+-]+) mean=([\d.eE+-]+)"
        r" \| outputscale=([\d.eE+-]+) \| noise=([\d.eE+-]+)"
    )

    data: Dict[str, list] = {
        "epochs_gp_loss": [], "gp_losses": [],
        "epochs_train_mse": [], "train_mses": [],
        "epochs_val_mse": [], "val_mses": [],
        "epochs_hyper": [],
        "ls_min": [], "ls_max": [], "ls_mean": [],
        "outputscale": [], "noise": [],
    }

    last_train_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            m = avg_train_re.search(line)
            if m:
                epoch = int(m.group(1))
                data["epochs_gp_loss"].append(epoch)
                data["gp_losses"].append(float(m.group(2)))
                if m.group(3) is not None:
                    data["epochs_train_mse"].append(epoch)
                    data["train_mses"].append(float(m.group(3)))
                last_train_epoch = epoch
                continue
            m = avg_val_re.search(line)
            if m:
                data["epochs_val_mse"].append(int(m.group(1)))
                data["val_mses"].append(float(m.group(2)))
                continue
            m = hyper_re.search(line)
            if m:
                data["epochs_hyper"].append(
                    last_train_epoch if last_train_epoch is not None else 0
                )
                data["ls_min"].append(float(m.group(1)))
                data["ls_max"].append(float(m.group(2)))
                data["ls_mean"].append(float(m.group(3)))
                data["outputscale"].append(float(m.group(4)))
                data["noise"].append(float(m.group(5)))

    return data


def accumulate_logs(log_paths: List[Path]) -> Dict[str, list]:
    """Parse all logs and merge into one timeline using original epoch numbers.

    Log paths are assumed sorted oldest-first (by mtime).  For overlapping
    epochs the values from the *latest* log win.
    """
    GROUPS = [
        ("epochs_gp_loss", ["gp_losses"]),
        ("epochs_train_mse", ["train_mses"]),
        ("epochs_val_mse", ["val_mses"]),
        ("epochs_hyper", ["ls_min", "ls_max", "ls_mean", "outputscale", "noise"]),
    ]

    group_dicts: Dict[str, Dict[int, tuple]] = {ek: {} for ek, _ in GROUPS}

    for log_path in log_paths:
        data = parse_log(log_path)
        for epoch_key, val_keys in GROUPS:
            epochs = data[epoch_key]
            for i, ep in enumerate(epochs):
                group_dicts[epoch_key][ep] = tuple(data[vk][i] for vk in val_keys)

    merged: Dict[str, list] = {}
    for epoch_key, val_keys in GROUPS:
        sorted_items = sorted(group_dicts[epoch_key].items())
        merged[epoch_key] = [ep for ep, _ in sorted_items]
        for j, vk in enumerate(val_keys):
            merged[vk] = [vals[j] for _, vals in sorted_items]

    return merged


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pattern = sys.argv[1]
    else:
        pattern = DEFAULT_PATTERN

    print(f"Searching for files matching: {pattern}\n")

    log_paths = find_log_files(pattern)
    print(f"Found {len(log_paths)} file(s) matching pattern: {pattern}")
    for p in log_paths:
        print(f"  - {p.name}")
    print()

    d = accumulate_logs(log_paths)

    has_mse = d["epochs_train_mse"] or d["epochs_val_mse"]
    has_gp = d["epochs_gp_loss"]
    has_hyper = d["epochs_hyper"]
    if not has_mse and not has_gp and not has_hyper:
        print("No head metrics found in any file.", file=sys.stderr)
        sys.exit(1)

    n_plots = 5 if has_hyper else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 2:
        axes = list(axes)

    ax_idx = 0

    # --- Subplot 1: Avg Train MSE & Avg Val Head MSE ---
    ax = axes[ax_idx]; ax_idx += 1
    if d["epochs_train_mse"]:
        ax.plot(
            d["epochs_train_mse"], d["train_mses"], "b-o",
            label="Avg Train MSE", markersize=3,
            markevery=max(1, len(d["epochs_train_mse"]) // 50),
        )
    if d["epochs_val_mse"]:
        ax.plot(
            d["epochs_val_mse"], d["val_mses"], "r-s",
            label="Avg Val Head MSE", markersize=3,
            markevery=max(1, len(d["epochs_val_mse"]) // 50),
        )
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"Train & Val MSE vs Epoch  ({len(log_paths)} log(s))")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    # --- Subplot 2: Avg Train Head Loss (linear scale) ---
    ax = axes[ax_idx]; ax_idx += 1
    if d["epochs_gp_loss"]:
        ax.plot(
            d["epochs_gp_loss"], d["gp_losses"], "-",
            color="tab:green", linewidth=1.2,
            label="Avg Train Head Loss",
        )
    ax.set_ylabel("Head Loss")
    ax.set_title("Avg Train Head Loss vs Epoch")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if has_hyper:
        # --- Subplot 3: Lengthscale ---
        ax = axes[ax_idx]; ax_idx += 1
        ep = d["epochs_hyper"]
        ax.fill_between(
            ep, d["ls_min"], d["ls_max"],
            alpha=0.2, color="tab:blue", label="Min\u2013Max range",
        )
        ax.plot(ep, d["ls_mean"], "-", color="tab:blue", linewidth=1.2, label="Mean")
        ax.set_ylabel("Lengthscale")
        ax.set_title("GP Lengthscale vs Epoch")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # --- Subplot 4: Outputscale ---
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(ep, d["outputscale"], "-", color="tab:orange", linewidth=1.2, label="Outputscale")
        ax.set_ylabel("Outputscale")
        ax.set_title("GP Outputscale vs Epoch")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # --- Subplot 5: Noise ---
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(ep, d["noise"], "-", color="tab:red", linewidth=1.2, label="Noise")
        ax.set_ylabel("Noise")
        ax.set_title("GP Noise vs Epoch")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()

    # output_file = "stage_2_mlp_head.png"
    # output_file = "stage_2_gp_head.png"
    output_file = "stage_2_gp_v3_head.png"
    
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_file}")


if __name__ == "__main__":
    main()
