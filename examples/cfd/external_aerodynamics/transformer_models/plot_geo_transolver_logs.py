#!/usr/bin/env python3
"""
Plot Geo-Transolver training metrics from active learning sbatch .out logs.

Supports two log formats:
  1. Simple format:  Epoch [X/Y] Train Loss: ... Val Loss: ...  (single line)
  2. Combined training format with separate train/val summary lines:
       Epoch [X/Y] Avg Train GP Loss: ... Avg Train MSE: ... Avg Field MSE: ... Avg Cons Loss: ... Avg Total Loss: ...
       Epoch [X/Y] Avg Val GP MSE: ... Avg Val Field MSE: ... Avg Val Consistency Gap: ...

Accumulates results from all files matching the pattern.

Usage:
  python plot_geo_transolver_logs.py [pattern]
  python plot_geo_transolver_logs.py "coreai_modulus_cae-active-learning:drivaer-geotransolver-gp-combined-try-1_*.out"
  python plot_geo_transolver_logs.py   # uses default pattern (see below)

Output: saves plot as geo_transolver_train_val_loss.png (accumulated across all matching logs).
"""

import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

JOB_ID_RE = re.compile(r"_(\d+)\.out$", re.IGNORECASE)


DEFAULT_PATTERN = "slurm_logs/drivaer-gp-head_*.out"

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


# --- Format 1: simple single-line train+val ---
EPOCH_SUMMARY_RE = re.compile(
    r"Epoch \[(\d+)/\d+\]\s+"
    r"Train Loss:\s*([\d.eE+-]+)\s+\[duration:\s*[\d.]+s\]\s+"
    r"Val Loss:\s*([\d.eE+-]+)"
)

# --- Format 2: combined training with separate summary lines ---
COMBINED_TRAIN_RE = re.compile(
    r"Epoch \[(\d+)/\d+\]\s+"
    r"Avg Train (?:GP|Head) Loss:\s*([\d.eE+-]+)\s+"
    r"Avg Train MSE:\s*([\d.eE+-]+)\s+"
    r"Avg Field MSE:\s*([\d.eE+-]+)\s+"
    r"Avg Cons Loss:\s*([\d.eE+-]+)\s+"
    r"Avg Total Loss:\s*([\d.eE+-]+)"
)

COMBINED_VAL_RE = re.compile(
    r"Epoch \[(\d+)/\d+\]\s+"
    r"Avg Val (?:GP|Head) MSE:\s*([\d.eE+-]+)\s+"
    r"Avg Val Field MSE:\s*([\d.eE+-]+)\s+"
    r"Avg Val Consistency Gap:\s*([\d.eE+-]+)"
)

SCHEDULER_NAME_RE = re.compile(r"^\s*name:\s*(\S+)", re.MULTILINE)
STEP_SIZE_RE = re.compile(r"^\s*step_size:\s*(\d+)", re.MULTILINE)
GAMMA_RE = re.compile(r"^\s*gamma:\s*([\d.eE+-]+)", re.MULTILINE)
BASE_LR_RE = re.compile(r"^\s*lr:\s*([\d.eE+-]+)", re.MULTILINE)


def parse_scheduler_config(log_path: Path) -> Optional[Tuple[float, int, float]]:
    """Extract (base_lr, step_size, gamma) from the config dumped at the top of a log.

    Only reads the first ~500 lines to avoid scanning the entire file.
    Returns None if not found.
    """
    header_lines = []
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i > 500:
                break
            header_lines.append(line)
    header = "".join(header_lines)

    sched_match = SCHEDULER_NAME_RE.search(header)
    if not sched_match or sched_match.group(1) != "StepLR":
        return None

    step_m = STEP_SIZE_RE.search(header)
    gamma_m = GAMMA_RE.search(header)
    lr_m = BASE_LR_RE.search(header)
    if not (step_m and gamma_m and lr_m):
        return None

    return float(lr_m.group(1)), int(step_m.group(1)), float(gamma_m.group(1))


def compute_lr_step(epoch: int, base_lr: float, step_size: int, gamma: float) -> float:
    """Replicate PyTorch StepLR: lr = base_lr * gamma^(epoch // step_size)."""
    return base_lr * (gamma ** (epoch // step_size))


def parse_log(log_path: Path) -> Dict[str, list]:
    """Extract epoch-level metrics from a single log file.

    Auto-detects whether the log uses the simple format (train+val on one line)
    or the combined training format (separate train/val summary lines with
    multiple metrics).
    """
    simple_data: Dict[str, list] = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
    }

    train_map: Dict[int, dict] = {}
    val_map: Dict[int, dict] = {}

    with open(log_path, "r") as f:
        for line in f:
            m = EPOCH_SUMMARY_RE.search(line)
            if m:
                simple_data["epochs"].append(int(m.group(1)))
                simple_data["train_losses"].append(float(m.group(2)))
                simple_data["val_losses"].append(float(m.group(3)))
                continue

            m = COMBINED_TRAIN_RE.search(line)
            if m:
                ep = int(m.group(1))
                train_map[ep] = {
                    "gp_loss": float(m.group(2)),
                    "train_mse": float(m.group(3)),
                    "field_mse": float(m.group(4)),
                    "cons_loss": float(m.group(5)),
                    "total_loss": float(m.group(6)),
                }
                continue

            m = COMBINED_VAL_RE.search(line)
            if m:
                ep = int(m.group(1))
                val_map[ep] = {
                    "val_gp_mse": float(m.group(2)),
                    "val_field_mse": float(m.group(3)),
                    "val_consistency_gap": float(m.group(4)),
                }

    if simple_data["epochs"]:
        return {"format": "simple", **simple_data}

    epochs = sorted(set(train_map.keys()) | set(val_map.keys()))
    combined: Dict[str, list] = {
        "format": "combined",
        "epochs": epochs,
        "train_gp_loss": [],
        "train_mse": [],
        "train_field_mse": [],
        "train_cons_loss": [],
        "train_total_loss": [],
        "val_gp_mse": [],
        "val_field_mse": [],
        "val_consistency_gap": [],
    }
    for ep in epochs:
        t = train_map.get(ep, {})
        v = val_map.get(ep, {})
        combined["train_gp_loss"].append(t.get("gp_loss"))
        combined["train_mse"].append(t.get("train_mse"))
        combined["train_field_mse"].append(t.get("field_mse"))
        combined["train_cons_loss"].append(t.get("cons_loss"))
        combined["train_total_loss"].append(t.get("total_loss"))
        combined["val_gp_mse"].append(v.get("val_gp_mse"))
        combined["val_field_mse"].append(v.get("val_field_mse"))
        combined["val_consistency_gap"].append(v.get("val_consistency_gap"))

    return combined


def accumulate_logs(log_paths: List[Path]) -> Dict[str, list]:
    """Parse all logs and merge into one timeline using original epoch numbers.

    Log paths are assumed sorted oldest-first (by mtime).  For overlapping
    epochs the values from the *latest* log win.
    """
    sched_cfg: Optional[Tuple[float, int, float]] = None
    detected_format = None

    simple_epoch_map: Dict[int, tuple] = {}
    combined_train_map: Dict[int, dict] = {}
    combined_val_map: Dict[int, dict] = {}

    for log_path in log_paths:
        data = parse_log(log_path)
        fmt = data.get("format", "simple")
        if detected_format is None:
            detected_format = fmt
        elif fmt != detected_format and data["epochs"]:
            detected_format = fmt

        if fmt == "simple":
            for i, ep in enumerate(data["epochs"]):
                simple_epoch_map[ep] = (data["train_losses"][i], data["val_losses"][i])
        else:
            for i, ep in enumerate(data["epochs"]):
                if data["train_total_loss"][i] is not None:
                    combined_train_map[ep] = {
                        "gp_loss": data["train_gp_loss"][i],
                        "train_mse": data["train_mse"][i],
                        "field_mse": data["train_field_mse"][i],
                        "cons_loss": data["train_cons_loss"][i],
                        "total_loss": data["train_total_loss"][i],
                    }
                if data["val_field_mse"][i] is not None:
                    combined_val_map[ep] = {
                        "val_gp_mse": data["val_gp_mse"][i],
                        "val_field_mse": data["val_field_mse"][i],
                        "val_consistency_gap": data["val_consistency_gap"][i],
                    }

        cfg = parse_scheduler_config(log_path)
        if cfg is not None:
            sched_cfg = cfg

    if detected_format == "combined" and combined_train_map:
        epochs = sorted(set(combined_train_map.keys()) | set(combined_val_map.keys()))
        merged: Dict[str, list] = {
            "format": "combined",
            "epochs": epochs,
            "train_gp_loss": [combined_train_map.get(ep, {}).get("gp_loss") for ep in epochs],
            "train_mse": [combined_train_map.get(ep, {}).get("train_mse") for ep in epochs],
            "train_field_mse": [combined_train_map.get(ep, {}).get("field_mse") for ep in epochs],
            "train_cons_loss": [combined_train_map.get(ep, {}).get("cons_loss") for ep in epochs],
            "train_total_loss": [combined_train_map.get(ep, {}).get("total_loss") for ep in epochs],
            "val_gp_mse": [combined_val_map.get(ep, {}).get("val_gp_mse") for ep in epochs],
            "val_field_mse": [combined_val_map.get(ep, {}).get("val_field_mse") for ep in epochs],
            "val_consistency_gap": [combined_val_map.get(ep, {}).get("val_consistency_gap") for ep in epochs],
            "learning_rates": [],
        }
    else:
        sorted_items = sorted(simple_epoch_map.items())
        epochs = [ep for ep, _ in sorted_items]
        merged = {
            "format": "simple",
            "epochs": epochs,
            "train_losses": [vals[0] for _, vals in sorted_items],
            "val_losses": [vals[1] for _, vals in sorted_items],
            "learning_rates": [],
        }

    if sched_cfg is not None:
        base_lr, step_size, gamma = sched_cfg
        merged["learning_rates"] = [
            compute_lr_step(ep, base_lr, step_size, gamma) for ep in merged["epochs"]
        ]

    return merged


def _filter_none(epochs, values):
    """Return (epochs, values) with None entries removed."""
    pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
    if not pairs:
        return [], []
    return zip(*pairs)


def plot_combined(d: Dict[str, list], n_logs: int):
    """Plot the combined training format with GP, field, and consistency metrics."""
    has_lr = bool(d.get("learning_rates"))
    n_plots = 4 if has_lr else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)

    epochs = d["epochs"]
    markevery = max(1, len(epochs) // 50)
    mk = dict(markersize=3, markevery=markevery)

    # --- Subplot 1: Train losses ---
    ax = axes[0]
    for key, label, fmt in [
        ("train_total_loss", "Total Loss", "k-o"),
        ("train_field_mse", "Field MSE", "b-s"),
        ("train_cons_loss", "Consistency Loss", "m-^"),
        ("train_mse", "Train MSE", "c-d"),
    ]:
        ep, vals = _filter_none(epochs, d[key])
        if vals:
            ax.plot(ep, vals, fmt, label=label, **mk)
    ax.set_ylabel("Train Losses")
    ax.set_yscale("log")
    ax.set_title(f"Geo-Transolver Combined Training – Train Metrics  ({n_logs} log(s))")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Subplot 2: Train GP Loss (can be negative, use linear scale) ---
    ax = axes[1]
    ep, vals = _filter_none(epochs, d["train_gp_loss"])
    if vals:
        ax.plot(ep, vals, "g-o", label="GP Loss (marginal log-likelihood)", **mk)
    ax.set_ylabel("Train GP Loss")
    ax.set_title("GP Marginal Log-Likelihood (more negative = better fit)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Subplot 3: Val metrics ---
    ax = axes[2]
    for key, label, fmt in [
        ("val_field_mse", "Val Field MSE", "r-s"),
        ("val_gp_mse", "Val GP MSE", "orange"),
        ("val_consistency_gap", "Val Consistency Gap", "purple"),
    ]:
        ep, vals = _filter_none(epochs, d[key])
        if vals:
            if fmt.startswith("#") or " " not in fmt and "-" not in fmt:
                ax.plot(ep, vals, "-o", color=fmt, label=label, **mk)
            else:
                ax.plot(ep, vals, fmt, label=label, **mk)
    ax.set_ylabel("Val Metrics")
    ax.set_yscale("log")
    ax.set_title("Validation Metrics")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # --- Subplot 4 (optional): Learning Rate ---
    if has_lr:
        ax = axes[3]
        ax.plot(
            epochs, d["learning_rates"], "-",
            color="tab:green", linewidth=1.5, label="Learning Rate",
        )
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.set_title("Learning Rate vs Epoch (StepLR)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, which="both")

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    return fig


def plot_simple(d: Dict[str, list], n_logs: int):
    """Plot the simple format with single train/val loss."""
    has_lr = bool(d.get("learning_rates"))
    n_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    markevery = max(1, len(d["epochs"]) // 50)

    ax1.plot(
        d["epochs"], d["train_losses"], "b-o",
        label="Train Loss", markersize=3, markevery=markevery,
    )
    ax1.set_ylabel("Train Loss")
    ax1.set_yscale("log")
    ax1.set_title(f"Geo-Transolver Train & Val Loss  ({n_logs} log(s))")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.plot(
        d["epochs"], d["val_losses"], "r-s",
        label="Val Loss", markersize=3, markevery=markevery,
    )
    ax2.set_ylabel("Val Loss")
    ax2.set_yscale("log")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, which="both")

    if has_lr:
        ax3.plot(
            d["epochs"], d["learning_rates"], "-",
            color="tab:green", linewidth=1.5, label="Learning Rate",
        )
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")
        ax3.set_title("Learning Rate vs Epoch (StepLR)")
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3, which="both")
    else:
        ax2.set_xlabel("Epoch")
        print("Warning: could not find StepLR scheduler config; "
              "skipping LR subplot.")

    fig.tight_layout()
    return fig


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        pattern = sys.argv[1]
    else:
        pattern = DEFAULT_PATTERN

    print(f"Searching for files matching: {pattern}\n")

    log_paths = find_log_files(pattern)
    print(f"Found {len(log_paths)} file(s) matching pattern: {pattern}")
    for p in log_paths:
        jid = job_id_from_path(p)
        print(f"  - {p.name}  (job {jid})" if jid else f"  - {p.name}")
    print()

    d = accumulate_logs(log_paths)

    if not d["epochs"]:
        print("No epoch-level metrics found in any file.", file=sys.stderr)
        sys.exit(1)

    fmt = d.get("format", "simple")
    print(f"Detected log format: {fmt}")
    print(f"Epoch range: {d['epochs'][0]} – {d['epochs'][-1]}  "
          f"({len(d['epochs'])} epochs)")

    if fmt == "combined":
        fig = plot_combined(d, len(log_paths))
    else:
        fig = plot_simple(d, len(log_paths))

    output_file = "geotransolver_train_val_loss_gp_head.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_file}")


if __name__ == "__main__":
    main()
