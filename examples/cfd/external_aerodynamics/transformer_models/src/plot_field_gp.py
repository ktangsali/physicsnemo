# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-point field-GP uncertainty diagnostics.

Loads a trained GeoTransolver + :class:`FieldVariationalGPHead`, collects per-point GP
mean, std and ground-truth error over the validation set (and any ``test_*``
OOD sets in the config), then produces:

* **Calibration / reliability** — expected vs observed coverage of the GP
  confidence intervals (a well-calibrated GP traces the diagonal).
* **Error vs UQ** — per-point |error| against predictive std (hexbin) with the
  Spearman rank correlation (higher = std is a better error proxy).
* **ID vs OOD std** — KDE of per-point predictive std for in-distribution vs
  OOD inputs (OOD should shift to higher uncertainty).

All quantities are computed in the GP's normalised target space so channels are
directly comparable.  Saves figures and a ``field_gp_uq_results.npz``.

Usage (from transformer_models/src)::

    python plot_field_gp.py
    python plot_field_gp.py ++checkpoint_epoch=201
    python plot_field_gp.py ++max_points_per_set=2000000
"""

import collections
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset
from physicsnemo.utils import load_checkpoint

from train import cast_precisions, get_autocast_context
from field_gp_utils import (
    NUM_SURFACE_TASKS,
    build_field_gp_head,
    compute_field_targets_from_batch,
)

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])

# Channel labels (pressure + wall-shear-stress vector components).
CHANNEL_LABELS = ["pressure", "wss_x", "wss_y", "wss_z"]


@torch.no_grad()
def collect_pointwise(
    dataloader,
    model,
    head,
    precision,
    device,
    max_points: int,
) -> dict[str, np.ndarray]:
    """Gather per-point GP mean, std and |error| over a dataloader.

    Returns arrays of shape ``(P, num_tasks)`` (subsampled to ``max_points``
    total points), all in the GP's normalised target space.
    """
    errs: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    n_have = 0
    for batch in dataloader:
        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        targets = compute_field_targets_from_batch(batch).to(device)
        with get_autocast_context(precision):
            _, point_features = model(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=embeddings[:, :, :3],
                return_point_features=True,
            )
        pred = head.predict(point_features)
        mean = pred.mean.to(targets.dtype)
        std = pred.variance.clamp_min(0).sqrt().to(targets.dtype)

        err = (mean - targets).abs().reshape(-1, head.num_tasks).cpu().numpy()
        sd = std.reshape(-1, head.num_tasks).cpu().numpy()
        errs.append(err)
        stds.append(sd)
        n_have += err.shape[0]
        if n_have >= max_points:
            break

    err_all = np.concatenate(errs, axis=0)
    std_all = np.concatenate(stds, axis=0)
    if err_all.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(
            err_all.shape[0], size=max_points, replace=False
        )
        err_all, std_all = err_all[idx], std_all[idx]
    return {"abs_err": err_all, "std": std_all}


def _coverage_curve(
    abs_err: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Observed coverage vs expected coverage for a Gaussian posterior.

    For each nominal two-sided coverage ``c`` (e.g. 0.5, 0.9), the matching
    z-multiplier is ``z = Phi^{-1}((1+c)/2)``; observed coverage is the fraction
    of points with ``|err| <= z * std``.
    """
    from scipy.stats import norm

    expected = np.linspace(0.05, 0.99, 20)
    z = norm.ppf((1.0 + expected) / 2.0)
    safe_std = np.maximum(std, 1e-12)
    ratio = abs_err / safe_std
    observed = np.array([(ratio <= zz).mean() for zz in z])
    return expected, observed


def main(cfg: DictConfig) -> None:
    """Collect per-point UQ diagnostics and write figures + npz."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()

    precision = getattr(cfg, "precision", "float32")
    max_points = int(getattr(cfg, "max_points_per_set", 1_000_000))

    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    surface_factors = {
        "mean": torch.from_numpy(np.load(norm_file)["mean"]).to(device),
        "std": torch.from_numpy(np.load(norm_file)["std"]).to(device),
    }

    def _make_loader(data_path_override: str | None = None):
        cfg_data = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(cfg.data, resolve=True)
        )
        if data_path_override is not None:
            cfg_data.val.data_path = data_path_override
        return create_transolver_dataset(
            cfg_data, phase="val", surface_factors=surface_factors, volume_factors=None
        )

    val_loader = _make_loader()

    ood_sets: list[tuple[str, Any]] = []
    for key in sorted(cfg.data.keys()):
        if not key.startswith("test_"):
            continue
        entry = cfg.data[key]
        path = (
            getattr(entry, "data_path", None) if hasattr(entry, "data_path") else None
        )
        if path is None or not Path(path).is_dir():
            print(f"  Skipping {key}: path={path} (not found)")
            continue
        label = key.replace("test_", "OOD ").replace("_", " ").title()
        print(f"  Loading OOD set: {key} -> {path}")
        ood_sets.append((label, _make_loader(path)))

    # ---- Build backbone + head ----
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)
    model.eval()

    num_tasks = getattr(cfg, "num_tasks", NUM_SURFACE_TASKS)
    n_inducing = getattr(cfg, "gp_n_inducing", 256)
    mlp_hidden_cfg = getattr(cfg, "gp_mlp_hidden", None)
    mlp_hidden = list(mlp_hidden_cfg) if mlp_hidden_cfg is not None else None
    ls_range = tuple(getattr(cfg, "gp_lengthscale_range", [0.01, 10.0]))
    ls_prior_cfg = getattr(cfg, "gp_lengthscale_prior", None)
    ls_prior = tuple(ls_prior_cfg) if ls_prior_cfg is not None else None
    os_prior_cfg = getattr(cfg, "gp_outputscale_prior", None)
    os_prior = tuple(os_prior_cfg) if os_prior_cfg is not None else None

    # Probe feature dim.
    probe_batch = next(iter(val_loader))
    with torch.no_grad():
        feats = cast_precisions(probe_batch["fx"], precision)
        emb = cast_precisions(probe_batch["embeddings"][:, :512], precision)
        geo = (
            cast_precisions(probe_batch["geometry"], precision)
            if "geometry" in probe_batch
            else None
        )
        with get_autocast_context(precision):
            _, pf = model(
                global_embedding=feats,
                local_embedding=emb,
                geometry=geo,
                local_positions=emb[:, :, :3],
                return_point_features=True,
            )
        feature_dim = int(pf.shape[-1])

    head = build_field_gp_head(
        input_dim=feature_dim,
        n_train_points=1,
        num_tasks=num_tasks,
        n_inducing=n_inducing,
        mlp_hidden=mlp_hidden,
        lengthscale_range=ls_range,
        lengthscale_prior=ls_prior,
        outputscale_prior=os_prior,
    )
    head.to(device)

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or (
        f"{cfg.output_dir}/{cfg.run_id}/checkpoints_field_gp"
    )
    load_checkpoint(
        path=checkpoint_dir,
        models=[model, head],
        device=device,
        epoch=getattr(cfg, "checkpoint_epoch", None),
    )
    model.eval()
    head.eval()
    head.likelihood.eval()

    # ---- Collect ----
    print("Collecting per-point UQ on validation set ...")
    results = [
        (
            "Validation (in-distribution)",
            collect_pointwise(val_loader, model, head, precision, device, max_points),
        )
    ]
    for name, loader in ood_sets:
        print(f"Collecting per-point UQ on {name} ...")
        results.append(
            (
                name,
                collect_pointwise(loader, model, head, precision, device, max_points),
            )
        )

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save raw arrays ----
    npz = {}
    for name, res in results:
        tag = name.replace(" ", "_").lower()
        npz[f"{tag}__abs_err"] = res["abs_err"]
        npz[f"{tag}__std"] = res["std"]
    np.savez_compressed(out_dir / "field_gp_uq_results.npz", **npz)
    print(f"Saved arrays to {out_dir / 'field_gp_uq_results.npz'}")

    # ---- Plot: per-channel calibration ----
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    if num_tasks == 1:
        axes = [axes]
    for c in range(num_tasks):
        ax = axes[c]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="ideal")
        for name, res in results:
            exp, obs = _coverage_curve(res["abs_err"][:, c], res["std"][:, c])
            ax.plot(exp, obs, marker="o", ms=3, label=name)
        ax.set_xlabel("Expected coverage")
        ax.set_ylabel("Observed coverage")
        ax.set_title(f"Calibration — {CHANNEL_LABELS[c]}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "field_gp_calibration.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'field_gp_calibration.png'}")

    # ---- Plot: error vs UQ (validation, per channel) ----
    val_res = results[0][1]
    fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
    if num_tasks == 1:
        axes = [axes]
    for c in range(num_tasks):
        ax = axes[c]
        std_c = val_res["std"][:, c]
        err_c = val_res["abs_err"][:, c]
        hb = ax.hexbin(std_c, err_c, gridsize=40, bins="log", cmap="viridis")
        fig.colorbar(hb, ax=ax, label="log10(count)")
        # Spearman on a capped subsample (rank corr is robust to the cap).
        n = min(len(std_c), 200_000)
        sub = np.random.default_rng(0).choice(len(std_c), size=n, replace=False)
        rho, _ = spearmanr(std_c[sub], err_c[sub])
        ax.set_xlabel("GP predictive std")
        ax.set_ylabel("|error|")
        ax.set_title(f"Error vs UQ — {CHANNEL_LABELS[c]}\nSpearman={rho:.3f}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "field_gp_error_vs_uq.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'field_gp_error_vs_uq.png'}")

    # ---- Plot: ID vs OOD std distributions ----
    if len(results) > 1:
        from scipy.stats import gaussian_kde

        fig, axes = plt.subplots(1, num_tasks, figsize=(5 * num_tasks, 5))
        if num_tasks == 1:
            axes = [axes]
        cmap = plt.cm.get_cmap("tab10", len(results))
        for c in range(num_tasks):
            ax = axes[c]
            for idx, (name, res) in enumerate(results):
                vals = res["std"][:, c]
                vals = vals[np.isfinite(vals)]
                if len(vals) < 3:
                    continue
                # Subsample for KDE speed.
                if len(vals) > 50_000:
                    vals = np.random.default_rng(0).choice(vals, 50_000, replace=False)
                xs = np.linspace(max(0, vals.min() * 0.8), vals.max() * 1.2, 400)
                kde = gaussian_kde(vals)
                is_id = idx == 0
                ax.plot(
                    xs,
                    kde(xs),
                    color=cmap(idx),
                    lw=2.5 if is_id else 1.5,
                    ls="-" if is_id else "--",
                    label=name,
                )
            ax.set_xlabel("GP predictive std")
            ax.set_ylabel("Density")
            ax.set_title(f"Std: ID vs OOD — {CHANNEL_LABELS[c]}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "field_gp_id_vs_ood_std.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'field_gp_id_vs_ood_std.png'}")


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="geotransolver_surface_field_gp",
)
def launch(cfg: DictConfig) -> None:
    """Hydra entry point for field-GP UQ diagnostics."""
    main(cfg)


if __name__ == "__main__":
    launch()
