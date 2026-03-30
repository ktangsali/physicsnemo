#!/usr/bin/env python3
"""
Plot GP and Geo-Transolver drag predictions vs true drag on validation/test set.

True drag coefficient (Cd) on x-axis, predicted Cd on y-axis. Includes:
- GP mean ± 1 std (shaded band)
- Geo-Transolver: predicted surface fields → unnormalize → integrate to Cd (same
  pipeline as true drag). A diagonal line (y = x) shows perfect prediction.

Usage (from transformer_models/src):
  python plot_gp_predictions.py
  python plot_gp_predictions.py checkpoint_epoch=501
  python plot_gp_predictions.py data.val.data_path=/path/to/test/

Output: saves gp_drag_predictions.png (and optionally shows interactively).
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

# Allowlist for torch.load(weights_only=True) when loading checkpoints (PyTorch 2.6+)
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])

import gpytorch
from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset
from physicsnemo.utils import load_checkpoint

# Import model classes and helpers from training script
from train_variational_gp import (
    DRAG_COEFF_SCALE,
    DragGP,
    compute_drag_target_from_batch,
    create_embedding_reduction,
    load_pretrained_model_only,
)
from train_variational_gp import cast_precisions  # noqa: F401 (used in eval)


def predict_full_mesh_in_chunks(
    batch_full: dict,
    model: torch.nn.Module,
    chunk_size: int,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    """
    Run the geo-transolver on a full-mesh batch in chunks, then stitch and unshuffle.
    Returns predicted fields of shape (1, N_full, 4) so drag can be computed with
    full-mesh normals/areas.
    """
    N = batch_full["embeddings"].shape[1]
    indices = torch.randperm(N, device=batch_full["fx"].device)
    index_blocks = torch.split(indices, chunk_size)

    preds = []
    for index_block in index_blocks:
        local_embeddings = batch_full["embeddings"][:, index_block]
        local_positions = local_embeddings[:, :, :3]
        features = cast_precisions(batch_full["fx"], precision)
        local_embeddings = cast_precisions(local_embeddings, precision)
        if "geometry" in batch_full:
            geometry = cast_precisions(batch_full["geometry"], precision)
        else:
            geometry = None
        outputs, _ = model(
            global_embedding=features,
            local_embedding=local_embeddings,
            geometry=geometry,
            local_positions=local_positions,
        )
        preds.append(outputs)
    # (1, N_full, 4) after concat along dim=1 (shuffled order)
    stitched = torch.cat(preds, dim=1)
    inverse_indices = torch.empty_like(indices)
    inverse_indices[indices] = torch.arange(N, device=indices.device)
    return stitched[:, inverse_indices]


@hydra.main(version_base=None, config_path="conf", config_name="geotransolver_surface_combined")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir
    combined_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_combined"
    use_combined = Path(combined_ckpt_path).exists()

    if not use_combined:
        pretrained_ckpt_path = getattr(
            cfg, "pretrained_checkpoint_path", None
        ) or f"{checkpoint_dir}/{cfg.run_id}/checkpoints"
        gp_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_gp"

    print(
        f"Loading from {'combined' if use_combined else 'separate'} "
        f"checkpoint(s) under {checkpoint_dir}/{cfg.run_id}/"
    )

    # Load normalization
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
    surface_factors = {
        "mean": torch.from_numpy(np.load(norm_file)["mean"]).to(device),
        "std": torch.from_numpy(np.load(norm_file)["std"]).to(device),
    }

    # Helper: build a subsampled + full-resolution dataloader pair for a given phase.
    # For OOD splits we swap val.data_path in a cloned config so
    # create_transolver_dataset (which only knows "train"/"val") works unchanged.
    def _make_dataloaders(data_path_override: str | None = None):
        cfg_data = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(cfg.data, resolve=True)
        )
        if data_path_override is not None:
            cfg_data.val.data_path = data_path_override

        dl = create_transolver_dataset(
            cfg_data, phase="val", surface_factors=surface_factors, volume_factors=None,
        )
        cfg_data_full = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(cfg_data, resolve=True)
        )
        cfg_data_full.resolution = None
        cfg_data_full.return_mesh_features = True
        dl_full = create_transolver_dataset(
            cfg_data_full, phase="val", surface_factors=surface_factors, volume_factors=None,
        )
        return dl, dl_full

    # Data
    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=None,
    )
    val_dataloader, val_dataloader_full = _make_dataloaders()

    # OOD test dataloaders — auto-discover all "test_*" keys in cfg.data
    ood_sets: list[tuple[str, Any, Any]] = []
    for key in sorted(cfg.data.keys()):
        if not key.startswith("test_"):
            continue
        entry = cfg.data[key]
        path = getattr(entry, "data_path", None) if hasattr(entry, "data_path") else None
        if path is None or not Path(path).is_dir():
            print(f"  Skipping {key}: path={path} (not found)")
            continue
        label = key.replace("test_", "OOD ").replace("_", " ").title()
        print(f"  Loading OOD set: {key} -> {path}")
        dl, dl_full = _make_dataloaders(path)
        ood_sets.append((label, dl, dl_full))

    chunk_size = getattr(cfg.data, "resolution", 51200) or 51200
    n_train = len(train_dataloader)
    n_val = len(val_dataloader)

    # Models
    embed_dim = getattr(cfg, "embed_dim", 32)
    feat_dim = getattr(cfg, "embedding_feat_dim", 448)
    n_inducing = getattr(cfg, "n_inducing", 128)
    pooling_type = cfg.get("embedding_pooling", "attention")
    use_spectral_norm = getattr(cfg, "spectral_norm_embedding", False)
    normalize_embeddings = getattr(cfg, "normalize_embeddings", False)
    embedding_target_scale = getattr(cfg, "embedding_target_scale", 1.0)

    ls_range = tuple(getattr(cfg, "gp_lengthscale_range", [0.01, 10.0]))
    ls_prior_cfg = getattr(cfg, "gp_lengthscale_prior", None)
    ls_prior = tuple(ls_prior_cfg) if ls_prior_cfg is not None else None
    os_prior_cfg = getattr(cfg, "gp_outputscale_prior", None)
    os_prior = tuple(os_prior_cfg) if os_prior_cfg is not None else None

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)

    embedding_reduction_model = create_embedding_reduction(
        pooling=pooling_type, feat_dim=feat_dim, embed_dim=embed_dim,
        spectral_norm=use_spectral_norm,
        normalize=normalize_embeddings,
        target_scale=embedding_target_scale,
    )
    embedding_reduction_model.to(device)

    gp = DragGP(
        embed_dim=embed_dim, n_inducing=n_inducing, n_train=n_train,
        lengthscale_range=ls_range,
        lengthscale_prior=ls_prior,
        outputscale_prior=os_prior,
    )
    gp.to(device)

    checkpoint_epoch = getattr(cfg, "checkpoint_epoch", None)
    if use_combined:
        load_checkpoint(
            path=combined_ckpt_path,
            models=[model, embedding_reduction_model, gp],
            device=device,
            epoch=checkpoint_epoch,
        )
    else:
        load_pretrained_model_only(model, pretrained_ckpt_path)
        load_checkpoint(
            path=gp_ckpt_path,
            models=[embedding_reduction_model, gp],
            device=device,
            epoch=checkpoint_epoch,
        )

    model.eval()
    embedding_reduction_model.eval()
    gp.eval()

    precision = getattr(cfg, "precision", "float32")

    # --- Reusable prediction collector ---
    def collect_predictions(dl_sub, dl_full):
        """Run GP + Geo-Transolver inference on a dataloader pair.
        Returns dict with arrays: true_cd, pred_mean_cd, pred_std_cd, transolver_cd.
        """
        true_list, mean_list, std_list, trans_list = [], [], [], []
        full_iter = iter(dl_full)
        with torch.no_grad():
            for batch in dl_sub:
                features = cast_precisions(batch["fx"], precision)
                embeddings = cast_precisions(batch["embeddings"], precision)
                geometry = (
                    cast_precisions(batch["geometry"], precision)
                    if "geometry" in batch else None
                )
                local_positions = embeddings[:, :, :3]

                outputs, learned_embeddings = model(
                    global_embedding=features,
                    local_embedding=embeddings,
                    geometry=geometry,
                    local_positions=local_positions,
                )
                reduced = embedding_reduction_model(learned_embeddings[0])

                mean_scaled, var_scaled, _, _ = gp.predict(reduced)
                mean_np = mean_scaled.cpu().numpy().flatten()
                std_np = np.sqrt(var_scaled.cpu().numpy().flatten())

                target_scaled = compute_drag_target_from_batch(
                    batch, surface_factors, device
                )
                true_np = target_scaled.cpu().numpy().flatten()

                batch_full = next(full_iter)
                outputs_full = predict_full_mesh_in_chunks(
                    batch_full, model, chunk_size, device, precision,
                )
                mod_full = dict(batch_full)
                mod_full["fields_full"] = outputs_full
                trans_val = float(
                    compute_drag_target_from_batch(mod_full, surface_factors, device)
                    .cpu().numpy().flatten()[0] * DRAG_COEFF_SCALE
                )

                for k in range(len(mean_np)):
                    true_list.append(float(true_np[k] * DRAG_COEFF_SCALE))
                    mean_list.append(float(mean_np[k] * DRAG_COEFF_SCALE))
                    std_list.append(float(std_np[k] * DRAG_COEFF_SCALE))
                    trans_list.append(trans_val)

        return {
            "true_cd": np.array(true_list),
            "pred_mean_cd": np.array(mean_list),
            "pred_std_cd": np.array(std_list),
            "transolver_cd": np.array(trans_list),
        }

    # Collect for in-distribution val
    print("Collecting predictions on validation set ...")
    val_results = collect_predictions(val_dataloader, val_dataloader_full)

    # Collect for each OOD set
    ood_results: list[tuple[str, dict]] = []
    for name, dl_sub, dl_full in ood_sets:
        print(f"Collecting predictions on {name} ...")
        ood_results.append((name, collect_predictions(dl_sub, dl_full)))

    # --- Plotting helpers ---
    all_results = [("Validation (in-distribution)", val_results)] + ood_results

    # Pre-compute derived quantities for every dataset
    for _name, res in all_results:
        res["abs_diff"] = np.abs(res["pred_mean_cd"] - res["transolver_cd"])
        res["joint_uq"] = np.maximum(res["abs_diff"], res["pred_std_cd"])

    # --- Global axis ranges (consistent across rows for each column) ---
    def _global_range(key):
        vals = np.concatenate([res[key] for _, res in all_results])
        lo, hi = vals.min(), vals.max()
        margin = 0.05 * (hi - lo) if hi > lo else 0.01
        return lo - margin, hi + margin

    # Col 0: scatter — use global Cd range, clamped to >= 0
    all_cd = np.concatenate([
        np.concatenate([res["true_cd"], res["pred_mean_cd"], res["transolver_cd"]])
        for _, res in all_results
    ])
    cd_margin = 0.05 * (all_cd.max() - all_cd.min()) if all_cd.max() > all_cd.min() else 0.01
    scatter_lo = max(0.0, all_cd.min() - cd_margin)
    scatter_hi = all_cd.max() + cd_margin

    # Cols 1-2: histogram x ranges
    diff_lo, diff_hi = _global_range("abs_diff")
    diff_lo = max(diff_lo, 0.0)
    std_lo, std_hi = _global_range("pred_std_cd")
    std_lo = max(std_lo, 0.0)

    _PERCENTILES = [80, 90, 95]
    _PCT_COLORS = ["C6", "C8", "C9"]

    def _add_percentile_lines(ax, data):
        for pct, color in zip(_PERCENTILES, _PCT_COLORS):
            val = np.percentile(data, pct)
            ax.axvline(val, color=color, ls="-.", lw=1.2,
                        label=f"P{pct} = {val:.4f}")

    def _plot_scatter(ax, res, title):
        true_cd = res["true_cd"]
        pred_mean_cd = res["pred_mean_cd"]
        pred_std_cd = res["pred_std_cd"]
        transolver_cd = res["transolver_cd"]

        ax.plot(
            [scatter_lo, scatter_hi], [scatter_lo, scatter_hi],
            "k--", lw=1.5, alpha=0.7, label="y = x",
        )
        sort_idx = np.argsort(true_cd)
        ax.fill_between(
            true_cd[sort_idx],
            (pred_mean_cd - 2 * pred_std_cd)[sort_idx],
            (pred_mean_cd + 2 * pred_std_cd)[sort_idx],
            alpha=0.3, color="C1", label=r"GP $\pm$ 2 std",
        )
        ax.plot(true_cd, pred_mean_cd, "o", ms=1.5, color="C1", alpha=0.9, label="GP mean")
        ax.plot(true_cd, transolver_cd, "s", ms=1.5, color="C2", alpha=0.9,
                label="Geo-Transolver (field→Cd)")
        ax.set_xlabel("True Cd")
        ax.set_ylabel("Predicted Cd")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(scatter_lo, scatter_hi)
        ax.set_ylim(scatter_lo, scatter_hi)

    def _plot_joint_scatter(ax, res, title):
        """Scatter: true Cd on x, GP mean on y, with joint UQ shading."""
        true_cd = res["true_cd"]
        pred_mean_cd = res["pred_mean_cd"]
        joint_uq = res["joint_uq"]

        ax.plot(
            [scatter_lo, scatter_hi], [scatter_lo, scatter_hi],
            "k--", lw=1.5, alpha=0.7, label="y = x",
        )
        sort_idx = np.argsort(true_cd)
        ax.fill_between(
            true_cd[sort_idx],
            (pred_mean_cd - joint_uq)[sort_idx],
            (pred_mean_cd + joint_uq)[sort_idx],
            alpha=0.3, color="C5", label=r"GP mean $\pm$ joint UQ",
        )
        ax.plot(true_cd, pred_mean_cd, "o", ms=1.5, color="C1", alpha=0.9, label="GP mean")
        ax.set_xlabel("True Cd")
        ax.set_ylabel("Predicted Cd")
        ax.set_title(f"{title}\nJoint UQ = max(|disagree|, GP std)")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(scatter_lo, scatter_hi)
        ax.set_ylim(scatter_lo, scatter_hi)

    def _plot_hist(ax, data, xlabel, title, color, xlims):
        ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.75,
                range=xlims)
        ax.axvline(np.mean(data), color="k", ls="--", lw=1.5,
                    label=f"mean = {np.mean(data):.4f}")
        ax.axvline(np.median(data), color="C0", ls=":", lw=1.5,
                    label=f"median = {np.median(data):.4f}")
        _add_percentile_lines(ax, data)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.set_xlim(xlims)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    # --- Layout: one row per dataset, 4 columns ---
    # Col 0: scatter (GP + Transolver vs true)
    # Col 1: histogram |disagreement|
    # Col 2: histogram GP std
    # Col 3: scatter with joint UQ shading
    n_rows = len(all_results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(28, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 4)

    for row, (name, res) in enumerate(all_results):
        _plot_scatter(axes[row, 0], res, name)
        _plot_hist(
            axes[row, 1], res["abs_diff"],
            xlabel="|Cd_GP − Cd_GeoTransolver|",
            title=f"{name}\n|GP − GeoTransolver| gap",
            color="C3", xlims=(diff_lo, diff_hi),
        )
        _plot_hist(
            axes[row, 2], res["pred_std_cd"],
            xlabel="GP Predictive Std Dev (Cd)",
            title=f"{name}\nGP Std Dev",
            color="C4", xlims=(std_lo, std_hi),
        )
        _plot_joint_scatter(axes[row, 3], res, name)

    fig.tight_layout()

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gp_drag_predictions.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
