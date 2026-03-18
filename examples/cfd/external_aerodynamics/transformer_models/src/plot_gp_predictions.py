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


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DistributedManager.initialize()

    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir
    pretrained_ckpt_path = getattr(
        cfg, "pretrained_checkpoint_path", None
    ) or f"{checkpoint_dir}/{cfg.run_id}/checkpoints"
    gp_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_gp"

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

    # OOD test dataloaders
    test_1_path = getattr(cfg.data.get("test_1", {}), "data_path", None)
    test_2_path = getattr(cfg.data.get("test_2", {}), "data_path", None)
    ood_sets: list[tuple[str, Any, Any]] = []
    if test_1_path:
        dl, dl_full = _make_dataloaders(test_1_path)
        ood_sets.append(("OOD class_N (test_1)", dl, dl_full))
    if test_2_path:
        dl, dl_full = _make_dataloaders(test_2_path)
        ood_sets.append(("OOD class_E (test_2)", dl, dl_full))

    chunk_size = getattr(cfg.data, "resolution", 51200) or 51200
    n_train = len(train_dataloader)
    n_val = len(val_dataloader)

    # Models
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)
    load_pretrained_model_only(model, pretrained_ckpt_path)
    model.eval()

    pooling_type = cfg.get("embedding_pooling", "attention")
    embedding_reduction_model = create_embedding_reduction(
        pooling=pooling_type, feat_dim=448, embed_dim=32,
    )
    embedding_reduction_model.to(device)

    gp = DragGP(embed_dim=32, n_inducing=128, n_train=n_train)
    gp.to(device)

    # Load GP and embedding reduction from checkpoints_gp
    checkpoint_epoch = getattr(cfg, "checkpoint_epoch", None)
    load_checkpoint(
        path=gp_ckpt_path,
        models=[embedding_reduction_model, gp],
        device=device,
        epoch=checkpoint_epoch,
    )
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

    # --- Plotting ---
    n_panels = 1 + len(ood_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    def _plot_panel(ax, res, title):
        true_cd = res["true_cd"]
        pred_mean_cd = res["pred_mean_cd"]
        pred_std_cd = res["pred_std_cd"]
        transolver_cd = res["transolver_cd"]

        cd_min = min(true_cd.min(), pred_mean_cd.min(), transolver_cd.min())
        cd_max = max(true_cd.max(), pred_mean_cd.max(), transolver_cd.max())
        margin = 0.05 * (cd_max - cd_min) if cd_max > cd_min else 0.01
        diag_lo, diag_hi = cd_min - margin, cd_max + margin

        ax.plot(
            [diag_lo, diag_hi], [diag_lo, diag_hi],
            "k--", lw=1.5, alpha=0.7, label="y = x",
        )
        sort_idx = np.argsort(true_cd)
        ax.fill_between(
            true_cd[sort_idx],
            (pred_mean_cd - 2 * pred_std_cd)[sort_idx],
            (pred_mean_cd + 2 * pred_std_cd)[sort_idx],
            alpha=0.3, color="C1", label=r"GP $\pm$ 2 std",
        )
        ax.plot(true_cd, pred_mean_cd, "o", ms=2, color="C1", alpha=0.9, label="GP mean")
        ax.plot(true_cd, transolver_cd, "s", ms=2, color="C2", alpha=0.9,
                label="Geo-Transolver (field→Cd)")
        ax.set_xlabel("True Cd")
        ax.set_ylabel("Predicted Cd")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(diag_lo, diag_hi)
        ax.set_ylim(diag_lo, diag_hi)

    _plot_panel(axes[0], val_results, "Validation (in-distribution)")
    for idx, (name, res) in enumerate(ood_results):
        _plot_panel(axes[idx + 1], res, name)

    fig.tight_layout()

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gp_drag_predictions.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
