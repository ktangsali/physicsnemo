#!/usr/bin/env python3
"""
Plot GP drag predictions vs true drag on validation/test set.

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
    AttentionPooling,
    DragGP,
    compute_drag_target_from_batch,
    load_pretrained_model_only,
)
from train_variational_gp import cast_precisions  # noqa: F401 (used in eval)


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

    # Data
    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=None,
    )
    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=None,
    )
    n_train = len(train_dataloader)
    n_val = len(val_dataloader)

    # Models
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)
    load_pretrained_model_only(model, pretrained_ckpt_path)
    model.eval()

    embedding_reduction_model = AttentionPooling(feat_dim=448, embed_dim=32)
    embedding_reduction_model.to(device)

    gp = DragGP(embed_dim=32, n_inducing=64, n_train=n_train)
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

    # Collect predictions and true drag over val set
    indices = []
    true_cd = []
    pred_mean_cd = []
    pred_std_cd = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            features = batch["fx"]
            embeddings = batch["embeddings"]
            if "geometry" in batch:
                geometry = cast_precisions(batch["geometry"], precision)
            else:
                geometry = None
            features = cast_precisions(features, precision)
            embeddings = cast_precisions(embeddings, precision)
            local_positions = embeddings[:, :, :3]

            _, learned_embeddings = model(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
            )
            reduced = embedding_reduction_model(learned_embeddings[0])

            mean_scaled, var_scaled, _, _ = gp.predict(reduced)
            mean_scaled = mean_scaled.cpu().numpy().flatten()
            std_scaled = np.sqrt(var_scaled.cpu().numpy().flatten())

            target_scaled = compute_drag_target_from_batch(
                batch, surface_factors, device
            )
            true_scaled = target_scaled.cpu().numpy().flatten()

            # One value per sample (batch size 1)
            for k in range(len(mean_scaled)):
                indices.append(len(true_cd) + k)
                true_cd.append(float(true_scaled[k] * DRAG_COEFF_SCALE))
                pred_mean_cd.append(float(mean_scaled[k] * DRAG_COEFF_SCALE))
                pred_std_cd.append(float(std_scaled[k] * DRAG_COEFF_SCALE))

    indices = np.array(indices)
    true_cd = np.array(true_cd)
    pred_mean_cd = np.array(pred_mean_cd)
    pred_std_cd = np.array(pred_std_cd)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(indices, true_cd, "o", ms=4, label="True drag (Cd)", color="C0", alpha=0.8)
    ax.plot(
        indices,
        pred_mean_cd,
        "-",
        label="GP mean (Cd)",
        color="C1",
        lw=1.5,
    )
    ax.fill_between(
        indices,
        pred_mean_cd - pred_std_cd,
        pred_mean_cd + pred_std_cd,
        alpha=0.3,
        color="C1",
        label=r"GP $\pm$ 1 std",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Drag coefficient (Cd)")
    ax.set_title("GP drag predictions on validation set")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gp_drag_predictions.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
