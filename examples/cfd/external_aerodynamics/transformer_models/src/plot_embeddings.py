#!/usr/bin/env python3
"""
Plot first 10 dimensions of GeoTransolver embeddings under two pooling strategies:
  (1) Mean pooling over the spatial (point) dimension
  (2) Attention pooling (current trained AttentionPooling module)

Usage (from transformer_models/src):
  python plot_embeddings.py
  python plot_embeddings.py data.val.data_path=/path/to/data/

Output: saves embedding_comparison.png
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

from train_variational_gp import (
    create_embedding_reduction,
    load_pretrained_model_only,
    cast_precisions,
)

N_DIMS_TO_PLOT = 10


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
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

    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=None,
    )

    feat_dim = getattr(cfg, "embedding_feat_dim", 448)
    embed_dim = getattr(cfg, "embed_dim", 32)
    pooling_type = cfg.get("embedding_pooling", "attention")

    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    model.to(device)

    use_spectral_norm = getattr(cfg, "spectral_norm_embedding", False)
    embedding_reduction_model = create_embedding_reduction(
        pooling=pooling_type,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        spectral_norm=use_spectral_norm,
    )
    embedding_reduction_model.to(device)

    if use_combined:
        load_checkpoint(
            path=combined_ckpt_path,
            models=[model, embedding_reduction_model],
            device=device,
        )
    else:
        load_pretrained_model_only(model, pretrained_ckpt_path)
        load_checkpoint(
            path=gp_ckpt_path,
            models=[embedding_reduction_model],
            device=device,
        )

    model.eval()
    embedding_reduction_model.eval()

    precision = getattr(cfg, "precision", "float32")

    mean_embeds_list = []
    attn_embeds_list = []

    with torch.no_grad():
        for batch in val_dataloader:
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
            # learned_embeddings[0]: (1, N_points, 448)
            point_feats = learned_embeddings[0]

            # (1) Mean pooling over spatial dimension -> (1, 448)
            mean_pooled = point_feats.mean(dim=1)
            # (2) Attention pooling -> (1, 32)
            attn_pooled = embedding_reduction_model(point_feats)

            mean_embeds_list.append(mean_pooled.cpu().numpy())
            attn_embeds_list.append(attn_pooled.cpu().numpy())

    mean_embeds = np.concatenate(mean_embeds_list, axis=0)
    attn_embeds = np.concatenate(attn_embeds_list, axis=0)
    n_samples = mean_embeds.shape[0]
    indices = np.arange(n_samples)

    # Plot first 10 dimensions for each pooling strategy
    fig, (ax_mean, ax_attn) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for d in range(min(N_DIMS_TO_PLOT, mean_embeds.shape[1])):
        ax_mean.plot(
            indices,
            mean_embeds[:, d],
            "-o",
            ms=2,
            label=f"Dim {d}",
            alpha=0.8,
        )
    ax_mean.set_ylabel("Embedding value")
    ax_mean.set_title("GeoTransolver embeddings: mean pooling (first 10 dims)")
    ax_mean.legend(loc="upper right", ncol=2, fontsize=8)
    ax_mean.grid(True, alpha=0.3)

    for d in range(min(N_DIMS_TO_PLOT, attn_embeds.shape[1])):
        ax_attn.plot(
            indices,
            attn_embeds[:, d],
            "-o",
            ms=2,
            label=f"Dim {d}",
            alpha=0.8,
        )
    ax_attn.set_xlabel("Sample index")
    ax_attn.set_ylabel("Embedding value")
    ax_attn.set_title(
        f"GeoTransolver embeddings: {pooling_type} pooling (first 10 dims)"
    )
    ax_attn.legend(loc="upper right", ncol=2, fontsize=8)
    ax_attn.grid(True, alpha=0.3)

    fig.tight_layout()
    out_dir = Path(cfg.output_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedding_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
