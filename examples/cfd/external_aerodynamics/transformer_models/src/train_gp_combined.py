"""Combined training: GeoTransolver (field prediction) + Variational GP (drag prediction).

Trains both models jointly with three loss terms:
  1. MSE loss on per-point surface fields (pressure + wall shear stress)
  2. Variational ELBO loss on GP drag prediction from learned embeddings
  3. Consistency loss between GP-predicted drag and field-integrated drag

The GP and consistency losses activate after a configurable warmup period with a
linear ramp (default: epoch 50-60).  Inducing points are re-initialised at the
warmup start from the now-meaningful GeoTransolver embeddings.

Key config overrides (command-line):
    +lambda_gp=0.01            GP ELBO loss weight (after ramp reaches 1.0)
    +lambda_consistency=1.0    Consistency loss weight (0 disables)
    +gp_warmup_start=50        Epoch at which GP/consistency ramp begins
    +gp_warmup_end=60          Epoch at which ramp reaches full weight
    +consistency_every_n=1     Compute consistency loss every N steps
    +consistency_detach_transolver=true   Detach GeoTransolver in consistency loss (false = backprop through it)
    +n_inducing=128            Number of GP inducing points
    +embed_dim=32              Dimension of GP embedding
"""

import os
import time
import collections
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import hydra
import omegaconf
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tabulate import tabulate
import torchinfo
import numpy as np

from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.transolver_datapipe import create_transolver_dataset

from train import (
    CombinedOptimizer,
    get_autocast_context,
    cast_precisions,
    pad_input_for_fp8,
    unpad_output_for_fp8,
    loss_fn,
    update_model_params_for_fp8,
)
from train_variational_gp import (
    DRAG_COEFF_SCALE,
    compute_drag_target_from_batch,
    create_embedding_reduction,
    DragGP,
)
from metrics import metrics_fn
from utils import tensorwise

from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
else:
    te, Format, DelayedScaling = None, None, None

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gp_ramp_weight(epoch: int, warmup_start: int, warmup_end: int) -> float:
    """0 before *warmup_start*, linear 0→1 over [start, end), 1 after."""
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    return (epoch - warmup_start) / (warmup_end - warmup_start)


def sync_non_ddp_gradients(modules: list[nn.Module], world_size: int) -> None:
    """All-reduce gradients for modules that are NOT wrapped in DDP."""
    if world_size <= 1:
        return
    for module in modules:
        for p in module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


@torch.no_grad()
def reinitialize_inducing_points(
    model: nn.Module,
    embedding_reduction: nn.Module,
    gp: DragGP,
    dataloader,
    n_inducing: int,
    n_train: int,
    train_indices: list[int],
    precision: str,
    device: torch.device,
    logger,
) -> None:
    """Re-collect inducing-point embeddings from the current (trained) model.

    Temporarily uses the full (undistributed) dataset so that every rank can
    collect at least *n_inducing* samples, then restores the distributed
    indices before returning.
    """
    dataloader.dataset.set_indices(list(range(n_train)))

    model.eval()
    embedding_reduction.eval()
    init_embeddings: list[torch.Tensor] = []
    for batch in dataloader:
        if len(init_embeddings) >= n_inducing:
            break
        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        local_positions = embeddings[:, :, :3]
        _, _, emb_states = model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=local_positions,
        )
        reduced = embedding_reduction(emb_states.flatten(1, 2))
        init_embeddings.append(reduced.cpu())

    dataloader.dataset.set_indices(train_indices)

    init_embeddings_t = torch.cat(init_embeddings, dim=0)[:n_inducing].to(device)
    gp.gp_layer.variational_strategy.inducing_points.data.copy_(init_embeddings_t)

    # Reset variational distribution to match new inducing locations
    vd = gp.gp_layer.variational_strategy._variational_distribution
    vd.variational_mean.data.zero_()
    vd.chol_variational_covar.data.copy_(
        torch.eye(n_inducing, device=device) * 0.01
    )
    logger.info(
        f"Re-initialised {n_inducing} inducing points from current embeddings "
        f"(norm range [{init_embeddings_t.norm(dim=1).min():.4f}, "
        f"{init_embeddings_t.norm(dim=1).max():.4f}])"
    )


def compute_transolver_drag_full_mesh(
    batch_full: dict,
    model: nn.Module,
    chunk_size: int,
    surface_factors: dict,
    device: torch.device,
    precision: str,
    detach: bool = True,
) -> torch.Tensor:
    """Run GeoTransolver on full mesh in chunks; integrate predicted fields to Cd.

    Returns a (1,) tensor in GP-scaled drag space (Cd / DRAG_COEFF_SCALE).
    When *detach=True* (default) the forward pass runs under ``torch.no_grad()``
    and intermediates are offloaded to CPU for memory savings.  When *detach=False*
    gradients flow through the full-mesh pass back into the GeoTransolver (higher
    memory, but allows the consistency loss to update the GeoTransolver).
    """
    ctx = torch.no_grad() if detach else nullcontext()
    with ctx:
        fx_full = cast_precisions(batch_full["fx"].to(device), precision)
        geo_full = (
            cast_precisions(batch_full["geometry"].to(device), precision)
            if "geometry" in batch_full
            else None
        )

        N = batch_full["embeddings"].shape[1]
        indices = torch.randperm(N, device=device)
        index_blocks = torch.split(indices, chunk_size)

        preds: list[torch.Tensor] = []
        for idx_block in index_blocks:
            local_emb = cast_precisions(
                batch_full["embeddings"][:, idx_block].to(device), precision
            )
            local_pos = local_emb[:, :, :3]
            outputs, *_ = model(
                global_embedding=fx_full,
                local_embedding=local_emb,
                geometry=geo_full,
                local_positions=local_pos,
            )
            preds.append(outputs.cpu() if detach else outputs)

        stitched = torch.cat(preds, dim=1)
        if detach:
            inverse = torch.empty_like(indices, device="cpu")
            inverse[indices.cpu()] = torch.arange(N)
            outputs_full = stitched[:, inverse].to(device)
        else:
            inverse = torch.empty_like(indices)
            inverse[indices] = torch.arange(N, device=device)
            outputs_full = stitched[:, inverse]

        mod = dict(batch_full)
        mod["fields_full"] = outputs_full
    return compute_drag_target_from_batch(mod, surface_factors, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(
        PythonLogger(name="combined_training"), dist_manager
    )

    # ---- Configurable hyper-parameters with sane defaults ----
    gp_warmup_start = getattr(cfg, "gp_warmup_start", 50)
    gp_warmup_end = getattr(cfg, "gp_warmup_end", 60)
    lambda_gp = getattr(cfg, "lambda_gp", 0.01)
    lambda_consistency = getattr(cfg, "lambda_consistency", 1.0)
    consistency_every_n = getattr(cfg, "consistency_every_n_steps", 1)
    consistency_detach = getattr(cfg, "consistency_detach_transolver", True)
    use_spectral_norm = getattr(cfg, "spectral_norm_embedding", False)
    n_inducing = getattr(cfg, "n_inducing", 128)
    embed_dim = getattr(cfg, "embed_dim", 32)
    feat_dim = getattr(cfg, "embedding_feat_dim", 256)
    accumulation_steps = getattr(cfg.training, "gradient_accumulation_steps", 1)
    use_consistency = lambda_consistency > 0

    # ---- Directories and writers ----
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir
    ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_combined"

    if dist_manager.rank == 0:
        os.makedirs(ckpt_path, exist_ok=True)
        writer = SummaryWriter(
            log_dir=f"{cfg.output_dir}/{cfg.run_id}/combined_train"
        )
        val_writer = SummaryWriter(
            log_dir=f"{cfg.output_dir}/{cfg.run_id}/combined_val"
        )
    else:
        writer = val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Output directory: {cfg.output_dir}/{cfg.run_id}")
    logger.info(f"Checkpoint directory: {ckpt_path}")
    logger.info(
        f"GP warmup: epochs {gp_warmup_start}-{gp_warmup_end}, "
        f"lambda_gp={lambda_gp}, lambda_consistency={lambda_consistency}, "
        f"consistency_detach_transolver={consistency_detach}, "
        f"spectral_norm_embedding={use_spectral_norm}"
    )

    precision = cfg.precision
    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    # ---- GeoTransolver (DDP-wrapped, trainable) ----
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.to(dist_manager.device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
    )
    num_geo_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GeoTransolver parameters: {num_geo_params:,}")

    # ---- Embedding reduction + GP (non-DDP; synced manually) ----
    pooling_type = cfg.get("embedding_pooling", "attention")
    embedding_reduction = create_embedding_reduction(
        pooling=pooling_type, feat_dim=feat_dim, embed_dim=embed_dim,
        spectral_norm=use_spectral_norm,
    )
    embedding_reduction.to(dist_manager.device)

    # ---- Normalization factors ----
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    surface_factors = volume_factors = None
    if cfg.data.mode in ("surface", "combined"):
        nd = np.load(str(Path(norm_dir) / "surface_fields_normalization.npz"))
        surface_factors = {
            "mean": torch.from_numpy(nd["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(nd["std"]).to(dist_manager.device),
        }
    if cfg.data.mode in ("volume", "combined"):
        nd = np.load(str(Path(norm_dir) / "volume_fields_normalization.npz"))
        volume_factors = {
            "mean": torch.from_numpy(nd["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(nd["std"]).to(dist_manager.device),
        }

    # ---- Dataloaders ----
    train_dl = create_transolver_dataset(
        cfg.data, phase="train",
        surface_factors=surface_factors, volume_factors=volume_factors,
    )
    val_dl = create_transolver_dataset(
        cfg.data, phase="val",
        surface_factors=surface_factors, volume_factors=volume_factors,
    )

    if use_consistency:
        cfg_data_full = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(cfg.data, resolve=True)
        )
        cfg_data_full.resolution = None
        cfg_data_full.return_mesh_features = True
        train_dl_full = create_transolver_dataset(
            cfg_data_full, phase="train",
            surface_factors=surface_factors, volume_factors=volume_factors,
        )
        val_dl_full = create_transolver_dataset(
            cfg_data_full, phase="val",
            surface_factors=surface_factors, volume_factors=volume_factors,
        )
    else:
        train_dl_full = val_dl_full = None

    # ---- Distributed samplers ----
    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dl, num_replicas=num_replicas, rank=data_rank,
        shuffle=True, drop_last=True,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dl, num_replicas=num_replicas, rank=data_rank,
        shuffle=False, drop_last=True,
    )

    # ---- GP (needs n_train) ----
    n_train = len(train_dl)
    gp = DragGP(
        embed_dim=embed_dim, n_inducing=n_inducing, n_train=n_train,
    )
    gp.to(dist_manager.device)
    num_gp_params = (
        sum(p.numel() for p in embedding_reduction.parameters())
        + sum(p.numel() for p in gp.parameters())
    )
    logger.info(f"Embedding reduction + GP parameters: {num_gp_params:,}")

    # ---- Optimizer ----
    geo_muon_params = [p for p in model.parameters() if p.ndim == 2]
    geo_other_params = [p for p in model.parameters() if p.ndim != 2]

    geo_adamw = hydra.utils.instantiate(
        cfg.training.optimizer, params=geo_other_params,
    )
    geo_muon = torch.optim.Muon(
        geo_muon_params,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    gp_opt = torch.optim.AdamW(
        [
            {"params": embedding_reduction.parameters(), "lr": 1e-3},
            {"params": gp.gp_layer.variational_parameters(), "lr": 1e-2},
            {"params": gp.gp_layer.hyperparameters(), "lr": 1e-2},
            {"params": gp.likelihood.parameters(), "lr": 1e-2},
        ],
        weight_decay=1e-4,
    )
    optimizer = CombinedOptimizer([geo_muon, geo_adamw, gp_opt])

    # ---- Scheduler ----
    scheduler_params = dict(cfg.training.scheduler.params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    scaler = GradScaler() if precision == "float16" else None
    if precision == "float8" and not TE_AVAILABLE:
        raise ImportError(
            "TransformerEngine is required for float8 precision."
        )

    # ---- Checkpoint ----
    ckpt_args = {
        "path": ckpt_path,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": [model, embedding_reduction, gp],
    }
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    inducing_reinit_done = loaded_epoch > gp_warmup_start

    chunk_size = getattr(cfg.data, "resolution", 51200) or 51200

    # Field-metric modes (surface / volume / combined)
    data_mode = cfg.data.mode
    if data_mode == "combined":
        modes = ["surface", "volume"]
    elif data_mode == "surface":
        modes = ["surface"]
    else:
        modes = ["volume"]

    # ---- Training loop ----
    logger.info("Starting combined training ...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_indices = list(train_sampler)
        val_indices = list(val_sampler)
        train_dl.dataset.set_indices(train_indices)
        val_dl.dataset.set_indices(val_indices)
        if use_consistency:
            train_dl_full.dataset.set_indices(train_indices)
            val_dl_full.dataset.set_indices(val_indices)

        epoch_len = len(train_indices)
        w_gp = gp_ramp_weight(epoch, gp_warmup_start, gp_warmup_end)

        # Re-initialise inducing points once at warmup start
        if epoch == gp_warmup_start and not inducing_reinit_done:
            reinitialize_inducing_points(
                model, embedding_reduction, gp, train_dl,
                n_inducing, n_train, train_indices,
                precision, dist_manager.device, logger,
            )
            inducing_reinit_done = True

        # Accumulators
        epoch_mse = 0.0
        epoch_gp_loss = 0.0
        epoch_consistency = 0.0
        epoch_total = 0.0
        epoch_gp_train_mse = 0.0

        model.train()
        embedding_reduction.train()
        gp.train()
        gp.likelihood.train()

        full_iter = iter(train_dl_full) if use_consistency else None
        start_time = time.time()

        for i, batch in enumerate(train_dl):
            # Keep full-res iter in sync even if we skip the consistency loss
            batch_full = next(full_iter) if full_iter is not None else None

            features = cast_precisions(batch["fx"], precision)
            embeddings = cast_precisions(batch["embeddings"], precision)
            targets = batch["fields"]
            geometry = (
                cast_precisions(batch["geometry"], precision)
                if "geometry" in batch
                else None
            )

            # ---- GeoTransolver forward (with grad) ----
            with get_autocast_context(precision):
                if precision == "float8" and TE_AVAILABLE:
                    features, geometry = pad_input_for_fp8(
                        features, embeddings, geometry,
                    )

                if geometry is not None:
                    local_positions = embeddings[:, :, :3]
                    outputs, _, embedding_states = model(
                        global_embedding=features,
                        local_embedding=embeddings,
                        geometry=geometry,
                        local_positions=local_positions,
                    )
                    outputs = unpad_output_for_fp8(outputs, output_pad_size)
                    mse_loss = torch.mean(loss_fn(outputs, targets))
                else:
                    outputs = model(fx=features, embedding=embeddings)
                    outputs = unpad_output_for_fp8(outputs, output_pad_size)
                    mse_loss = F.mse_loss(outputs, targets)
                    embedding_states = None

            if embedding_states is None:
                raise RuntimeError(
                    "Model did not return embedding_states.  "
                    "Combined training requires a GeoTransolver (geometry) model."
                )

            # ---- GP ELBO loss ----
            gp_loss_val = 0.0
            gp_train_mse_val = 0.0

            if w_gp > 0:
                reduced = embedding_reduction(embedding_states.flatten(1, 2))
                drag_target = compute_drag_target_from_batch(
                    batch, surface_factors, dist_manager.device,
                ).to(reduced.dtype)

                gp_dist = gp(reduced)
                gp_mean = gp_dist.mean
                gp_loss = -gp.mll(gp_dist, drag_target)

                with torch.no_grad():
                    gp_train_mse_val = F.mse_loss(gp_mean, drag_target).item()
                gp_loss_val = gp_loss.detach().item()
            else:
                gp_loss = torch.tensor(0.0, device=dist_manager.device)
                gp_mean = None

            # ---- Consistency loss ----
            c_loss_val = 0.0
            if (
                use_consistency
                and w_gp > 0
                and gp_mean is not None
                and batch_full is not None
                and (i % consistency_every_n == 0)
            ):
                transolver_cd = compute_transolver_drag_full_mesh(
                    batch_full, model, chunk_size,
                    surface_factors, dist_manager.device, precision,
                    detach=consistency_detach,
                ).to(gp_mean.dtype)
                if consistency_detach:
                    transolver_cd = transolver_cd.detach()
                c_loss = F.mse_loss(gp_mean, transolver_cd)
                c_loss_val = c_loss.detach().item()
            else:
                c_loss = torch.tensor(0.0, device=dist_manager.device)

            # ---- Weighted total ----
            total_loss = (
                mse_loss
                + w_gp * lambda_gp * gp_loss
                + w_gp * lambda_consistency * c_loss
            )

            # ---- Gradient accumulation with DDP no_sync ----
            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            is_step_boundary = (
                (i + 1) % accumulation_steps == 0 or (i + 1) == epoch_len
            )
            sync_ctx = nullcontext() if is_step_boundary else model.no_sync()
            scaled_loss = total_loss / accumulation_steps
            with sync_ctx:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if is_step_boundary:
                sync_non_ddp_gradients(
                    [embedding_reduction, gp], dist_manager.world_size,
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            # ---- Per-batch logging ----
            mse_val = mse_loss.detach().item()
            total_val = total_loss.detach().item()
            epoch_mse += mse_val
            epoch_gp_loss += gp_loss_val
            epoch_consistency += c_loss_val
            epoch_total += total_val
            epoch_gp_train_mse += gp_train_mse_val

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            logger.info(
                f"Epoch {epoch} [{i}/{epoch_len}] "
                f"Loss: {mse_val:.6f}  GP Loss: {gp_loss_val:.6f}  "
                f"Cons Loss: {c_loss_val:.6f}  Total Loss: {total_val:.6f}  "
                f"Train MSE: {gp_train_mse_val:.6f}  w_gp: {w_gp:.3f}  "
                f"Duration: {duration:.2f}s"
            )

            if dist_manager.rank == 0 and writer is not None:
                gs = i + epoch_len * epoch
                writer.add_scalar("batch/mse_loss", mse_val, gs)
                writer.add_scalar("batch/gp_loss", gp_loss_val, gs)
                writer.add_scalar("batch/consistency_loss", c_loss_val, gs)
                writer.add_scalar("batch/total_loss", total_val, gs)
                writer.add_scalar("batch/gp_train_mse", gp_train_mse_val, gs)
                writer.add_scalar(
                    "batch/gp_weight", w_gp * lambda_gp, gs,
                )
                writer.add_scalar(
                    "batch/consistency_weight", w_gp * lambda_consistency, gs,
                )
                writer.add_scalar(
                    "batch/learning_rate",
                    optimizer.param_groups[0]["lr"], gs,
                )

        # ---- Epoch-level training summary ----
        avg_mse = epoch_mse / max(epoch_len, 1)
        avg_gp = epoch_gp_loss / max(epoch_len, 1)
        avg_cons = epoch_consistency / max(epoch_len, 1)
        avg_total = epoch_total / max(epoch_len, 1)
        avg_gp_mse = epoch_gp_train_mse / max(epoch_len, 1)

        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] "
            f"Avg Train GP Loss: {avg_gp:.6f}  Avg Train MSE: {avg_gp_mse:.6f}  "
            f"Avg Field MSE: {avg_mse:.6f}  Avg Cons Loss: {avg_cons:.6f}  "
            f"Avg Total Loss: {avg_total:.6f}"
        )

        if dist_manager.rank == 0 and writer is not None:
            writer.add_scalar("epoch/mse_loss", avg_mse, epoch)
            writer.add_scalar("epoch/gp_loss", avg_gp, epoch)
            writer.add_scalar("epoch/consistency_loss", avg_cons, epoch)
            writer.add_scalar("epoch/total_loss", avg_total, epoch)
            writer.add_scalar("epoch/gp_train_mse", avg_gp_mse, epoch)
            writer.add_scalar("epoch/gp_weight", w_gp * lambda_gp, epoch)
            writer.add_scalar(
                "epoch/consistency_weight", w_gp * lambda_consistency, epoch,
            )

        if w_gp > 0:
            ls = gp.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu()
            os_ = gp.gp_layer.covar_module.outputscale.detach().cpu().item()
            noise = gp.likelihood.noise.detach().cpu().item()
            logger.info(
                f"  GP hypers — lengthscale: min={ls.min():.4f} "
                f"max={ls.max():.4f} mean={ls.mean():.4f} | "
                f"outputscale={os_:.6f} | noise={noise:.6f}"
            )
            logger.info(
                f"  last-batch embedding norm: {reduced.detach().norm(dim=1).mean():.4f}"
            )
            if dist_manager.rank == 0 and writer is not None:
                writer.add_scalar(
                    "epoch/gp_lengthscale_mean", ls.mean().item(), epoch,
                )
                writer.add_scalar("epoch/gp_outputscale", os_, epoch)
                writer.add_scalar("epoch/gp_noise", noise, epoch)

        # ================================================================
        # Validation
        # ================================================================
        model.eval()
        embedding_reduction.eval()
        gp.eval()
        gp.likelihood.eval()

        val_epoch_len = len(val_indices)
        val_mse_sum = 0.0
        val_gp_mse_sum = 0.0
        val_consistency_gap_sum = 0.0
        val_metrics_sum: dict[str, float] = {}

        full_val_iter = iter(val_dl_full) if use_consistency else None

        with torch.no_grad():
            for vi, batch in enumerate(val_dl):
                batch_full_v = (
                    next(full_val_iter) if full_val_iter is not None else None
                )

                features = cast_precisions(batch["fx"], precision)
                embeddings_v = cast_precisions(batch["embeddings"], precision)
                targets = batch["fields"]
                geometry = (
                    cast_precisions(batch["geometry"], precision)
                    if "geometry" in batch
                    else None
                )

                with get_autocast_context(precision):
                    if precision == "float8" and TE_AVAILABLE:
                        features, geometry = pad_input_for_fp8(
                            features, embeddings_v, geometry,
                        )

                    local_positions = embeddings_v[:, :, :3]
                    outputs, _, emb_states_v = model(
                        global_embedding=features,
                        local_embedding=embeddings_v,
                        geometry=geometry,
                        local_positions=local_positions,
                    )
                    outputs = unpad_output_for_fp8(outputs, output_pad_size)
                    val_mse = torch.mean(loss_fn(outputs, targets)).item()

                val_mse_sum += val_mse

                # Per-field metrics (unscaled)
                air_density = batch.get("air_density", None)
                stream_velocity = batch.get("stream_velocity", None)
                unscaled_out = tensorwise(val_dl.unscale_model_targets)(
                    outputs,
                    air_density=air_density,
                    stream_velocity=stream_velocity,
                    factor_type=modes,
                )
                unscaled_tgt = tensorwise(val_dl.unscale_model_targets)(
                    targets,
                    air_density=air_density,
                    stream_velocity=stream_velocity,
                    factor_type=modes,
                )
                step_metrics = metrics_fn(
                    unscaled_out, unscaled_tgt, dist_manager, modes,
                )
                if isinstance(step_metrics, list):
                    step_metrics = {
                        k: v for d in step_metrics for k, v in d.items()
                    }
                if vi == 0:
                    val_metrics_sum = {
                        k: float(v) for k, v in step_metrics.items()
                    }
                else:
                    for k in step_metrics:
                        val_metrics_sum[k] = (
                            val_metrics_sum.get(k, 0.0) + float(step_metrics[k])
                        )

                # GP drag validation
                reduced_v = embedding_reduction(emb_states_v.flatten(1, 2))
                drag_target_v = compute_drag_target_from_batch(
                    batch, surface_factors, dist_manager.device,
                ).to(reduced_v.dtype)

                pred_mean, pred_var, _, _ = gp.predict(reduced_v)
                val_gp_mse = F.mse_loss(pred_mean, drag_target_v).item()
                val_gp_mse_sum += val_gp_mse

                # Consistency gap
                if batch_full_v is not None:
                    trans_cd = compute_transolver_drag_full_mesh(
                        batch_full_v, model, chunk_size,
                        surface_factors, dist_manager.device, precision,
                    ).to(pred_mean.dtype)
                    gap = torch.abs(pred_mean - trans_cd).mean().item()
                    val_consistency_gap_sum += gap

                logger.info(
                    f"Val [{vi}/{val_epoch_len}] "
                    f"Field MSE: {val_mse:.6f}  GP MSE: {val_gp_mse:.6f}"
                )

        # ---- Epoch-level validation summary ----
        n = max(val_epoch_len, 1)
        avg_val_mse = val_mse_sum / n
        avg_val_gp_mse = val_gp_mse_sum / n
        avg_val_gap = val_consistency_gap_sum / n
        avg_val_metrics = {k: v / n for k, v in val_metrics_sum.items()}

        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] "
            f"Avg Val GP MSE: {avg_val_gp_mse:.6f}  "
            f"Avg Val Field MSE: {avg_val_mse:.6f}  "
            f"Avg Val Consistency Gap: {avg_val_gap:.6f}"
        )
        if avg_val_metrics:
            table = tabulate(
                [[k, v] for k, v in avg_val_metrics.items()],
                headers=["Metric", "Average Value"],
                tablefmt="pretty",
            )
            logger.info(f"\nEpoch {epoch} Validation Metrics:\n{table}\n")

        if dist_manager.rank == 0 and val_writer is not None:
            val_writer.add_scalar("epoch/mse_loss", avg_val_mse, epoch)
            val_writer.add_scalar("epoch/gp_mse", avg_val_gp_mse, epoch)
            val_writer.add_scalar(
                "epoch/consistency_gap", avg_val_gap, epoch,
            )
            for mk, mv in avg_val_metrics.items():
                val_writer.add_scalar(f"epoch/{mk}", mv, epoch)

        # ---- Checkpoint ----
        save_interval = getattr(cfg.training, "save_interval", 10)
        if epoch % save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch + 1)

        scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="geotransolver_surface_combined")
def launch(cfg: DictConfig):
    main(cfg)


if __name__ == "__main__":
    launch()
