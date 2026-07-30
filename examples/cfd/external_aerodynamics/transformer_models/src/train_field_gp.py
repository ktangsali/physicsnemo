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

"""Train GeoTransolver with a pointwise multitask GP field head (from scratch).

The GP head *replaces* the GeoTransolver readout: per-point backbone features
feed a :class:`~physicsnemo.experimental.uq.FieldVariationalGPHead` whose
posterior mean is the per-point surface field prediction (pressure + 3
wall-shear-stress) and whose posterior variance is the per-point uncertainty.  A
single forward pass yields both the field and its UQ.

The total loss is::

    w_nll * neg_elbo + lambda_mean_mse * mse + dist_penalty_weight * dist_pen

where ``neg_elbo`` is the negative multitask variational ELBO on the
(normalised) per-point fields with KL annealing (``beta``), ``mse`` is an
auxiliary anchor on the GP mean that leads the early epochs, and ``dist_pen`` is
the within-sample latent distance penalty (see :func:`_dist_penalty`).  ``beta``
and ``w_nll`` are both ramped from 0 over their warmup windows.

See the "Field GP" section of the example README for the exact launch command.
"""

import os
import re
import glob
import time
import shutil
import collections
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import hydra
import omegaconf
from omegaconf import DictConfig

import torch
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
    update_model_params_for_fp8,
)
from field_gp_utils import (
    beta_ramp_weight,
    collect_inducing_features,
    compute_field_targets_from_batch,
    sync_non_ddp_gradients,
)
from metrics import metrics_fn
from utils import tensorwise

from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

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
# Crash-consistent (atomic) checkpointing
# ---------------------------------------------------------------------------
# In Slurm jobs, a SIGKILL can land *during* the (non-atomic) end-of-
# epoch save, which writes three separate files per epoch:
#   GeoTransolver.0.<tag>.mdlus  (backbone)
#   FieldVariationalGPHead.0.<tag>.pt  (GP head)
#   checkpoint.0.<tag>.pt        (optimizer + scheduler)
# physicsnemo's loader globs the latest index of *each* file independently, so a
# torn write (truncated file) or a partial set (e.g. backbone@N but optimizer@
# N-1) makes a resume load a self-inconsistent state. The variational GP is
# sensitive to that micro-desync, which surfaces as the post-resume z-RMS /
# epistemic-std spikes. The two helpers below make saves atomic and resumes pick
# only a fully-written, consistent epoch set.

_CKPT_GROUPS = (
    ("GeoTransolver.0.*.mdlus", ".mdlus"),
    # physicsnemo names head checkpoints after the class, so runs started before
    # FieldGPHead was renamed to FieldVariationalGPHead use the old stem. Accept
    # either so an in-flight run stays resumable across the rename.
    (("FieldVariationalGPHead.0.*.pt", "FieldGPHead.0.*.pt"), ".pt"),
    ("checkpoint.0.*.pt", ".pt"),
)


def _ckpt_indices(ckpt_dir: str, pattern: str | tuple[str, ...]) -> set[int]:
    """Epoch tags present for one checkpoint-file pattern in ``ckpt_dir``.

    ``pattern`` may be a tuple of alternatives, in which case the union of their
    matches is returned.
    """
    patterns = (pattern,) if isinstance(pattern, str) else pattern
    out: set[int] = set()
    for pat in patterns:
        for f in glob.glob(os.path.join(ckpt_dir, pat)):
            m = re.search(r"\.(\d+)\.[^.]+$", os.path.basename(f))
            if m:
                out.add(int(m.group(1)))
    return out


def _safe_resume_epoch(ckpt_dir: str) -> int | None:
    """Largest epoch tag with a COMPLETE set (backbone + head + training state).

    Returns ``None`` for a fresh run (no complete set), so the caller falls back
    to ``load_checkpoint(epoch=None)`` which is a no-op on an empty directory.
    Passing the explicit epoch sidesteps physicsnemo's per-file "latest index"
    globbing, which could otherwise pair an orphan backbone@N with optimizer@N-1.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    complete = (
        _ckpt_indices(ckpt_dir, _CKPT_GROUPS[0][0])
        & _ckpt_indices(ckpt_dir, _CKPT_GROUPS[1][0])
        & _ckpt_indices(ckpt_dir, _CKPT_GROUPS[2][0])
    )
    return max(complete) if complete else None


def _fsync_path(path: str) -> None:
    """Flush a file's (or directory's) contents to stable storage."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_save_checkpoint(ckpt_args: dict, epoch: int) -> None:
    """Crash-consistent wrapper around :func:`save_checkpoint`.

    Writes the full checkpoint set into a staging subdir, fsyncs every file,
    then atomically ``os.replace``-moves them into the live checkpoint dir with
    the training-state file (``checkpoint.*.pt``) committed **last**. Because
    ``os.replace`` is atomic on the same filesystem and the model/head files
    always land before the training-state file, a reader (resume) never sees a
    truncated file, and the presence of ``checkpoint.0.<tag>.pt`` is a commit
    marker for a consistent (backbone + head + optimizer) set.
    """
    ckpt_dir = str(ckpt_args["path"])
    staging = os.path.join(ckpt_dir, f".staging_{epoch}")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    save_args = dict(ckpt_args)
    save_args["path"] = staging
    save_checkpoint(**save_args, epoch=epoch)

    produced = sorted(
        p for p in glob.glob(os.path.join(staging, "*")) if os.path.isfile(p)
    )
    for f in produced:
        _fsync_path(f)
    # Commit order: backbone/head first, training-state file last.
    train_state = [f for f in produced if os.path.basename(f).startswith("checkpoint.")]
    others = [f for f in produced if f not in train_state]
    for f in others + train_state:
        os.replace(f, os.path.join(ckpt_dir, os.path.basename(f)))
    _fsync_path(ckpt_dir)
    shutil.rmtree(staging, ignore_errors=True)


def _maybe_subsample(
    point_features: torch.Tensor,
    target: torch.Tensor,
    n_points: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally subsample the point dimension to ``n_points`` for the GP step."""
    if n_points is None:
        return point_features, target
    n_total = point_features.shape[1]
    if n_points >= n_total:
        return point_features, target
    idx = torch.randperm(n_total, device=point_features.device)[:n_points]
    return point_features[:, idx], target[:, idx]


def _dist_penalty(
    gp_feats: torch.Tensor,
    target: torch.Tensor,
    n_pairs: int,
    margin: float,
) -> torch.Tensor:
    """Within-sample latent distance penalty (bi-Lipschitz-style, target-aware).

    For random point pairs inside each geometry, penalise pairs whose GP-input
    (kernel-space) distance is *smaller* than their target-space distance. This
    forces the encoder to keep points with different target fields far apart in
    the space the kernel measures, so the GP can assign larger variance where
    predictions differ.

    Both distances are made scale-free by dividing by their (detached) batch
    mean, so the penalty shapes *relative* geometry rather than absolute scale
    (which the feature-norm already pins). One-sided hinge => pairs that are
    already well-separated are not pulled together.

    Parameters
    ----------
    gp_feats : (B, P, D) tensor
        Per-point features in GP-input space (output of ``transform_features``),
        with gradient to the backbone.
    target : (B, P, T) tensor
        Per-point field targets.
    n_pairs : int
        Random point pairs sampled per geometry.
    margin : float
        Multiplier on the (normalised) target distance the latent distance must
        meet or exceed before the hinge stops penalising.
    """
    if gp_feats.dim() != 3:
        gp_feats = gp_feats.reshape(1, -1, gp_feats.shape[-1])
        target = target.reshape(1, -1, target.shape[-1])
    b, p, _ = gp_feats.shape
    if p < 2 or n_pairs < 1:
        return gp_feats.new_zeros(())
    target = target.to(gp_feats.dtype)
    total = gp_feats.new_zeros(())
    for k in range(b):
        i = torch.randint(0, p, (n_pairs,), device=gp_feats.device)
        j = torch.randint(0, p, (n_pairs,), device=gp_feats.device)
        d_lat = (gp_feats[k, i] - gp_feats[k, j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        d_tgt = (target[k, i] - target[k, j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        d_lat = d_lat / (d_lat.mean().detach() + 1e-12)
        d_tgt = d_tgt / (d_tgt.mean().detach() + 1e-12)
        total = total + torch.relu(margin * d_tgt - d_lat).mean()
    return total / max(b, 1)


def main(cfg: DictConfig):
    """Run GeoTransolver + field-GP training from scratch."""
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    logger = RankZeroLoggingWrapper(
        PythonLogger(name="field_gp_training"),
        dist_manager,
    )

    # ---- Field-GP config ----
    # cfg.gp_head is a hydra _target_ block; the eval scripts instantiate the
    # same block, so the structure baked into the checkpoint cannot drift apart.
    n_inducing = cfg.gp_head.n_inducing

    # KL annealing window (epochs) and auxiliary mean-MSE weight
    beta_warmup_start = getattr(cfg, "beta_warmup_start", 0)
    beta_warmup_end = getattr(cfg, "beta_warmup_end", 20)
    lambda_mean_mse = getattr(cfg, "lambda_mean_mse", 1.0)
    # Optional scale on the (fully-ramped) KL term. <1.0 keeps the variational
    # posterior more input-dependent (sharper / heteroscedastic) instead of
    # collapsing toward the prior. Default 1.0 == standard ELBO.
    kl_weight = float(getattr(cfg, "gp_kl_weight", 1.0))

    # ---- NLL (ELBO) warmup + latent distance penalty ----
    # NLL warmup: ramp the weight on the *whole* negative ELBO 0->1 over
    # [nll_warmup_start, nll_warmup_end). Early on the loss is dominated by the
    # mean-MSE anchor (lambda_mean_mse) so the backbone + GP mean learn an
    # accurate, non-collapsed field before the likelihood term can flatten the
    # variance. Empty window (end<=start) => weight 1.0 always (disabled).
    nll_warmup_start = int(getattr(cfg, "nll_warmup_start", 0))
    nll_warmup_end = int(getattr(cfg, "nll_warmup_end", 0))
    # Distance penalty (within-sample, point-level): a one-sided hinge that
    # forces pairs of points with large target differences to also be far apart
    # in GP-input (kernel) space, so the kernel can express larger variance
    # where predictions differ. 0.0 => disabled.
    dist_penalty_weight = float(getattr(cfg, "dist_penalty_weight", 0.0))
    dist_penalty_pairs = int(getattr(cfg, "dist_penalty_pairs", 4096))
    dist_penalty_margin = float(getattr(cfg, "dist_penalty_margin", 1.0))

    # Max global grad-norm for the GP head (0 / unset = no clipping, which is
    # what every earlier run used). Catches the overcorrection that follows a
    # noise collapse; gp_head.noise_std_range prevents the collapse itself.
    grad_clip_norm = float(getattr(cfg, "head_grad_clip_norm", 0.0) or 0.0)

    # Optional further subsampling of points used by the GP each step
    gp_points_per_step = getattr(cfg, "gp_points_per_step", None)
    # Points per geometry used to size the ELBO num_data normaliser
    points_per_geometry = getattr(cfg, "points_per_geometry", None) or (
        getattr(cfg.data, "resolution", 51200) or 51200
    )

    accumulation_steps = getattr(cfg.training, "gradient_accumulation_steps", 1)

    # ---- Directories and writers ----
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None) or cfg.output_dir
    ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_field_gp"

    if dist_manager.rank == 0:
        os.makedirs(ckpt_path, exist_ok=True)
        writer = SummaryWriter(log_dir=f"{cfg.output_dir}/{cfg.run_id}/field_gp_train")
        val_writer = SummaryWriter(
            log_dir=f"{cfg.output_dir}/{cfg.run_id}/field_gp_val"
        )
    else:
        writer = val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Output directory: {cfg.output_dir}/{cfg.run_id}")
    logger.info(f"Checkpoint directory: {ckpt_path}")
    logger.info(f"Field GP head: {omegaconf.OmegaConf.to_container(cfg.gp_head)}")
    logger.info(
        f"KL warmup epochs [{beta_warmup_start}, {beta_warmup_end}), "
        f"lambda_mean_mse={lambda_mean_mse}, gp_points_per_step={gp_points_per_step}"
    )
    logger.info(
        f"NLL warmup epochs [{nll_warmup_start}, {nll_warmup_end}); "
        f"dist_penalty: weight={dist_penalty_weight}, pairs={dist_penalty_pairs}, "
        f"margin={dist_penalty_margin}"
    )
    logger.info(f"head_grad_clip_norm={grad_clip_norm}")

    precision = cfg.precision
    cfg, _ = update_model_params_for_fp8(cfg, logger)

    # ---- GeoTransolver backbone ----
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.to(dist_manager.device)

    # The GP head replaces the readout, so the backbone's output projection
    # (`ln_mlp_out`) is intentionally unused — DDP must tolerate that.
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
        find_unused_parameters=True,
    )
    num_geo_params = sum(p.numel() for p in model.parameters())
    logger.info(f"GeoTransolver parameters: {num_geo_params:,}")

    # ---- Normalization factors (surface) ----
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    surface_factors = None
    if cfg.data.mode in ("surface", "combined"):
        nd = np.load(str(Path(norm_dir) / "surface_fields_normalization.npz"))
        surface_factors = {
            "mean": torch.from_numpy(nd["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(nd["std"]).to(dist_manager.device),
        }

    # ---- Dataloaders ----
    train_dl = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=None,
    )
    val_dl = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=None,
    )

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dl,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dl,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,
        drop_last=True,
    )

    # ---- Field-GP head ----
    n_train = len(train_dl)
    # ELBO normaliser: total number of training points seen across the dataset.
    n_train_points = max(n_train * points_per_geometry, 1)
    # Probe the backbone feature dimension with one forward pass. Set indices
    # first since the per-epoch sampler hasn't assigned them yet.
    train_dl.dataset.set_indices(list(range(n_train)))
    feature_dim = _probe_feature_dim(
        model, train_dl, precision, dist_manager.device, logger
    )
    head = hydra.utils.instantiate(
        cfg.gp_head,
        input_dim=feature_dim,
        n_train=n_train_points,
        _convert_="all",
    )
    head.to(dist_manager.device)
    num_head_params = sum(p.numel() for p in head.parameters())
    logger.info(
        f"Field GP head parameters: {num_head_params:,} "
        f"(feature_dim={feature_dim}, gp_dim={head.gp_input_dim}, "
        f"n_train_points={n_train_points:,})"
    )

    # ---- Optimizer ----
    # Rates are set relative to cfg.training.optimizer.lr and split by parameter
    # kind: GP-native parameters (variational, kernel, noise scale) at 10x it,
    # network weights at 1x. Keep that ratio when retuning; see "Choosing the
    # head learning rates" in the README.
    head_param_groups = [
        {"params": head.gp_layer.variational_parameters(), "lr": 1e-2},
        {"params": head.gp_layer.hyperparameters(), "lr": 1e-2},
        {"params": head.likelihood.parameters(), "lr": 1e-2},
    ]
    if head.feature_extractor is not None:
        head_param_groups.append(
            {"params": head.feature_extractor.parameters(), "lr": 1e-3}
        )
    if head.noise_head is not None:
        head_param_groups.append({"params": head.noise_head.parameters(), "lr": 1e-3})
        head_param_groups.append({"params": [head.log_base_noise], "lr": 1e-2})
    head_opt = torch.optim.AdamW(head_param_groups, weight_decay=1e-4)
    geo_muon_params = [p for p in model.parameters() if p.ndim == 2]
    geo_other_params = [p for p in model.parameters() if p.ndim != 2]
    geo_adamw = hydra.utils.instantiate(
        cfg.training.optimizer,
        params=geo_other_params,
    )
    geo_muon = torch.optim.Muon(
        geo_muon_params,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    optimizer = CombinedOptimizer([geo_muon, geo_adamw, head_opt])

    # ---- Scheduler ----
    scheduler_params = dict(cfg.training.scheduler.params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    scaler = GradScaler() if precision == "float16" else None
    if precision == "float8" and not TE_AVAILABLE:
        raise ImportError("TransformerEngine is required for float8 precision.")

    # ---- Checkpoint ----
    ckpt_args = {
        "path": ckpt_path,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": [model, head],
    }
    # Resume only from a fully-written, consistent epoch set (atomic-save guard);
    # explicit epoch avoids physicsnemo's per-file "latest index" globbing pairing
    # mismatched files after a preemption mid-save.
    resume_epoch = _safe_resume_epoch(ckpt_path)
    loaded_epoch = load_checkpoint(
        device=dist_manager.device, epoch=resume_epoch, **ckpt_args
    )
    inducing_init_done = loaded_epoch > 0

    modes = ["surface"]

    # ==================================================================
    # Training loop
    # ==================================================================
    logger.info("Starting field-GP training ...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_indices = list(train_sampler)
        val_indices = list(val_sampler)
        train_dl.dataset.set_indices(train_indices)
        val_dl.dataset.set_indices(val_indices)

        epoch_len = len(train_indices)
        beta = kl_weight * beta_ramp_weight(epoch, beta_warmup_start, beta_warmup_end)
        # Ramp the whole negative ELBO 0->1 so mean-MSE (+ dist penalty) lead the
        # early epochs and the likelihood term only shapes variance once the mean
        # field is accurate. Empty window => 1.0 (standard joint ELBO).
        w_nll = beta_ramp_weight(epoch, nll_warmup_start, nll_warmup_end)

        # Seed inducing points from real features before the first epoch.
        if not inducing_init_done:
            inducing = collect_inducing_features(
                model,
                train_dl,
                n_inducing,
                precision,
                dist_manager.device,
                logger,
            )
            # The head is not DDP-wrapped, so every rank must start from the
            # SAME inducing points; each rank collected from its own data shard,
            # so broadcast rank 0's to all ranks for a consistent init.
            if dist_manager.world_size > 1:
                dist.broadcast(inducing, src=0)
            head.set_inducing_points(inducing)
            inducing_init_done = True
            train_dl.dataset.set_indices(train_indices)

        epoch_elbo = 0.0
        epoch_mean_mse = 0.0
        epoch_total = 0.0
        epoch_dist_pen = 0.0

        model.train()
        head.train()
        head.likelihood.train()

        start_time = time.time()
        for i, batch in enumerate(train_dl):
            features = cast_precisions(batch["fx"], precision)
            embeddings = cast_precisions(batch["embeddings"], precision)
            targets = compute_field_targets_from_batch(batch).to(dist_manager.device)
            geometry = (
                cast_precisions(batch["geometry"], precision)
                if "geometry" in batch
                else None
            )

            with get_autocast_context(precision):
                if precision == "float8" and TE_AVAILABLE:
                    features, geometry = pad_input_for_fp8(
                        features, embeddings, geometry
                    )
                if geometry is None:
                    raise RuntimeError(
                        "Field-GP training requires a GeoTransolver (geometry) model."
                    )
                local_positions = embeddings[:, :, :3]
                # Per-point features are taken *before* the readout projection,
                # so they are not subject to the fp8 output padding (no unpad).
                _, point_features = model(
                    global_embedding=features,
                    local_embedding=embeddings,
                    geometry=geometry,
                    local_positions=local_positions,
                    return_point_features=True,
                )

            pf_step, tgt_step = _maybe_subsample(
                point_features, targets, gp_points_per_step
            )
            # Transform once into GP-input (kernel) space so both the ELBO and
            # the distance penalty see the exact features the kernel measures.
            gp_feats = head.transform_features(pf_step)
            # GP forward + ELBO (run outside autocast for float64 GP internals).
            mean, neg_elbo = head.forward_and_loss(
                gp_feats, tgt_step.to(gp_feats.dtype), beta=beta, pretransformed=True
            )
            mean_mse = F.mse_loss(mean, tgt_step.to(mean.dtype))
            if dist_penalty_weight > 0.0:
                dist_pen = _dist_penalty(
                    gp_feats, tgt_step, dist_penalty_pairs, dist_penalty_margin
                )
            else:
                dist_pen = gp_feats.new_zeros(())
            total_loss = (
                w_nll * neg_elbo
                + lambda_mean_mse * mean_mse
                + dist_penalty_weight * dist_pen
            )

            head_grad_norm: float | None = None
            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            is_step_boundary = (i + 1) % accumulation_steps == 0 or (i + 1) == epoch_len
            sync_ctx = model.no_sync() if not is_step_boundary else nullcontext()
            scaled_loss = total_loss / accumulation_steps
            with sync_ctx:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if is_step_boundary:
                sync_non_ddp_gradients([head], dist_manager.world_size)
                if grad_clip_norm > 0.0:
                    if scaler is not None:
                        # Gradients are still scaled; unscale in place first so
                        # the clip threshold is in true gradient units.
                        scaler.unscale_(optimizer)
                    # Returns the pre-clip norm, which is the diagnostic we
                    # actually want logged -- it shows how often (and by how
                    # much) the threshold binds.
                    head_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            head.parameters(), grad_clip_norm
                        )
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            elbo_val = neg_elbo.detach().item()
            mean_mse_val = mean_mse.detach().item()
            dist_pen_val = dist_pen.detach().item()
            total_val = total_loss.detach().item()
            epoch_elbo += elbo_val
            epoch_mean_mse += mean_mse_val
            epoch_dist_pen += dist_pen_val
            epoch_total += total_val

            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time

            gnorm_str = (
                f"  HeadGradNorm: {head_grad_norm:.3f}"
                if head_grad_norm is not None
                else ""
            )
            logger.info(
                f"Epoch {epoch} [{i}/{epoch_len}] "
                f"NegELBO: {elbo_val:.6f}  Mean MSE: {mean_mse_val:.6f}  "
                f"DistPen: {dist_pen_val:.6f}  "
                f"Total: {total_val:.6f}  beta: {beta:.3f}  w_nll: {w_nll:.3f}  "
                f"Duration: {duration:.2f}s{gnorm_str}"
            )
            if dist_manager.rank == 0 and writer is not None:
                gs = i + epoch_len * epoch
                writer.add_scalar("batch/neg_elbo", elbo_val, gs)
                writer.add_scalar("batch/mean_mse", mean_mse_val, gs)
                writer.add_scalar("batch/dist_penalty", dist_pen_val, gs)
                writer.add_scalar("batch/total_loss", total_val, gs)
                writer.add_scalar("batch/beta", beta, gs)
                writer.add_scalar("batch/w_nll", w_nll, gs)
                writer.add_scalar(
                    "batch/learning_rate", optimizer.param_groups[0]["lr"], gs
                )

        n = max(epoch_len, 1)
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] "
            f"Avg NegELBO: {epoch_elbo / n:.6f}  "
            f"Avg Mean MSE: {epoch_mean_mse / n:.6f}  "
            f"Avg DistPen: {epoch_dist_pen / n:.6f}  "
            f"Avg Total: {epoch_total / n:.6f}"
        )
        if dist_manager.rank == 0 and writer is not None:
            writer.add_scalar("epoch/neg_elbo", epoch_elbo / n, epoch)
            writer.add_scalar("epoch/mean_mse", epoch_mean_mse / n, epoch)
            writer.add_scalar("epoch/dist_penalty", epoch_dist_pen / n, epoch)
            writer.add_scalar("epoch/total_loss", epoch_total / n, epoch)
            writer.add_scalar("epoch/beta", beta, epoch)
            writer.add_scalar("epoch/w_nll", w_nll, epoch)

        if beta > 0:
            os_ = head.gp_layer.covar_module.outputscale.detach().cpu()
            noise = head.likelihood.noise.detach().cpu()
            ls = head.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu()
            logger.info(
                f"  GP hypers — lengthscale mean={ls.mean():.4f} | "
                f"outputscale mean={os_.mean():.6f} | noise mean={noise.mean():.6f}"
            )

        # ==============================================================
        # Validation
        # ==============================================================
        _validate(
            model,
            head,
            val_dl,
            val_indices,
            precision,
            modes,
            dist_manager,
            logger,
            val_writer,
            epoch,
        )

        # Checkpoints are tagged ``epoch + 1`` (the numbering the inference and
        # evaluation tooling expects).
        save_interval = getattr(cfg.training, "save_interval", 10)
        if dist_manager.rank == 0 and epoch % save_interval == 0:
            _atomic_save_checkpoint(ckpt_args, epoch=epoch + 1)

        scheduler.step()

    logger.info("Training completed!")


@torch.no_grad()
def _probe_feature_dim(model, dataloader, precision, device, logger) -> int:
    """Return the per-point backbone feature dimension via one forward pass."""
    model.eval()
    batch = next(iter(dataloader))
    features = cast_precisions(batch["fx"], precision)
    embeddings = cast_precisions(batch["embeddings"], precision)
    geometry = (
        cast_precisions(batch["geometry"], precision) if "geometry" in batch else None
    )
    local_positions = embeddings[:, :, :3]
    _, point_features = model(
        global_embedding=features,
        local_embedding=embeddings,
        geometry=geometry,
        local_positions=local_positions,
        return_point_features=True,
    )
    dim = int(point_features.shape[-1])
    logger.info(f"Probed per-point backbone feature dim: {dim}")
    return dim


def _validate(
    model,
    head,
    val_dl,
    val_indices,
    precision,
    modes,
    dist_manager,
    logger,
    val_writer,
    epoch,
):
    """Validate the GP mean field (metrics) and log mean predictive std."""
    model.eval()
    head.eval()
    head.likelihood.eval()
    # Per-geometry, per-point rank correlation between epistemic std and |error|
    # (averaged over channels): the within-sample error<->variance link.
    insample_corr_local: list[float] = []

    val_epoch_len = len(val_indices)
    val_metrics_sum: dict[str, float] = {}
    std_sum = np.zeros(head.num_tasks, dtype=np.float64)
    n_seen = 0

    # Calibration / likelihood accumulators (normalised target space), summed
    # over every point & channel, then all-reduced so every rank agrees on the
    # early-stopping decision (avoids a distributed deadlock on the stop break).
    device = dist_manager.device
    nlpd_sum = torch.zeros((), dtype=torch.float64, device=device)
    z2_sum = torch.zeros((), dtype=torch.float64, device=device)
    cov_sum = torch.zeros((), dtype=torch.float64, device=device)
    stdscalar_sum = torch.zeros((), dtype=torch.float64, device=device)
    elem_count = torch.zeros((), dtype=torch.float64, device=device)
    log_2pi = float(np.log(2.0 * np.pi))

    with torch.no_grad():
        for vi, batch in enumerate(val_dl):
            features = cast_precisions(batch["fx"], precision)
            embeddings = cast_precisions(batch["embeddings"], precision)
            targets = compute_field_targets_from_batch(batch).to(dist_manager.device)
            geometry = (
                cast_precisions(batch["geometry"], precision)
                if "geometry" in batch
                else None
            )
            with get_autocast_context(precision):
                if precision == "float8" and TE_AVAILABLE:
                    features, geometry = pad_input_for_fp8(
                        features, embeddings, geometry
                    )
                local_positions = embeddings[:, :, :3]
                _, point_features = model(
                    global_embedding=features,
                    local_embedding=embeddings,
                    geometry=geometry,
                    local_positions=local_positions,
                    return_point_features=True,
                )

            pred = head.predict(point_features)
            mean = pred.mean.to(targets.dtype)
            std = pred.variance.clamp_min(0).sqrt()

            # Per-channel mean std (in normalised space).
            std_sum += (
                std.reshape(-1, head.num_tasks).mean(dim=0).double().cpu().numpy()
            )
            n_seen += 1

            # ---- Within-sample per-point error<->epistemic-std correlation ----
            epi = pred.epistemic_variance.clamp_min(0).sqrt()
            aen = (
                (mean - targets)
                .abs()
                .reshape(-1, head.num_tasks)
                .double()
                .cpu()
                .numpy()
            )
            esn = epi.reshape(-1, head.num_tasks).double().cpu().numpy()
            _corrs = []
            for _c in range(head.num_tasks):
                _a, _b = esn[:, _c], aen[:, _c]
                if _a.std() > 0 and _b.std() > 0:
                    _ra = np.argsort(np.argsort(_a)).astype(np.float64)
                    _rb = np.argsort(np.argsort(_b)).astype(np.float64)
                    _corrs.append(float(np.corrcoef(_ra, _rb)[0, 1]))
            if _corrs:
                insample_corr_local.append(float(np.mean(_corrs)))

            # NLPD / z-RMS / 95%-coverage on the TOTAL predictive variance
            # (epistemic + observation-noise floor) — the proper scoring rule we
            # early-stop on. Computed in the same normalised space the GP trains
            # in, summed over all points & channels.
            err64 = (mean - targets).double()
            std64 = std.double()
            var64 = std64.clamp_min(1e-12) ** 2
            nlpd_sum += (0.5 * (log_2pi + torch.log(var64) + err64**2 / var64)).sum()
            z2_sum += (err64**2 / var64).sum()
            cov_sum += (err64.abs() <= 1.96 * std64).double().sum()
            stdscalar_sum += std64.sum()
            elem_count += float(err64.numel())

            air_density = batch.get("air_density", None)
            stream_velocity = batch.get("stream_velocity", None)
            unscaled_pred = tensorwise(val_dl.unscale_model_targets)(
                mean,
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
            step_metrics = metrics_fn(unscaled_pred, unscaled_tgt, dist_manager, modes)
            if isinstance(step_metrics, list):
                step_metrics = {k: v for d in step_metrics for k, v in d.items()}
            if vi == 0:
                val_metrics_sum = {k: float(v) for k, v in step_metrics.items()}
            else:
                for k in step_metrics:
                    val_metrics_sum[k] = val_metrics_sum.get(k, 0.0) + float(
                        step_metrics[k]
                    )

            logger.info(
                f"Val [{vi}/{val_epoch_len}] mean std (norm): "
                f"{std.reshape(-1, head.num_tasks).mean(dim=0).cpu().numpy()}"
            )

    vn = max(val_epoch_len, 1)
    avg_metrics = {k: v / vn for k, v in val_metrics_sum.items()}
    avg_std = std_sum / max(n_seen, 1)
    logger.info(f"Epoch [{epoch}] Avg per-channel predictive std (norm): {avg_std}")
    if avg_metrics:
        table = tabulate(
            [[k, v] for k, v in avg_metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch} Validation Metrics:\n{table}\n")

    if dist_manager.rank == 0 and val_writer is not None:
        for mk, mv in avg_metrics.items():
            val_writer.add_scalar(f"epoch/{mk}", mv, epoch)
        for t in range(head.num_tasks):
            val_writer.add_scalar(f"epoch/pred_std_task{t}", float(avg_std[t]), epoch)

    # ---- Global (all-rank) calibration / likelihood metrics ----
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        packed = torch.stack([nlpd_sum, z2_sum, cov_sum, stdscalar_sum, elem_count])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        nlpd_sum, z2_sum, cov_sum, stdscalar_sum, elem_count = packed.unbind()
    ne = float(elem_count.item())
    ne = ne if ne > 0 else 1.0
    val_nlpd = float(nlpd_sum.item()) / ne
    val_zrms = (float(z2_sum.item()) / ne) ** 0.5
    val_cov95 = float(cov_sum.item()) / ne
    val_mean_std = float(stdscalar_sum.item()) / ne
    # ---- Within-sample error<->std correlation across ALL val geometries ----
    # Gather the per-geometry values from every rank so rank 0 can average over
    # the whole split. All ranks must call the collective, even empty shards.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        g_ins: list = [None] * dist.get_world_size()
        dist.all_gather_object(g_ins, insample_corr_local)
        all_ins = [v for part in g_ins for v in part]
    else:
        all_ins = insample_corr_local
    val_insample_corr = float(np.mean(all_ins)) if all_ins else float("nan")

    logger.info(
        f"Epoch [{epoch}] val NLPD(total)={val_nlpd:.4f}  z-RMS={val_zrms:.3f}  "
        f"cov95={val_cov95:.3f}  mean_std={val_mean_std:.4f}  "
        f"pt_err_std_corr={val_insample_corr:+.3f}"
    )
    if dist_manager.rank == 0 and val_writer is not None:
        val_writer.add_scalar("epoch/val_nlpd_total", val_nlpd, epoch)
        val_writer.add_scalar("epoch/val_zrms", val_zrms, epoch)
        val_writer.add_scalar("epoch/val_cov95", val_cov95, epoch)
        if all_ins:
            val_writer.add_scalar("epoch/val_pt_err_std_corr", val_insample_corr, epoch)

    return {
        "nlpd": val_nlpd,
        "zrms": val_zrms,
        "cov95": val_cov95,
        "mean_std": val_mean_std,
        "pt_err_std_corr": val_insample_corr,
    }


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="geotransolver_surface_field_gp",
)
def launch(cfg: DictConfig):
    """Hydra entry point for field-GP training."""
    main(cfg)


if __name__ == "__main__":
    launch()
