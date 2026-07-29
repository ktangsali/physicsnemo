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

"""Utilities for the GeoTransolver + field-GP per-point UQ pipeline.

This module supports a pointwise multitask GP head
(:class:`~physicsnemo.experimental.uq.FieldVariationalGPHead`) that
*replaces* the GeoTransolver readout: the GP posterior mean is the per-point
surface field prediction (pressure + 3 wall-shear-stress components) and the
posterior variance is the per-point uncertainty.

Provides:

* ``beta_ramp_weight`` — KL-annealing schedule for the variational ELBO.
* ``compute_field_targets_from_batch`` — per-point field-target extraction.
* ``collect_inducing_features`` — gather per-point backbone features to seed
  the GP inducing points.
* ``compute_drag_uq_stats`` — propagate per-point UQ to a drag coefficient.
* Re-exports of common helpers from :mod:`gp_utils`
  (``cast_precisions``, ``sync_non_ddp_gradients``,
  ``compute_force_coefficients_torch``, ``load_pretrained_model_only``).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from physicsnemo.experimental.uq import FieldVariationalGPHead

from gp_utils import (  # noqa: F401 (re-exported for convenience)
    FRONTAL_AREA,
    REFERENCE_DENSITY,
    REFERENCE_VELOCITY,
    cast_precisions,
    compute_force_coefficients_torch,
    load_pretrained_model_only,
    sync_non_ddp_gradients,
)

# Number of surface field channels predicted by the field GP:
#   index 0      -> pressure
#   indices 1:4  -> wall shear stress (x, y, z)
NUM_SURFACE_TASKS = 4


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------


def beta_ramp_weight(epoch: int, warmup_start: int, warmup_end: int) -> float:
    """Linear KL-weight ramp: 0 before *warmup_start*, 0->1 over the window, 1 after.

    During the early epochs the ELBO is dominated by its data-fit (expected
    log-likelihood) term, which behaves like a regression loss and lets the
    backbone + GP mean learn a sensible field before the KL regulariser pulls
    the variational posterior toward the prior.
    """
    if warmup_end <= warmup_start:
        return 1.0
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    return (epoch - warmup_start) / (warmup_end - warmup_start)


# ---------------------------------------------------------------------------
# Head construction
# ---------------------------------------------------------------------------


def build_field_gp_head(
    input_dim: int,
    n_train_points: int,
    num_tasks: int = NUM_SURFACE_TASKS,
    n_inducing: int = 256,
    mlp_hidden: list[int] | None = None,
    lengthscale_range: tuple[float, float] = (0.01, 10.0),
    lengthscale_prior: tuple[float, float] | None = None,
    outputscale_prior: tuple[float, float] | None = None,
    feature_norm: str = "none",
    noise_mlp_hidden: list[int] | None = None,
    noise_std_range: tuple[float, float] = (1e-3, 10.0),
) -> FieldVariationalGPHead:
    """Construct a :class:`FieldVariationalGPHead` with the given hyperparameters.

    Any *eval* script loading a checkpoint must pass the same ``feature_norm``,
    ``mlp_hidden`` and ``noise_mlp_hidden`` used at training time, since those
    determine the module structure and hence the state_dict layout
    (``l2_radial`` adds BatchNorm buffers and one extra kernel dimension).

    ``noise_std_range`` hard-clamps the heteroscedastic noise std.  The lower
    bound matters: the heteroscedastic ELBO weights each point by
    ``1 / sigma^2(x)``, so on unit-scale normalised fields a 1e-3 floor permits
    a weight of 1e6 and a single collapsing point can blow the loss up.  It is
    purely a clamp, so it does not change the state_dict -- but an eval script
    that omits it un-does the guard at inference.
    """
    return FieldVariationalGPHead(
        input_dim=input_dim,
        num_tasks=num_tasks,
        n_inducing=n_inducing,
        n_train=n_train_points,
        mlp_hidden=mlp_hidden,
        lengthscale_range=lengthscale_range,
        lengthscale_prior=lengthscale_prior,
        outputscale_prior=outputscale_prior,
        feature_norm=feature_norm,
        noise_mlp_hidden=noise_mlp_hidden,
        noise_std_range=noise_std_range,
    )


# ---------------------------------------------------------------------------
# Per-point targets
# ---------------------------------------------------------------------------


def compute_field_targets_from_batch(batch: dict) -> torch.Tensor:
    """Return the per-point (normalised) surface field targets.

    Shape ``(B, N, num_tasks)`` — the same normalised space the GP is trained
    in.  Uses ``fields`` (the subsampled stream the GP head also sees).
    """
    fields = batch["fields"]
    if isinstance(fields, list):
        fields = fields[0]
    return fields


# ---------------------------------------------------------------------------
# Inducing-point seeding
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_inducing_features(
    model: nn.Module,
    dataloader: DataLoader,
    n_inducing: int,
    precision: str,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Collect ``n_inducing`` per-point backbone features to seed the GP.

    Runs the backbone (eval mode) over batches, harvesting a random subset of
    per-point features from each until ``n_inducing`` have been gathered.
    Returns a ``(n_inducing, D)`` tensor on *device* (raw feature space; the
    GP head applies its DKL extractor, if any, when these are installed).
    """
    model.eval()
    collected: list[torch.Tensor] = []
    n_have = 0
    for batch in dataloader:
        features = cast_precisions(batch["fx"], precision)
        embeddings = cast_precisions(batch["embeddings"], precision)
        geometry = (
            cast_precisions(batch["geometry"], precision)
            if "geometry" in batch
            else None
        )
        local_positions = embeddings[:, :, :3]
        _, point_features = model(
            global_embedding=features,
            local_embedding=embeddings,
            geometry=geometry,
            local_positions=local_positions,
            return_point_features=True,
        )
        # point_features: (B, N, D) -> (B*N, D)
        pf = point_features.reshape(-1, point_features.shape[-1])
        take = min(pf.shape[0], n_inducing - n_have)
        idx = torch.randperm(pf.shape[0], device=pf.device)[:take]
        collected.append(pf[idx].detach().cpu())
        n_have += take
        if n_have >= n_inducing:
            break

    inducing = torch.cat(collected, dim=0)[:n_inducing].to(device)
    if logger is not None:
        logger.info(
            f"Collected {inducing.shape[0]} inducing-point features "
            f"(dim {inducing.shape[1]}, norm range "
            f"[{inducing.norm(dim=1).min():.4f}, {inducing.norm(dim=1).max():.4f}])"
        )
    return inducing


# ---------------------------------------------------------------------------
# Drag uncertainty propagation
# ---------------------------------------------------------------------------


def default_drag_coeff() -> float:
    """Cd prefactor ``2 / (A * rho * U^2)`` from the reference constants."""
    return 2.0 / (FRONTAL_AREA * REFERENCE_DENSITY * REFERENCE_VELOCITY**2)


@torch.no_grad()
def compute_drag_uq_stats(
    mean_norm: torch.Tensor,
    target_norm: torch.Tensor,
    epi_std_norm: torch.Tensor,
    total_std_norm: torch.Tensor,
    normals: torch.Tensor,
    areas: torch.Tensor,
    surface_factors: dict,
    coeff: float | None = None,
    force_direction: torch.Tensor | None = None,
) -> dict:
    """Integrate per-point field mean/uncertainty into a drag coefficient.

    Drag is a *linear* functional of the surface fields, so:

    * the predicted-drag **mean** is the surface integral of the predicted mean
      field (identical to :func:`compute_force_coefficients_torch`), and
    * the drag **variance** is the area/normal-weighted sum of the per-point
      field variances.

    The variance propagation assumes the per-point GP posterior is *diagonal*
    (independent points).  The real posterior is spatially correlated, so the
    reported drag std is a **lower bound** on the true propagated uncertainty;
    it becomes exact only for an uncorrelated posterior.  This is the standard
    cheap linear-error-propagation estimate and is sufficient for *ranking*
    geometries by drag uncertainty (the quantity sparsification needs).

    Parameters
    ----------
    mean_norm, target_norm, epi_std_norm, total_std_norm : torch.Tensor
        Per-point ``(N, 4)`` tensors in *normalised* target space:
        predicted mean, ground-truth field, epistemic std, total predictive
        std.  Channel 0 is pressure, 1:4 are wall-shear-stress (x, y, z).
    normals : torch.Tensor
        Per-point surface normals ``(N, 3)`` (same convention as
        :func:`compute_force_coefficients_torch`).
    areas : torch.Tensor
        Per-point cell areas ``(N,)`` or ``(N, 1)``.
    surface_factors : dict
        ``{"mean": (4,), "std": (4,)}`` standardisation factors.
    coeff : float | None
        Cd prefactor ``2 / (A * rho * U^2)``; defaults to
        :func:`default_drag_coeff`.  Only sets the overall scale (cancels in
        sparsification / AUSE).
    force_direction : torch.Tensor | None
        Projection unit vector; defaults to ``[1, 0, 0]`` (streamwise drag).

    Returns
    -------
    dict
        ``drag_pred``, ``drag_true``, ``drag_abs_err``, ``drag_epi_std``,
        ``drag_total_std`` (all Python floats).
    """
    device = mean_norm.device
    dtype = mean_norm.dtype
    if coeff is None:
        coeff = default_drag_coeff()
    if force_direction is None:
        force_direction = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    else:
        force_direction = force_direction.to(device=device, dtype=dtype)

    field_mean = surface_factors["mean"].to(device=device, dtype=dtype).view(-1)
    field_std = surface_factors["std"].to(device=device, dtype=dtype).view(-1)

    normals = normals.to(device=device, dtype=dtype)
    area = areas.to(device=device, dtype=dtype).view(-1)

    # Physical mean fields (affine unscale): phys = norm * std + mean.
    pred_phys = mean_norm * field_std + field_mean
    true_phys = target_norm * field_std + field_mean

    drag_pred, _, _ = compute_force_coefficients_torch(
        normals, area, coeff, pred_phys[:, 0], pred_phys[:, 1:4], force_direction
    )
    drag_true, _, _ = compute_force_coefficients_torch(
        normals, area, coeff, true_phys[:, 0], true_phys[:, 1:4], force_direction
    )

    # Per-point linear weights of each field channel on the drag integral.
    #   c_p = coeff * sum( (n . f) * a * p )      -> w_p = coeff * (n . f) * a
    #   c_f = -coeff * sum( (tau . f) * a )       -> w_tau_j = -coeff * a * f_j
    n_dot_f = (normals * force_direction).sum(dim=-1)  # (N,)
    w_p = coeff * n_dot_f * area  # (N,)
    # weights on shear channels (x, y, z); only components along f contribute.
    w_tau = -coeff * area.unsqueeze(-1) * force_direction.view(1, 3)  # (N, 3)

    # Physical per-point variances per channel: var_phys = (std_norm * field_std)^2.
    def _drag_var(std_norm: torch.Tensor) -> torch.Tensor:
        var_phys = (std_norm * field_std) ** 2  # (N, 4)
        var_p = (w_p**2 * var_phys[:, 0]).sum()
        var_tau = (w_tau**2 * var_phys[:, 1:4]).sum()
        return (var_p + var_tau).clamp_min(0)

    drag_epi_std = _drag_var(epi_std_norm).sqrt()
    drag_total_std = _drag_var(total_std_norm).sqrt()

    return {
        "drag_pred": float(drag_pred.item()),
        "drag_true": float(drag_true.item()),
        "drag_abs_err": float((drag_pred - drag_true).abs().item()),
        "drag_epi_std": float(drag_epi_std.item()),
        "drag_total_std": float(drag_total_std.item()),
    }
