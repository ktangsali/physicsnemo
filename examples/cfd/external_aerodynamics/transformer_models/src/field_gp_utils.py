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
replaces the GeoTransolver readout: the GP posterior mean is the per-point
surface field prediction (pressure + 3 wall-shear-stress components) and the
posterior variance is the per-point uncertainty.

Provides:

* ``beta_ramp_weight`` — KL-annealing schedule for the variational ELBO.
* ``compute_field_targets_from_batch`` — per-point field-target extraction.
* ``collect_inducing_features`` — gather per-point backbone features to seed
  the GP inducing points.
* Re-export of ``sync_non_ddp_gradients`` from :mod:`gp_utils`.

"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gp_utils import (  # noqa: F401 (sync_non_ddp_gradients is re-exported)
    cast_precisions,
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
