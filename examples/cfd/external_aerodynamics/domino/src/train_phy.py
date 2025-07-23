# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
This code defines a distributed pipeline for training the DoMINO model on
CFD datasets. It includes the computation of scaling factors, instantiating 
the DoMINO model and datapipe, automatically loading the most recent checkpoint, 
training the model in parallel using DistributedDataParallel across multiple 
GPUs, calculating the loss and updating model parameters using mixed precision. 
This is a common recipe that enables training of combined models for surface and 
volume as well either of them separately. Validation is also conducted every epoch, 
where predictions are compared against ground truth values. The code logs training
and validation metrics to TensorBoard. The train tab in config.yaml can be used to 
specify batch size, number of epochs and other training parameters.
"""

import time
import os
import re
import torch
import torchinfo

from typing import Literal

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx


from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import (
    DoMINODataPipe,
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino_phy.model import DoMINOPhy
from physicsnemo.utils.domino.utils import *

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time

# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler

# Profiler().enable("line_profiler")
# Profiler().initialize()


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
    vol_factors: None = torch.Tensor,
) -> torch.Tensor:
    """Calculate mean squared error or root mean squared error with masking for padded values.

    Args:
        output: Predicted values from the model
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor

    Returns:
        Calculated loss as a scalar tensor
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    output, grad, grad_grad_uvw = output
    
    # unnormalize gradients
    # Grads have shape [1, batch_size, 3, num_variables]
    # 0, 1, 2 -> u, v, w; 3 -> p; 4 -> nut
    # print(output.device, vol_factors.device)
    # vol_factors = torch.from_numpy(vol_factors, device=output.device)
    fields = unnormalize(output, vol_factors[0], vol_factors[1])
    grad = unnormalize_grad(grad, vol_factors[0], vol_factors[1])
    grad_grad_uvw = unnormalize_grad(grad_grad_uvw, vol_factors[0][0:3].repeat_interleave(3), vol_factors[1][0:3].repeat_interleave(3))

    rho = 1.226
    nu = 1.507 * 1e-5

    u = fields[:,:,0:1]
    v = fields[:,:,1:2]
    w = fields[:,:,2:3]
    
    mu = rho * (nu + fields[:,:,4:5])   # mu + mu_t
    mu_grad = rho * grad[:,:,:,4:5]
    mu__x = mu_grad[:,:,0,:]
    mu__y = mu_grad[:,:,1,:]
    mu__z = mu_grad[:,:,2,:]

    # compute residuals
    p__x = grad[:,:,0,3:4]
    p__y = grad[:,:,1,3:4]
    p__z = grad[:,:,2,3:4]
    u__x = grad[:,:,0,0:1]
    u__y = grad[:,:,1,0:1]
    u__z = grad[:,:,2,0:1]
    v__x = grad[:,:,0,1:2]
    v__y = grad[:,:,1,1:2]
    v__z = grad[:,:,2,1:2]
    w__x = grad[:,:,0,2:3]
    w__y = grad[:,:,1,2:3]
    w__z = grad[:,:,2,2:3]
    u__x__x = grad_grad_uvw[:,:,0,0:1]
    u__x__y = grad_grad_uvw[:,:,1,0:1]
    u__x__z = grad_grad_uvw[:,:,2,0:1]
    u__y__y = grad_grad_uvw[:,:,1,1:2]
    u__y__z = grad_grad_uvw[:,:,2,1:2]
    u__z__z = grad_grad_uvw[:,:,2,2:3]
    v__x__x = grad_grad_uvw[:,:,0,3:4]
    v__x__y = grad_grad_uvw[:,:,1,3:4]
    v__x__z = grad_grad_uvw[:,:,2,3:4]
    v__y__y = grad_grad_uvw[:,:,1,4:5]
    v__y__z = grad_grad_uvw[:,:,2,4:5]
    v__z__z = grad_grad_uvw[:,:,2,5:6]
    w__x__x = grad_grad_uvw[:,:,0,6:7]
    w__x__y = grad_grad_uvw[:,:,1,6:7]
    w__x__z = grad_grad_uvw[:,:,2,6:7]
    w__y__y = grad_grad_uvw[:,:,1,7:8]
    w__y__z = grad_grad_uvw[:,:,2,7:8]
    w__z__z = grad_grad_uvw[:,:,2,8:9]

    u__y__x = u__x__y
    u__z__x = u__x__z
    u__z__y = u__y__z
    v__y__x = v__x__y
    v__z__x = v__x__z
    v__z__y = v__y__z
    w__y__x = w__x__y
    w__z__x = w__x__z
    w__z__y = w__y__z
    
    tau_xx__x = 2 * mu * u__x__x + 2 * mu__x * u__x
    tau_xy__y = mu * (u__y__y + v__x__y) + mu__y * (u__y + v__x)
    tau_xz__z = mu * (u__z__z + w__x__z) + mu__z * (u__z + w__x)
    tau_xy__x = mu * (u__y__x + v__x__x) + mu__x * (u__y + v__x)
    tau_yy__y = 2 * mu * v__y__y + 2 * mu__y * v__y
    tau_yz__z = mu * (v__z__z + w__y__z) + mu__z * (v__z + w__y)
    tau_xz__x = mu * (u__z__x + w__x__x) + mu__x * (u__z + w__x) 
    tau_yz__y = mu * (v__z__y + w__y__y) + mu__y * (v__z + w__y)
    tau_zz__z = 2 * mu * w__z__z + 2 * mu__z * w__z

    continuity = u__x + v__y + w__z
    momentum_x = rho * (u * u__x + v * u__y + w * u__z) + p__x - tau_xx__x - tau_xy__y - tau_xz__z
    momentum_y = rho * (u * v__x + v * v__y + w * v__z) + p__y - tau_xy__x - tau_yy__y - tau_yz__z
    momentum_z = rho * (u * w__x + v * w__y + w * w__z) + p__z - tau_xz__x - tau_yz__y - tau_zz__z
    
    # print(f"Conitnuity and Momentum shapes: {continuity.shape}, {momentum_x.shape}, {momentum_y.shape}, {momentum_z.shape}")
    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * target**2.0, dims)
    else:
        denom = torch.sum(mask)

    return torch.mean(num / denom), torch.mean(torch.abs(continuity)), torch.mean(torch.abs(momentum_x)), torch.mean(torch.abs(momentum_y)), torch.mean(torch.abs(momentum_z))


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Combined scalar and vector loss as a scalar tensor
    """
    # Separate the scalar and vector components:
    output_scalar, output_vector = torch.split(output, [1, 3], dim=2)
    target_scalar, target_vector = torch.split(target, [1, 3], dim=2)

    numerator = torch.mean((output_scalar - target_scalar) ** 2.0)
    vector_diff_sq = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "mse":
        masked_loss_pres = numerator
        masked_loss_ws = torch.sum(vector_diff_sq)
    else:
        denom = torch.mean((target_scalar) ** 2.0)
        masked_loss_pres = numerator / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        masked_loss_ws_num = vector_diff_sq
        masked_loss_ws_denom = torch.mean((target_vector) ** 2.0, (0, 1))
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_ws

    return loss / 4.0


def loss_fn_area(
    output: torch.Tensor,
    target: torch.Tensor,
    normals: torch.Tensor,
    area: torch.Tensor,
    area_scaling_factor: float,
    loss_type: Literal["mse", "rmse"],
) -> torch.Tensor:
    """Calculate area-weighted loss for surface data considering normal vectors.

    Args:
        output: Predicted surface values from the model
        target: Ground truth surface values
        normals: Normal vectors for the surface
        area: Area values for surface elements
        area_scaling_factor: Scaling factor for area weighting
        loss_type: Type of loss to calculate ("mse" or "rmse")

    Returns:
        Area-weighted loss as a scalar tensor
    """
    area = area * area_scaling_factor
    area_scale_factor = area

    # Separate the scalar and vector components.
    target_scalar, target_vector = torch.split(
        target * area_scale_factor, [1, 3], dim=2
    )
    output_scalar, output_vector = torch.split(
        output * area_scale_factor, [1, 3], dim=2
    )

    # Apply the normals to the scalar components (only [:,:,0]):
    normals, _ = torch.split(normals, [1, normals.shape[-1] - 1], dim=2)
    target_scalar = target_scalar * normals
    output_scalar = output_scalar * normals

    # Compute the mean diff**2 of the scalar component:
    masked_loss_pres = torch.mean(((output_scalar - target_scalar) ** 2.0), dim=(0, 1))
    if loss_type == "rmse":
        masked_loss_pres /= torch.mean(target_scalar**2.0, dim=(0, 1))

    # Compute the mean diff**2 of the vector component, leave the last dimension:
    masked_loss_ws = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))

    if loss_type == "rmse":
        masked_loss_ws /= torch.mean((target_vector) ** 2.0, (0, 1))

    # Combine the scalar and vector components:
    loss = 0.25 * (masked_loss_pres + torch.sum(masked_loss_ws))

    return loss


def integral_loss_fn(
    output, target, area, normals, stream_velocity=None, padded_value=-10
):
    drag_loss = drag_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    lift_loss = lift_loss_fn(
        output, target, area, normals, stream_velocity, padded_value=-10
    )
    return lift_loss + drag_loss


def lift_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3

    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    normals = torch.select(normals, 2, 2)
    # output_true_0 = output_true[:, :, 0]
    output_true_0 = output_true.select(2, 0)
    output_pred_0 = output_pred.select(2, 0)

    pres_true = output_true_0 * normals
    pres_pred = output_pred_0 * normals

    wz_true = output_true[:, :, -1]
    wz_pred = output_pred[:, :, -1]

    masked_pred = torch.mean(pres_pred + wz_pred, (1))
    masked_truth = torch.mean(pres_true + wz_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def drag_loss_fn(output, target, area, normals, stream_velocity=None, padded_value=-10):
    vel_inlet = stream_velocity  # Get this from the dataset
    mask = abs(target - padded_value) > 1e-3
    output_true = target * mask * area * (vel_inlet) ** 2.0
    output_pred = output * mask * area * (vel_inlet) ** 2.0

    pres_true = output_true[:, :, 0] * normals[:, :, 0]
    pres_pred = output_pred[:, :, 0] * normals[:, :, 0]

    wx_true = output_true[:, :, 1]
    wx_pred = output_pred[:, :, 1]

    masked_pred = torch.mean(pres_pred + wx_pred, (1))
    masked_truth = torch.mean(pres_true + wx_true, (1))

    loss = (masked_pred - masked_truth) ** 2.0
    loss = torch.mean(loss)
    return loss


def compute_loss_dict(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    integral_scaling_factor: float,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
    vol_factors: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the loss terms in a single function call.

    Computes:
    - Volume loss if prediction_vol is not None
    - Surface loss if prediction_surf is not None
    - Integral loss if prediction_surf is not None
    - Total loss as a weighted sum of the above

    Returns:
    - Total loss as a scalar tensor
    - Dictionary of loss terms (for logging, etc)
    """
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]

        loss_vol = loss_fn(
            prediction_vol, target_vol, loss_fn_type.loss_type, padded_value=-10, vol_factors=vol_factors
        )
        loss_dict["loss_vol"] = loss_vol[0]
        loss_dict["loss_continuity"] = loss_vol[1]
        loss_dict["loss_momentum_x"] = loss_vol[2]
        loss_dict["loss_momentum_y"] = loss_vol[3]
        loss_dict["loss_momentum_z"] = loss_vol[4]
        total_loss_terms.append(loss_vol[0])
        total_loss_terms.append(loss_vol[1])
        total_loss_terms.append(loss_vol[2])
        total_loss_terms.append(loss_vol[3])
        total_loss_terms.append(loss_vol[4])
        
    if prediction_surf is not None:

        target_surf = batch_inputs["surface_fields"]
        surface_areas = batch_inputs["surface_areas"]
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_normals = batch_inputs["surface_normals"]
        stream_velocity = batch_inputs["stream_velocity"]
        loss_surf = loss_fn_surface(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        loss_surf_area = loss_fn_area(
            prediction_surf,
            target_surf,
            surface_normals,
            surface_areas,
            area_scaling_factor=loss_fn_type.area_weighing_factor,
            loss_type=loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling
            loss_surf_area = loss_surf_area * surf_loss_scaling

        total_loss_terms.append(0.5 * loss_surf)
        loss_dict["loss_surf"] = 0.5 * loss_surf
        total_loss_terms.append(0.5 * loss_surf_area)
        loss_dict["loss_surf_area"] = 0.5 * loss_surf_area
        loss_integral = (
            integral_loss_fn(
                prediction_surf,
                target_surf,
                surface_areas,
                surface_normals,
                stream_velocity,
                padded_value=-10,
            )
        ) * integral_scaling_factor
        loss_dict["loss_integral"] = loss_integral
        total_loss_terms.append(loss_integral)

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict


def validation_step(
    dataloader,
    model,
    device,
    logger,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    vol_factors=None
):
    running_vloss = 0.0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            with autocast(enabled=False):

                prediction_vol, prediction_surf = model(sampled_batched, compute_gradients=True)
                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    vol_factors,
                )
            
            # print(loss_dict.keys())
            running_vloss += loss.item()

    avg_vloss = running_vloss / (i_batch + 1)

    return avg_vloss


@profile
def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    integral_scaling_factor,
    loss_fn_type,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    vol_factors=None
):

    dist = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    for i_batch, sample_batched in enumerate(dataloader):

        sampled_batched = dict_to_device(sample_batched, device)

        with autocast(enabled=False):   # TODO: to support LSTSQ
            with nvtx.range("Model Forward Pass"):
                prediction_vol, prediction_surf = model(sampled_batched, compute_gradients=True)

            loss, loss_dict = compute_loss_dict(
                prediction_vol,
                prediction_surf,
                sampled_batched,
                loss_fn_type,
                integral_scaling_factor,
                surf_loss_scaling,
                vol_loss_scaling,
                vol_factors
            )

        loss = loss / loss_interval
        scaler.scale(loss).backward()

        if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Gather data and report
        running_loss += loss.item()
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        # Format the loss dict into a string:
        loss_string = (
            "  "
            + "\t".join([f"{key.replace('loss_',''):<10}" for key in loss_dict.keys()])
            + "\n"
        )
        loss_string += (
            "  " + f"\t".join([f"{l.item():<10.3e}" for l in loss_dict.values()]) + "\n"
        )

        logging_string += loss_string
        # for key, value in loss_dict.items():
        #     logging_string += f"    {key}: {value.item():.5f}\n"
        logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta:.3f} Gb\n"
        logging_string += f"  Time taken: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    if dist.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize NVML
    nvmlInit()

    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    compute_scaling_factors(
        cfg, cfg.data_processor.output_dir, use_cache=cfg.data_processor.use_cache
    )
    model_type = cfg.model.model_type

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
        vol_factors_tensor = torch.from_numpy(vol_factors).to(dist.device)
    else:
        vol_factors = None

    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
    else:
        surf_factors = None

    train_dataset = create_domino_dataset(
        cfg,
        phase="train",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )
    val_dataset = create_domino_dataset(
        cfg,
        phase="val",
        volume_variable_names=volume_variable_names,
        surface_variable_names=surface_variable_names,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.train.sampler,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.val.sampler,
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **cfg.train.dataloader,
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **cfg.val.dataloader,
    )

    model = DoMINOPhy(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        model_parameters=cfg.model,
    ).to(dist.device)
    model = torch.compile(model, disable=True)  # TODO make this configurable

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    # optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300, 400, 500, 600, 700, 800], gamma=0.5
    )

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    epoch_number = 0

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")
    if dist.rank == 0:
        create_directory(model_save_path)
        create_directory(param_save_path)
        create_directory(best_model_path)

    if dist.world_size > 1:
        torch.distributed.barrier()

    init_epoch = load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=dist.device,
    )

    if init_epoch != 0:
        init_epoch += 1  # Start with the next epoch
    epoch_number = init_epoch

    # retrive the smallest validation loss if available
    numbers = []
    for filename in os.listdir(best_model_path):
        match = re.search(r"\d+\.\d*[1-9]\d*", filename)
        if match:
            number = float(match.group(0))
            numbers.append(number)

    best_vloss = min(numbers) if numbers else 1_000_000.0

    initial_integral_factor_orig = cfg.model.integral_loss_scaling_factor

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch_number}:")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        initial_integral_factor = initial_integral_factor_orig

        if epoch > 250:
            surface_scaling_loss = 1.0 * cfg.model.surf_loss_scaling
        else:
            surface_scaling_loss = cfg.model.surf_loss_scaling

        model.train(True)
        epoch_start_time = time.perf_counter()
        avg_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            tb_writer=writer,
            logger=logger,
            gpu_handle=gpu_handle,
            epoch_index=epoch,
            device=dist.device,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            vol_factors=vol_factors_tensor
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )
        epoch_end_time = time.perf_counter()

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=dist.device,
            logger=logger,
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            vol_factors=vol_factors_tensor
        )

        scheduler.step()
        logger.info(
            f"Device {dist.device} "
            f"LOSS train {avg_loss:.5f} "
            f"valid {avg_vloss:.5f} "
            f"Current lr {scheduler.get_last_lr()[0]} "
            f"Integral factor {initial_integral_factor}"
        )

        if dist.rank == 0:
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number,
            )
            writer.flush()

        # Track best performance, and save the model's state
        if dist.world_size > 1:
            torch.distributed.barrier()

        if avg_vloss < best_vloss:  # This only considers GPU: 0, is that okay?
            best_vloss = avg_vloss

        if dist.rank == 0:
            print(f"Device { dist.device}, Best val loss {best_vloss}")

        if dist.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0.0:
            save_checkpoint(
                to_absolute_path(model_save_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
            )

        epoch_number += 1

        if scheduler.get_last_lr()[0] == 1e-6:
            print("Training ended")
            exit()


if __name__ == "__main__":
    main()
