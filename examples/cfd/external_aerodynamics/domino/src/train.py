# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Literal, Any
from tabulate import tabulate

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# This will set up the cupy-ecosystem and pytorch to share memory pools
from physicsnemo.utils.memory import unified_gpu_memory

import torchinfo
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_module

from torch.amp import GradScaler, autocast
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
    create_domino_dataset,
)
from physicsnemo.datapipes.cae.cae_dataset import CAEDataset
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *

from simple_datapipe import SimpleDoMINODataPipe
from utils import ScalingFactors, get_keys_to_read, coordinate_distributed_environment

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler


from loss import compute_loss_dict
from utils import get_num_vars, load_scaling_factors, compute_l2, all_reduce_dict


def compute_physics_loss_weight_curriculum(
    epoch: int,
    base_weight: float,
    curriculum_enabled: bool,
    warmup_epochs: int,
    rampup_epochs: int,
) -> float:
    """
    Compute the physics loss weight based on curriculum learning schedule.
    
    Schedule:
    - Epochs 0 to warmup_epochs: weight = 0
    - Epochs warmup_epochs to warmup_epochs+rampup_epochs: linear ramp from 0 to base_weight
    - Epochs > warmup_epochs+rampup_epochs: weight = base_weight
    
    Args:
        epoch: Current epoch number (0-indexed)
        base_weight: Final target weight for physics loss
        curriculum_enabled: Whether curriculum learning is enabled
        warmup_epochs: Number of epochs with zero physics loss weight
        rampup_epochs: Number of epochs to ramp up from 0 to base_weight
    
    Returns:
        Current physics loss weight for this epoch
    """
    if not curriculum_enabled:
        return base_weight
    
    if epoch < warmup_epochs:
        # Warmup phase: no physics loss
        return 0.0
    elif epoch < warmup_epochs + rampup_epochs:
        # Rampup phase: linear increase from 0 to base_weight
        progress = (epoch - warmup_epochs) / rampup_epochs
        return base_weight * progress
    else:
        # Full weight phase
        return base_weight


def validation_step(
    dataloader,
    model,
    device,
    logger,
    tb_writer,
    epoch_index,
    use_sdf_basis=False,
    use_surface_normals=False,
    integral_scaling_factor=1.0,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    add_physics_loss=False,
    log_physics_loss=False,
    autocast_enabled=None,
    physics_loss_weight=1.0,
):
    dm = DistributedManager()
    running_vloss = 0.0
    with torch.no_grad():
        metrics = None
        accumulated_losses = {}  # To accumulate loss_dict across batches

        for i_batch, sample_batched in enumerate(dataloader):
            sampled_batched = dict_to_device(sample_batched, device)

            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                # Compute two-pass forward if physics loss needed (for training or logging)
                if add_physics_loss or log_physics_loss:
                    # NEW: Two-pass approach for FVM physics loss
                    # Pass 1: Main cell centers
                    prediction_vol, prediction_surf = model(sampled_batched)
                    
                    # Pass 2: Neighbor cell centers
                    # dataloader is a DataLoader wrapper, access the SimpleDoMINODataPipe via .dataset
                    datapipe = dataloader.dataset
                    neighbor_centers, neighbor_mask = datapipe.get_neighbor_cell_centers(sampled_batched)
                    n_cells, max_nb = neighbor_centers.shape[:2]
                    
                    # Process neighbors
                    neighbor_centers_torch = torch.from_numpy(neighbor_centers).to(device)
                    prediction_vol_neighbors = torch.zeros(
                        1, n_cells, max_nb, prediction_vol.shape[-1], device=device
                    )
                    
                    for nb_idx in range(max_nb):
                        neighbor_batch = neighbor_centers_torch[:, nb_idx, :].unsqueeze(0)
                        input_dict_nb = {k: v for k, v in sampled_batched.items()}
                        input_dict_nb['volume_mesh_centers'] = neighbor_batch
                        
                        solutions_nb_vol, _ = model(input_dict_nb)
                        prediction_vol_neighbors[0, :, nb_idx, :] = solutions_nb_vol.squeeze(0)
                else:
                    prediction_vol, prediction_surf = model(sampled_batched)
                    prediction_vol_neighbors = None

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    add_physics_loss,
                    log_physics_loss_only=(log_physics_loss and not add_physics_loss),
                    # FVM Physics Loss Parameters
                    prediction_vol_neighbors=prediction_vol_neighbors,
                    datapipe=datapipe,  # Always pass datapipe for consistent unnormalized data loss
                    physics_loss_weight=physics_loss_weight,
                )

            running_vloss += loss.item()
            
            # Accumulate loss_dict values (including physics losses if computed)
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                if key not in accumulated_losses:
                    accumulated_losses[key] = 0.0
                accumulated_losses[key] += value
            
            local_metrics = compute_l2(
                prediction_surf, prediction_vol, sampled_batched, dataloader
            )
            if metrics is None:
                metrics = local_metrics
            else:
                metrics = {
                    key: metrics[key] + local_metrics[key] for key in metrics.keys()
                }

    avg_vloss = running_vloss / (i_batch + 1)
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}
    
    # Average the accumulated losses and convert back to tensors for all_reduce
    avg_losses = {
        key: torch.tensor(value / (i_batch + 1), device=device) 
        for key, value in accumulated_losses.items()
    }

    metrics = all_reduce_dict(metrics, dm)
    avg_losses = all_reduce_dict(avg_losses, dm)
    
    # Convert avg_losses back to Python floats for logging
    avg_losses = {key: value.item() if isinstance(value, torch.Tensor) else value 
                  for key, value in avg_losses.items()}

    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, VAL loss norm: {loss.detach().item():.5f}"
        )
        tb_x = epoch_index
        
        # Log L2 metrics to tensorboard
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/val/{key}", metrics[key], tb_x)
        
        # Log all losses (including physics losses) to tensorboard
        for key, value in avg_losses.items():
            tb_writer.add_scalar(f"Loss/val/{key}", value, tb_x)

        # Print L2 metrics table
        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(
            f"\nEpoch {epoch_index} VALIDATION Average Metrics:\n{metrics_table}\n"
        )
        
        # Print loss components table (including physics losses if present)
        if avg_losses:
            losses_table = tabulate(
                [[k, v] for k, v in avg_losses.items()],
                headers=["Loss Component", "Average Value"],
                tablefmt="pretty",
            )
            logger.info(
                f"\nEpoch {epoch_index} VALIDATION Loss Components:\n{losses_table}\n"
            )

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
    add_physics_loss=False,
    autocast_enabled=None,
    grad_clip_enabled=None,
    grad_max_norm=None,
    physics_loss_weight=1.0,
):
    dm = DistributedManager()

    running_loss = 0.0
    last_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    with Profiler():
        io_start_time = time.perf_counter()
        metrics = None
        for i_batch, sampled_batched in enumerate(dataloader):
            io_end_time = time.perf_counter()
            # Note: Autocast is now compatible with FVM physics loss
            # The model forward passes use mixed precision for memory efficiency,
            # but tensors are converted to float32 before FVM computation in loss.py

            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                with nvtx.range("Model Forward Pass"):
                    if add_physics_loss:
                        # NEW: Two-pass approach for FVM physics loss
                        # Pass 1: Main cell centers
                        prediction_vol, prediction_surf = model(sampled_batched)
                        
                        # Pass 2: Neighbor cell centers
                        # dataloader is a DataLoader wrapper, access the SimpleDoMINODataPipe via .dataset
                        datapipe = dataloader.dataset
                        neighbor_centers, neighbor_mask = datapipe.get_neighbor_cell_centers(sampled_batched)
                        n_cells, max_nb = neighbor_centers.shape[:2]
                        
                        # Process neighbors
                        neighbor_centers_torch = torch.from_numpy(neighbor_centers).to(device)
                        prediction_vol_neighbors = torch.zeros(
                            1, n_cells, max_nb, prediction_vol.shape[-1], device=device
                        )
                        
                        for nb_idx in range(max_nb):
                            neighbor_batch = neighbor_centers_torch[:, nb_idx, :].unsqueeze(0)
                            input_dict_nb = {k: v for k, v in sampled_batched.items()}
                            input_dict_nb['volume_mesh_centers'] = neighbor_batch
                            
                            solutions_nb_vol, _ = model(input_dict_nb)
                            prediction_vol_neighbors[0, :, nb_idx, :] = solutions_nb_vol.squeeze(0)
                    else:
                        prediction_vol, prediction_surf = model(sampled_batched)
                        prediction_vol_neighbors = None

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    integral_scaling_factor,
                    surf_loss_scaling,
                    vol_loss_scaling,
                    add_physics_loss,
                    # FVM Physics Loss Parameters
                    prediction_vol_neighbors=prediction_vol_neighbors,
                    datapipe=datapipe,  # Always pass datapipe for consistent unnormalized data loss
                    physics_loss_weight=physics_loss_weight,
                )

                # Compute metrics:
                if isinstance(prediction_vol, tuple):
                    # This is if return_neighbors is on for volume:
                    prediction_vol = prediction_vol[0]

                local_metrics = compute_l2(
                    prediction_surf, prediction_vol, sampled_batched, dataloader
                )
                if metrics is None:
                    metrics = local_metrics
                else:
                    # Sum the running total:
                    metrics = {
                        key: metrics[key] + local_metrics[key] for key in metrics.keys()
                    }

            loss = loss / loss_interval
            scaler.scale(loss).backward()

            if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
                if grad_clip_enabled:
                    # Unscales the gradients of optimizer's assigned params in-place.
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Gather data and report
            running_loss += loss.detach().item()
            elapsed_time = time.perf_counter() - start_time
            io_time = io_end_time - io_start_time
            start_time = time.perf_counter()
            gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used = gpu_end_info.used / (1024**3)
            gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

            logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
            # Format the loss dict into a string:
            loss_string = (
                "  "
                + "\t".join(
                    [f"{key.replace('loss_', ''):<10}" for key in loss_dict.keys()]
                )
                + "\n"
            )
            loss_string += (
                "  "
                + f"\t".join(
                    [f"{l.detach().item():<10.3e}" for l in loss_dict.values()]
                )
                + "\n"
            )

            logging_string += loss_string
            logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb (delta: {gpu_memory_delta:.3f})\n"
            logging_string += f"  Timings: (IO: {io_time:.2f}, Model: {elapsed_time - io_time:.2f}, Total: {elapsed_time:.2f})s\n"
            logger.info(logging_string)
            
            # Log individual loss components to tensorboard (including physics losses if present)
            if dm.rank == 0:
                tb_x_batch = epoch_index * len(dataloader) + i_batch + 1
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    tb_writer.add_scalar(f"Loss/train/{key}", value, tb_x_batch)
            
            gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            io_start_time = time.perf_counter()

    last_loss = running_loss / (i_batch + 1)  # loss per batch
    # Normalize metrics:
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}
    # reduce metrics across batch:
    metrics = all_reduce_dict(metrics, dm)
    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.detach().item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/train/{key}", metrics[key], epoch_index)

        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch_index} Average Metrics:\n{metrics_table}\n")

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ######################################################
    # initialize distributed manager
    ######################################################
    DistributedManager.initialize()
    dist = DistributedManager()

    # DoMINO supports domain parallel training.  This function helps coordinate
    # how to set that up, if needed.
    domain_mesh, data_mesh, placements = coordinate_distributed_environment(cfg)

    if data_mesh is not None:
        data_replica_size = data_mesh.size()
        data_rank = data_mesh.get_local_rank()
    else:
        data_replica_size = dist.world_size
        data_rank = dist.rank

    ################################
    # Initialize NVML
    ################################
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    ######################################################
    # Initialize logger
    ######################################################

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ######################################################
    # Get scaling factors - precompute them if this fails!
    ######################################################
    vol_factors, surf_factors = load_scaling_factors(cfg)

    ######################################################
    # Configure the model
    ######################################################
    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    if model_type == "combined" or model_type == "surface":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
    else:
        surface_variable_names = []

    if model_type == "combined" or model_type == "volume":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
    else:
        volume_variable_names = []

    ######################################################
    # Configure physics loss
    ######################################################
    add_physics_loss = getattr(cfg.train, "add_physics_loss", False)
    log_physics_loss = getattr(cfg.train, "log_physics_loss", False)
    physics_loss_weight_base = getattr(cfg.train, "physics_loss_weight", 1.0)
    
    # Curriculum learning parameters
    curriculum_cfg = getattr(cfg.train, "physics_loss_curriculum", None)
    if curriculum_cfg is not None:
        curriculum_enabled = getattr(curriculum_cfg, "enabled", False)
        warmup_epochs = getattr(curriculum_cfg, "warmup_epochs", 50)
        rampup_epochs = getattr(curriculum_cfg, "rampup_epochs", 50)
    else:
        curriculum_enabled = False
        warmup_epochs = 0
        rampup_epochs = 0

    ######################################################
    # Configure the dataset
    ######################################################

    # This helper function is to determine which keys to read from the data
    # (and which to use default values for, if they aren't present - like
    # air_density, for example)
    keys_to_read, keys_to_read_if_available = get_keys_to_read(
        cfg, model_type, get_ground_truth=True
    )

    # The dataset actually works in two pieces
    # The core dataset just reads data from disk, and puts it on the GPU if needed.
    # The data processesing pipeline will preprocess that data and prepare it for the model.
    # Obviously, you need both, so this function will return the datapipeline in
    # a way that can be iterated over.
    #
    # To properly shuffle the data, we use a distributed sampler too.
    # It's configured properly for optional domain parallelism, and you have
    # to make sure to call set_epoch below.

    # Create datasets using SimpleDoMINODataPipe for FVM physics loss support
    device = dist.device if cfg.data.gpu_preprocessing else "cpu"
    
    # Training dataset
    train_dataset = CAEDataset(
        data_dir=cfg.data.input_dir,
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        output_device=device,
        preload_depth=cfg.train.dataloader.preload_depth,
        pin_memory=cfg.train.dataloader.pin_memory,
    )
    
    train_dataloader = SimpleDoMINODataPipe(
        data_path=cfg.data.input_dir,
        phase="train",
        model_type=model_type,
        grid_resolution=cfg.model.interp_res,
        bounding_box_volume=(cfg.data.bounding_box.min, cfg.data.bounding_box.max),
        bounding_box_surface=(cfg.data.bounding_box_surface.min, cfg.data.bounding_box_surface.max),
        sampling=cfg.data.sampling,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        normalize_coordinates=cfg.data.normalize_coordinates,
        scaling_type=cfg.model.normalization,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        gpu_preprocessing=cfg.data.gpu_preprocessing,
        gpu_output=cfg.data.gpu_output,
    )
    train_dataloader.set_dataset(train_dataset)
    
    train_sampler = DistributedSampler(
        train_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.train.sampler,
    )
    
    # Wrap in DataLoader with DistributedSampler for proper DDP
    # batch_size=1 because SimpleDoMINODataPipe already returns batched data
    train_dataloader_wrapper = DataLoader(
        train_dataloader,
        batch_size=1,
        sampler=train_sampler,
        num_workers=0,  # Data is already on GPU from datapipe
        collate_fn=lambda x: x[0],  # Just extract the single batch, don't collate
        pin_memory=False,  # Already handled by datapipe
    )

    # Validation dataset
    val_dataset = CAEDataset(
        data_dir=cfg.data.input_dir,
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        output_device=device,
        preload_depth=cfg.val.dataloader.preload_depth,
        pin_memory=cfg.val.dataloader.pin_memory,
    )
    
    val_dataloader = SimpleDoMINODataPipe(
        data_path=cfg.data.input_dir,
        phase="val",
        model_type=model_type,
        grid_resolution=cfg.model.interp_res,
        bounding_box_volume=(cfg.data.bounding_box.min, cfg.data.bounding_box.max),
        bounding_box_surface=(cfg.data.bounding_box_surface.min, cfg.data.bounding_box_surface.max),
        sampling=cfg.data.sampling,
        volume_points_sample=cfg.model.volume_points_sample,
        surface_points_sample=cfg.model.surface_points_sample,
        geom_points_sample=cfg.model.geom_points_sample,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        normalize_coordinates=cfg.data.normalize_coordinates,
        scaling_type=cfg.model.normalization,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        gpu_preprocessing=cfg.data.gpu_preprocessing,
        gpu_output=cfg.data.gpu_output,
    )
    val_dataloader.set_dataset(val_dataset)
    
    val_sampler = DistributedSampler(
        val_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.val.sampler,
    )
    
    # Wrap in DataLoader with DistributedSampler for proper DDP
    val_dataloader_wrapper = DataLoader(
        val_dataloader,
        batch_size=1,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=lambda x: x[0],
        pin_memory=False,
    )

    ######################################################
    # Configure the model
    ######################################################
    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist.world_size > 1:
        if domain_mesh is None:
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            model = distribute_module(
                model,
                device_mesh=domain_mesh,
            )
            model = fully_shard(model, mesh=data_mesh)

    ######################################################
    # Initialize optimzer and gradient scaler
    ######################################################

    optimizer_class = None
    if cfg.train.optimizer.name == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg.train.optimizer.name == "AdamW":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer.name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    if cfg.train.lr_scheduler.name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train.lr_scheduler.milestones,
            gamma=cfg.train.lr_scheduler.gamma,
        )
    elif cfg.train.lr_scheduler.name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.lr_scheduler.T_max,
            eta_min=cfg.train.lr_scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.train.lr_scheduler.name}")

    # Initialize the scaler for mixed precision
    scaler = GradScaler()

    ######################################################
    # Initialize output tools
    ######################################################

    # Tensorboard Writer to track training.
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

    ######################################################
    # Load checkpoint if available
    ######################################################
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

    ######################################################
    # Begin Training loop over epochs
    ######################################################

    for epoch in range(init_epoch, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch_number}:")

        # Compute current physics loss weight based on curriculum
        current_physics_loss_weight = compute_physics_loss_weight_curriculum(
            epoch=epoch,
            base_weight=physics_loss_weight_base,
            curriculum_enabled=curriculum_enabled,
            warmup_epochs=warmup_epochs,
            rampup_epochs=rampup_epochs,
        )
        
        if epoch == init_epoch and add_physics_loss:
            logger.info(
                "Physics loss enabled - FVM residual computation uses PyTorch-Warp interop with automatic float32 conversion"
            )
            if curriculum_enabled:
                logger.info(
                    f"Physics loss curriculum enabled: warmup={warmup_epochs} epochs, rampup={rampup_epochs} epochs, final_weight={physics_loss_weight_base}"
                )
        
        if epoch == init_epoch and log_physics_loss and not add_physics_loss:
            logger.info(
                "Physics loss logging enabled - FVM residuals will be computed during validation for monitoring (not used in training)"
            )
        
        # Log current physics loss weight
        if add_physics_loss and dist.rank == 0:
            logger.info(f"Epoch {epoch}: Physics loss weight = {current_physics_loss_weight:.6f}")
            writer.add_scalar("Hyperparameters/physics_loss_weight", current_physics_loss_weight, epoch)

        # This controls what indices to use for each epoch.
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # Note: SimpleDoMINODataPipe wraps CAEDataset, so we access it via .dataset
        # With DistributedSampler in DataLoader, indices are automatically handled
        # if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, 'set_indices'):
        #     train_dataloader.dataset.set_indices(list(train_sampler))
        #     val_dataloader.dataset.set_indices(list(val_sampler))

        initial_integral_factor = initial_integral_factor_orig

        if epoch > 250:
            surface_scaling_loss = 1.0 * cfg.model.surf_loss_scaling
        else:
            surface_scaling_loss = cfg.model.surf_loss_scaling

        model.train(True)
        epoch_start_time = time.perf_counter()
        avg_loss = train_epoch(
            dataloader=train_dataloader_wrapper,
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
            add_physics_loss=add_physics_loss,
            autocast_enabled=cfg.train.amp.enabled,
            grad_clip_enabled=cfg.train.amp.clip_grad,
            grad_max_norm=cfg.train.amp.grad_max_norm,
            physics_loss_weight=current_physics_loss_weight,  # Use curriculum weight for training
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )
        epoch_end_time = time.perf_counter()

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader_wrapper,
            model=model,
            device=dist.device,
            logger=logger,
            tb_writer=writer,
            epoch_index=epoch,
            use_sdf_basis=cfg.model.use_sdf_in_basis_func,
            use_surface_normals=cfg.model.use_surface_normals,
            integral_scaling_factor=initial_integral_factor,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            add_physics_loss=add_physics_loss,
            log_physics_loss=log_physics_loss,
            autocast_enabled=cfg.train.amp.enabled,
            physics_loss_weight=1.0,  # Always use 1.0 for validation to ensure consistent comparison
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
            print(f"Device {dist.device}, Best val loss {best_vloss}")

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
