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

import torch
import numpy as np
from typing import Literal, Any

from physicsnemo.utils.domino.utils import unnormalize

import torch.cuda.nvtx as nvtx

from physicsnemo.utils.domino.utils import *
from fvm_residuals_warp import compute_residuals_warp_cell_centered, compute_residuals_warp_cell_centered_torch


def compute_fvm_physics_loss(
    solutions_main: torch.Tensor,
    solutions_neighbors: torch.Tensor,
    batch: dict,
    datapipe,
    nu: float = 1.5881327800829875e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute FVM-based physics loss by comparing predicted vs ground truth residuals.
    
    This function computes physics loss as the difference between:
    1. FVM residuals computed on MODEL PREDICTIONS (maintains gradients!)
    2. FVM residuals computed on GROUND TRUTH data
    
    IMPORTANT: Unnormalizes fields to restore non-dimensional values from preprocessed data.
    The viscosity is non-dimensionalized using the same scheme as curator:
        nu* = nu / (U_ref * L_ref)
    where U_ref = 30.0 m/s and L_ref = max dimension of vehicle STL.
    FVM residuals are computed in non-dimensional space for consistency.
    
    Uses Warp's PyTorch interop to maintain autodifferentiability throughout.
    Gradients can flow back through the FVM computation to the model.
    
    Args:
        solutions_main: Model predictions at main cell centers [batch, n_cells, 5]
                       where 5 = [u, v, w, p, nut] (in NORMALIZED space)
        solutions_neighbors: Model predictions at neighbor cell centers 
                            [batch, n_cells, max_neighbors, 5] (in NORMALIZED space)
        batch: Batch dictionary with 'volume_cell_indices' and 'volume_fields' keys
        datapipe: Datapipe with mesh connectivity and normalization parameters
        nu: Kinematic viscosity (m^2/s) in dimensional physical units
    
    Returns:
        Tuple of (continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss)
        Each loss is MSE between predicted and ground truth residuals.
        Gradients are preserved for backpropagation!
    """
    from physicsnemo.utils.domino.utils import unnormalize, unstandardize
    
    device = solutions_main.device
    
    # ========================================
    # STEP -1: Non-dimensionalize viscosity to match preprocessed data
    # ========================================
    # Curator applies: ν* = ν / (U_ref * L_ref)
    # where L_ref = max dimension of STL bounding box
    U_ref = 30.0  # m/s (STREAM_VELOCITY from curator constants)
    
    # Get L_ref from batch (computed in datapipe same way as curator)
    if 'length_scale' not in batch:
        raise ValueError(
            "Cannot compute physics loss: 'length_scale' not found in batch. "
            "Ensure datapipe computes this in process_data()."
        )
    
    L_ref = batch['length_scale']
    if isinstance(L_ref, torch.Tensor):
        L_ref = L_ref.item()
    
    # Non-dimensionalize viscosity
    nu_nondim = nu / (U_ref * L_ref)
    
    # Keep as torch tensors (don't detach - we want gradients!)
    # Explicitly convert to float32 (in case model output is float16 from autocast)
    solutions_main_squeezed = solutions_main.squeeze(0).float()  # [n_cells, 5]
    solutions_neighbors_squeezed = solutions_neighbors.squeeze(0).float()  # [n_cells, max_nb, 5]
    
    # ========================================
    # STEP 0: UNNORMALIZE model predictions to non-dimensional space
    # ========================================
    # Model outputs are statistically normalized (min-max or mean-std).
    # Unnormalizing restores the non-dimensional fields as stored in preprocessed data.
    
    if datapipe.volume_factors is not None:
        # Unscale fields based on scaling type
        if datapipe.scaling_type == "mean_std_scaling":
            # Unstandardize: x_physical = x_normalized * std + mean
            solutions_main_squeezed = unstandardize(
                solutions_main_squeezed,
                datapipe.volume_factors[0].to(device),  # mean
                datapipe.volume_factors[1].to(device),  # std
            )
            solutions_neighbors_squeezed = unstandardize(
                solutions_neighbors_squeezed,
                datapipe.volume_factors[0].to(device),
                datapipe.volume_factors[1].to(device),
            )
        elif datapipe.scaling_type == "min_max_scaling":
            # Unnormalize: x_physical = x_normalized * (max - min) + min
            solutions_main_squeezed = unnormalize(
                solutions_main_squeezed,
                datapipe.volume_factors[0].to(device),  # min
                datapipe.volume_factors[1].to(device),  # max
            )
            solutions_neighbors_squeezed = unnormalize(
                solutions_neighbors_squeezed,
                datapipe.volume_factors[0].to(device),
                datapipe.volume_factors[1].to(device),
            )
    
    # Get cell indices
    cell_indices_tensor = batch['volume_cell_indices']
    if isinstance(cell_indices_tensor, torch.Tensor):
        cell_indices_np = cell_indices_tensor.cpu().numpy()
    else:
        cell_indices_np = cell_indices_tensor
    if cell_indices_np.ndim > 1:
        cell_indices_np = cell_indices_np.squeeze(0)
    
    n_total_cells = len(datapipe.mesh_connectivity['cell_centers'])
    
    # ========================================
    # STEP 1: Compute residuals on PREDICTIONS (maintains gradients!)
    # ========================================
    # NOTE: solutions are now in NON-DIMENSIONAL units (after unnormalization)
    # Geometry will also be non-dimensionalized to match
    
    # Build field data dict with torch tensors
    velocity_pred = torch.zeros((n_total_cells, 3), dtype=torch.float32, device=device)
    pressure_pred = torch.zeros(n_total_cells, dtype=torch.float32, device=device)
    nut_pred = torch.zeros(n_total_cells, dtype=torch.float32, device=device)
    
    # Fill in main cells with model predictions
    for i, cell_idx in enumerate(cell_indices_np):
        velocity_pred[cell_idx] = solutions_main_squeezed[i, :3]
        pressure_pred[cell_idx] = solutions_main_squeezed[i, 3]
        nut_pred[cell_idx] = solutions_main_squeezed[i, 4]
    
    # Fill in neighbor cells with model predictions
    # NOTE: neighbors are capped at max_neighbors (default 12) in get_neighbor_cell_centers
    max_neighbors_in_pred = solutions_neighbors_squeezed.shape[1]  # Get actual neighbor dimension
    
    for i, cell_idx in enumerate(cell_indices_np):
        nb_start = datapipe.mesh_connectivity['neighbors_offsets'][cell_idx]
        nb_end = datapipe.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
        neighbor_ids = datapipe.mesh_connectivity['neighbors_flat'][nb_start:nb_end]
        
        # Cap to match the actual number of neighbors we have predictions for
        neighbor_ids_capped = neighbor_ids[:max_neighbors_in_pred]
        
        for j, nb_id in enumerate(neighbor_ids_capped):
            if nb_id >= 0:  # Valid neighbor
                velocity_pred[nb_id] = solutions_neighbors_squeezed[i, j, :3]
                pressure_pred[nb_id] = solutions_neighbors_squeezed[i, j, 3]
                nut_pred[nb_id] = solutions_neighbors_squeezed[i, j, 4]
    
    # Create mesh data dict with torch tensors (for autodiff!)
    field_data_pred_torch = {
        'velocity': velocity_pred,
        'pressure': pressure_pred,
        'nut': nut_pred,
    }
    
    # Get batched mesh data (will convert to numpy internally, but we'll pass torch tensors)
    batched_mesh_pred = datapipe.get_batched_mesh_data(batch, field_data=field_data_pred_torch)
    
    # Non-dimensionalize geometry to match non-dimensional fields
    batched_mesh_pred['points'] = batched_mesh_pred['points'] / L_ref
    batched_mesh_pred['cell_centers'] = batched_mesh_pred['cell_centers'] / L_ref
    batched_mesh_pred['cell_volumes'] = batched_mesh_pred['cell_volumes'] / (L_ref ** 3)
    
    # Compute FVM residuals using torch-compatible version with non-dimensional viscosity (maintains gradients!)
    continuity_pred_all, momentum_x_pred_all, momentum_y_pred_all, momentum_z_pred_all = \
        compute_residuals_warp_cell_centered_torch(batched_mesh_pred, nu_nondim, device=str(device))
    
    # Extract residuals for sampled cells only
    local_cell_indices = batched_mesh_pred.get('local_cell_indices', torch.arange(len(continuity_pred_all)))
    if isinstance(local_cell_indices, np.ndarray):
        local_cell_indices = torch.from_numpy(local_cell_indices).to(device)
    
    continuity_pred = continuity_pred_all[local_cell_indices]
    momentum_x_pred = momentum_x_pred_all[local_cell_indices]
    momentum_y_pred = momentum_y_pred_all[local_cell_indices]
    momentum_z_pred = momentum_z_pred_all[local_cell_indices]
    
    # ========================================
    # STEP 2: Compute residuals on GROUND TRUTH
    # ========================================
    # Use ground truth data (no field_data parameter = use mesh_connectivity data)
    batched_mesh_gt = datapipe.get_batched_mesh_data(batch, field_data=None)
    
    # Non-dimensionalize geometry to match non-dimensional fields
    batched_mesh_gt['points'] = batched_mesh_gt['points'] / L_ref
    batched_mesh_gt['cell_centers'] = batched_mesh_gt['cell_centers'] / L_ref
    batched_mesh_gt['cell_volumes'] = batched_mesh_gt['cell_volumes'] / (L_ref ** 3)
    
    # Compute FVM residuals on ground truth with non-dimensional viscosity (no gradients needed here)
    continuity_gt_all, momentum_x_gt_all, momentum_y_gt_all, momentum_z_gt_all = \
        compute_residuals_warp_cell_centered(batched_mesh_gt, nu_nondim)
    
    # Extract residuals for sampled cells only
    local_cell_indices_gt = batched_mesh_gt.get('local_cell_indices', np.arange(len(continuity_gt_all)))
    continuity_gt = continuity_gt_all[local_cell_indices_gt]
    momentum_x_gt = momentum_x_gt_all[local_cell_indices_gt]
    momentum_y_gt = momentum_y_gt_all[local_cell_indices_gt]
    momentum_z_gt = momentum_z_gt_all[local_cell_indices_gt]
    
    # Convert GT to torch (no gradients)
    continuity_gt_torch = torch.from_numpy(continuity_gt.astype(np.float32)).to(device)
    momentum_x_gt_torch = torch.from_numpy(momentum_x_gt.astype(np.float32)).to(device)
    momentum_y_gt_torch = torch.from_numpy(momentum_y_gt.astype(np.float32)).to(device)
    momentum_z_gt_torch = torch.from_numpy(momentum_z_gt.astype(np.float32)).to(device)
    
    # ========================================
    # STEP 3: Compute loss as MSE(predicted - ground_truth)
    # ========================================
    # Gradients flow back through these operations!
    continuity_loss = torch.mean((continuity_pred - continuity_gt_torch) ** 2)
    momentum_x_loss = torch.mean((momentum_x_pred - momentum_x_gt_torch) ** 2)
    momentum_y_loss = torch.mean((momentum_y_pred - momentum_y_gt_torch) ** 2)
    momentum_z_loss = torch.mean((momentum_z_pred - momentum_z_gt_torch) ** 2)
    
    return continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss


# =============================================================================
# DEPRECATED: Old gradient-based physics loss functions
# These are kept for reference only. Use compute_fvm_physics_loss() instead!
# =============================================================================

def compute_physics_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    dims: tuple[int, ...] | None,
    bounding_box: torch.Tensor,
    vol_factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DEPRECATED: Use compute_fvm_physics_loss() instead.
    
    Old gradient-based physics loss for Navier-Stokes equations.

    Args:
        output: Model output containing (output, coords_neighbors, output_neighbors, neighbors_list)
        target: Ground truth values
        mask: Mask for valid values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        dims: Dimensions for loss calculation
        first_deriv: First derivative calculator
        eqn: Equations
        bounding_box: Bounding box for normalization
        vol_factors: Volume factors for normalization

    Returns:
        Tuple of (data_loss, continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss)
    """
    # Physics loss enabled
    output, coords_neighbors, output_neighbors, neighbors_list = output
    batch_size = output.shape[1]
    fields, num_neighbors = output_neighbors.shape[3], output_neighbors.shape[2]
    coords_total = coords_neighbors[0, :]
    output_total = output_neighbors[0, :]
    output_total_unnormalized = unnormalize(
        output_total, vol_factors[0], vol_factors[1]
    )
    coords_total_unnormalized = unnormalize(
        coords_total, bounding_box[0], bounding_box[1]
    )

    # compute first order gradients on all the nodes from the neighbors_list
    grad_list = {}
    for parent_id, neighbor_ids in neighbors_list.items():
        neighbor_ids_tensor = torch.tensor(neighbor_ids).to(
            output_total_unnormalized.device
        )
        du = (
            output_total_unnormalized[:, [parent_id]]
            - output_total_unnormalized[:, neighbor_ids_tensor]
        )
        dv = (
            coords_total_unnormalized[:, [parent_id]]
            - coords_total_unnormalized[:, neighbor_ids_tensor]
        )
        grads = first_deriv.forward(
            coords=None, connectivity_tensor=None, y=None, du=du, dv=dv
        )
        grad = torch.cat(grads, dim=1)
        grad_list[parent_id] = grad

    # compute second order gradients on only the center node
    neighbor_ids_tensor = torch.tensor(neighbors_list[0]).to(
        output_total_unnormalized.device
    )
    grad_neighbors_center = torch.stack([v for v in grad_list.values()], dim=1)
    grad_neighbors_center = grad_neighbors_center.reshape(
        batch_size, len(neighbors_list[0]) + 1, -1
    )

    du = grad_neighbors_center[:, [0]] - grad_neighbors_center[:, neighbor_ids_tensor]
    dv = (
        coords_total_unnormalized[:, [0]]
        - coords_total_unnormalized[:, neighbor_ids_tensor]
    )

    # second order gradients
    ggrads_center = first_deriv.forward(
        coords=None, connectivity_tensor=None, y=None, du=du, dv=dv
    )
    ggrad_center = torch.cat(ggrads_center, dim=1)
    grad_neighbors_center = grad_neighbors_center.reshape(
        batch_size, len(neighbors_list[0]) + 1, 3, -1
    )

    # Get the outputs on the original nodes
    fields_center_unnormalized = output_total_unnormalized[:, 0, :]
    grad_center = grad_neighbors_center[:, 0, :, :]
    grad_grad_uvw_center = ggrad_center[:, :, :9]

    nu = 1.507 * 1e-5

    dict_mapping = {
        "u": fields_center_unnormalized[:, [0]],
        "v": fields_center_unnormalized[:, [1]],
        "w": fields_center_unnormalized[:, [2]],
        "p": fields_center_unnormalized[:, [3]],
        "nu": nu + fields_center_unnormalized[:, [4]],
        "u__x": grad_center[:, 0, [0]],
        "u__y": grad_center[:, 1, [0]],
        "u__z": grad_center[:, 2, [0]],
        "v__x": grad_center[:, 0, [1]],
        "v__y": grad_center[:, 1, [1]],
        "v__z": grad_center[:, 2, [1]],
        "w__x": grad_center[:, 0, [2]],
        "w__y": grad_center[:, 1, [2]],
        "w__z": grad_center[:, 2, [2]],
        "p__x": grad_center[:, 0, [3]],
        "p__y": grad_center[:, 1, [3]],
        "p__z": grad_center[:, 2, [3]],
        "nu__x": grad_center[:, 0, [4]],
        "nu__y": grad_center[:, 1, [4]],
        "nu__z": grad_center[:, 2, [4]],
        "u__x__x": grad_grad_uvw_center[:, 0, [0]],
        "u__x__y": grad_grad_uvw_center[:, 1, [0]],
        "u__x__z": grad_grad_uvw_center[:, 2, [0]],
        "u__y__x": grad_grad_uvw_center[:, 1, [0]],  # same as __x__y
        "u__y__y": grad_grad_uvw_center[:, 1, [1]],
        "u__y__z": grad_grad_uvw_center[:, 2, [1]],
        "u__z__x": grad_grad_uvw_center[:, 2, [0]],  # same as __x__z
        "u__z__y": grad_grad_uvw_center[:, 2, [1]],  # same as __y__z
        "u__z__z": grad_grad_uvw_center[:, 2, [2]],
        "v__x__x": grad_grad_uvw_center[:, 0, [3]],
        "v__x__y": grad_grad_uvw_center[:, 1, [3]],
        "v__x__z": grad_grad_uvw_center[:, 2, [3]],
        "v__y__x": grad_grad_uvw_center[:, 1, [3]],  # same as __x__y
        "v__y__y": grad_grad_uvw_center[:, 1, [4]],
        "v__y__z": grad_grad_uvw_center[:, 2, [4]],
        "v__z__x": grad_grad_uvw_center[:, 2, [3]],  # same as __x__z
        "v__z__y": grad_grad_uvw_center[:, 2, [4]],  # same as __y__z
        "v__z__z": grad_grad_uvw_center[:, 2, [5]],
        "w__x__x": grad_grad_uvw_center[:, 0, [6]],
        "w__x__y": grad_grad_uvw_center[:, 1, [6]],
        "w__x__z": grad_grad_uvw_center[:, 2, [6]],
        "w__y__x": grad_grad_uvw_center[:, 1, [6]],  # same as __x__y
        "w__y__y": grad_grad_uvw_center[:, 1, [7]],
        "w__y__z": grad_grad_uvw_center[:, 2, [7]],
        "w__z__x": grad_grad_uvw_center[:, 2, [6]],  # same as __x__z
        "w__z__y": grad_grad_uvw_center[:, 2, [7]],  # same as __y__z
        "w__z__z": grad_grad_uvw_center[:, 2, [8]],
    }
    continuity = eqn["continuity"].evaluate(dict_mapping)["continuity"]
    momentum_x = eqn["momentum_x"].evaluate(dict_mapping)["momentum_x"]
    momentum_y = eqn["momentum_y"].evaluate(dict_mapping)["momentum_y"]
    momentum_z = eqn["momentum_z"].evaluate(dict_mapping)["momentum_z"]

    # Compute the weights for the equation residuals
    weight_continuity = torch.sigmoid(0.5 * (torch.abs(continuity) - 10))
    weight_momentum_x = torch.sigmoid(0.5 * (torch.abs(momentum_x) - 10))
    weight_momentum_y = torch.sigmoid(0.5 * (torch.abs(momentum_y) - 10))
    weight_momentum_z = torch.sigmoid(0.5 * (torch.abs(momentum_z) - 10))

    weighted_continuity = weight_continuity * torch.abs(continuity)
    weighted_momentum_x = weight_momentum_x * torch.abs(momentum_x)
    weighted_momentum_y = weight_momentum_y * torch.abs(momentum_y)
    weighted_momentum_z = weight_momentum_z * torch.abs(momentum_z)

    # Compute data loss
    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * target**2.0, dims)
    else:
        denom = torch.sum(mask)

    del coords_total, output_total
    torch.cuda.empty_cache()

    return (
        torch.mean(num / denom),
        torch.mean(torch.abs(weighted_continuity)),
        torch.mean(torch.abs(weighted_momentum_x)),
        torch.mean(torch.abs(weighted_momentum_y)),
        torch.mean(torch.abs(weighted_momentum_z)),
    )


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
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

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * (target - torch.mean(target, (0, 1))) ** 2.0, dims)
        loss = torch.mean(num / denom)
    elif loss_type == "mse":
        denom = torch.sum(mask)
        loss = torch.mean(num / denom)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def loss_fn_with_physics(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
    first_deriv: torch.nn.Module = None,
    eqn: Any = None,
    bounding_box: torch.Tensor = None,
    vol_factors: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DEPRECATED: Use compute_fvm_physics_loss() instead.
    
    Old gradient-based loss with physics terms.

    Args:
        output: Predicted values from the model (with neighbor data when physics enabled)
        target: Ground truth values
        loss_type: Type of loss to calculate ("mse" or "rmse")
        padded_value: Value used for padding in the tensor
        first_deriv: First derivative calculator
        eqn: Equations
        bounding_box: Bounding box for normalization
        vol_factors: Volume factors for normalization

    Returns:
        Tuple of (data_loss, continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss)
    """
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    # Call the physics loss computation function
    return compute_physics_loss(
        output=output,
        target=target,
        mask=mask,
        loss_type=loss_type,
        dims=dims,
        first_deriv=first_deriv,
        eqn=eqn,
        bounding_box=bounding_box,
        vol_factors=vol_factors,
    )


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
        denom = torch.mean((target_scalar - torch.mean(target_scalar, (0, 1))) ** 2.0)
        masked_loss_pres = numerator / denom

        # Compute the mean diff**2 of the vector component, leave the last dimension:
        masked_loss_ws_num = vector_diff_sq
        masked_loss_ws_denom = torch.mean(
            (target_vector - torch.mean(target_vector, (0, 1))) ** 2.0, (0, 1)
        )
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
        masked_loss_pres /= torch.mean(
            (target_scalar - torch.mean(target_scalar, (0, 1))) ** 2.0, dim=(0, 1)
        )

    # Compute the mean diff**2 of the vector component, leave the last dimension:
    masked_loss_ws = torch.mean((target_vector - output_vector) ** 2.0, (0, 1))
    if loss_type == "rmse":
        masked_loss_ws /= torch.mean(
            (target_vector - torch.mean(target_vector, (0, 1))) ** 2.0, (0, 1)
        )

    # Combine the scalar and vector components:
    loss = 0.25 * (masked_loss_pres + torch.sum(masked_loss_ws))

    return loss


def integral_loss_fn(
    output, target, area, normals, stream_velocity=None, padded_value=-10
):
    drag_loss = drag_loss_fn(
        output, target, area, normals, stream_velocity=stream_velocity, padded_value=-10
    )
    lift_loss = lift_loss_fn(
        output, target, area, normals, stream_velocity=stream_velocity, padded_value=-10
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
    add_physics_loss: bool = False,
    log_physics_loss_only: bool = False,
    # FVM-Based Physics Loss Parameters
    prediction_vol_neighbors: torch.Tensor | None = None,
    datapipe = None,
    physics_loss_weight_continuity: float = 1.0,
    physics_loss_weight_momentum: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the loss terms in a single function call.

    Computes:
    - Volume loss (data loss) if prediction_vol is not None
    - Physics loss (FVM residuals) if add_physics_loss=True or log_physics_loss_only=True
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

        # Unnormalize prediction and target for consistent loss computation with physics loss
        if datapipe is not None and datapipe.volume_factors is not None:
            device = prediction_vol.device
            prediction_vol_unnorm = prediction_vol.clone()
            target_vol_unnorm = target_vol.clone()
            
            if datapipe.scaling_type == "mean_std_scaling":
                # Unstandardize: x_physical = x_normalized * std + mean
                prediction_vol_unnorm = unstandardize(
                    prediction_vol_unnorm,
                    datapipe.volume_factors[0].to(device),  # mean
                    datapipe.volume_factors[1].to(device),  # std
                )
                target_vol_unnorm = unstandardize(
                    target_vol_unnorm,
                    datapipe.volume_factors[0].to(device),
                    datapipe.volume_factors[1].to(device),
                )
            elif datapipe.scaling_type == "min_max_scaling":
                # Unnormalize: x_physical = x_normalized * (max - min) + min
                prediction_vol_unnorm = unnormalize(
                    prediction_vol_unnorm,
                    datapipe.volume_factors[0].to(device),  # min
                    datapipe.volume_factors[1].to(device),  # max
                )
                target_vol_unnorm = unnormalize(
                    target_vol_unnorm,
                    datapipe.volume_factors[0].to(device),
                    datapipe.volume_factors[1].to(device),
                )
        else:
            # Fall back to normalized loss if datapipe not available
            prediction_vol_unnorm = prediction_vol
            target_vol_unnorm = target_vol

        # Data loss (computed on unnormalized values for consistency with physics loss)
        loss_vol = loss_fn(
            prediction_vol_unnorm,
            target_vol_unnorm,
            loss_fn_type.loss_type,
            padded_value=-10,
        )
        loss_dict["loss_vol"] = loss_vol
        total_loss_terms.append(loss_vol * vol_loss_scaling)
        
        # FVM-based physics loss (if enabled for training OR logging)
        if (add_physics_loss or log_physics_loss_only) and datapipe is not None and prediction_vol_neighbors is not None:
            continuity_loss, momentum_x_loss, momentum_y_loss, momentum_z_loss = compute_fvm_physics_loss(
                solutions_main=prediction_vol,
                solutions_neighbors=prediction_vol_neighbors,
                batch=batch_inputs,
                datapipe=datapipe,
                nu=1.5881327800829875e-5,
            )
            
            # Always add to loss_dict for logging
            loss_dict["loss_continuity"] = continuity_loss
            loss_dict["loss_momentum_x"] = momentum_x_loss
            loss_dict["loss_momentum_y"] = momentum_y_loss
            loss_dict["loss_momentum_z"] = momentum_z_loss
            
            # Only add to total loss if add_physics_loss=True (not just logging)
            if add_physics_loss:
                total_loss_terms.append(continuity_loss * physics_loss_weight_continuity)
                total_loss_terms.append(momentum_x_loss * physics_loss_weight_momentum)
                total_loss_terms.append(momentum_y_loss * physics_loss_weight_momentum)
                total_loss_terms.append(momentum_z_loss * physics_loss_weight_momentum)

    if prediction_surf is not None:
        target_surf = batch_inputs["surface_fields"]
        surface_areas = batch_inputs["surface_areas"]
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_normals = batch_inputs["surface_normals"]

        # Needs to be taken from the dataset
        stream_velocity = batch_inputs["global_params_values"][:, 0, :]

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

        total_loss_terms.append(loss_surf)
        loss_dict["loss_surf"] = loss_surf
        total_loss_terms.append(loss_surf_area)
        loss_dict["loss_surf_area"] = loss_surf_area
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
