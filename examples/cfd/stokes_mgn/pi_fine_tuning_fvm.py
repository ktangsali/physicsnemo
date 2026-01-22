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
Physics-Informed Fine-Tuning using FVM Residuals with Delta Learning.

This script fine-tunes GNN predictions using Finite Volume Method (FVM) residuals
computed via Warp kernels with autodiff support.

**Delta Learning Approach:**
Instead of having a data loss, the GNN predictions are provided as INPUT to the model.
The model predicts corrections (Δu, Δv, Δp) that are added to the GNN predictions.
Physics and boundary condition losses are applied to the final result (GNN + delta).

This approach removes the need to balance data loss with physics loss - the physics
constraints directly guide the corrections needed to make GNN predictions physically
consistent.

Key features:
1. 2D mesh is extruded to 3D for FVM computation
2. Point data is converted to cell-centered data for training
3. FVM residuals (continuity, momentum) replace autodiff-based PDE residuals
4. GNN predictions serve as model inputs (not targets)
5. Model learns corrections to make predictions satisfy physics + BCs
6. Final results are converted back to point data for saving
"""

import os
import time
from typing import Any, Callable, Sequence

import hydra
import numpy as np
import torch
from torch.optim import Optimizer
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

try:
    import apex
except:
    pass

try:
    import pyvista as pv
except:
    raise ImportError(
        "Stokes Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )

from collections import OrderedDict

from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.logging.wandb import initialize_wandb


# Check if Muon optimizer is available (PyTorch 2.4+)
HAS_MUON = hasattr(torch.optim, "Muon")


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.
    
    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")
        self.optimizers = optimizers
        # Collect parameter groups from all optimizers. We pass an empty
        # *defaults* dict because hyper-parameters are managed by the inner
        # optimizers, not this wrapper.
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})
        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        """Nullify gradients"""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)
        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]

from scipy.spatial import cKDTree

from utils import get_dataset
from fvm_residuals import (
    build_face_connectivity,
    compute_residuals_warp_batch_autograd,
)


def identify_boundary_faces(
    face_data, 
    coords_inflow, 
    coords_outflow,
    coords_noslip,
    extrusion_length=1.0, 
    z_tol=0.01
):
    """
    Identify boundary face centers for applying BCs in the FVM framework.
    
    Uses the marker-derived boundary point coordinates to classify faces:
    - Inflow faces: face centers nearest to inflow boundary points (marker=1)
    - Outflow faces: face centers nearest to outflow boundary points (marker=2) - ignored
    - Noslip faces: face centers nearest to wall/polygon points (marker=3,4)
    
    Since the mesh is extruded in z-direction, we first filter out z=0 and 
    z=extrusion_length faces (the extruded caps we want to ignore).
    
    Args:
        face_data: dict from build_face_connectivity with face_neighbor, face_centers, etc.
        coords_inflow: [n_inflow, 2] inflow boundary point coordinates (marker=1)
        coords_outflow: [n_outflow, 2] outflow boundary point coordinates (marker=2)
        coords_noslip: [n_noslip, 2] noslip boundary point coordinates (marker=3,4)
        extrusion_length: the extrusion length in z direction
        z_tol: tolerance for filtering z=0 and z=extrusion_length faces
        
    Returns:
        inflow_face_indices: indices of inflow boundary faces
        noslip_face_indices: indices of no-slip boundary faces
    """
    face_neighbor = face_data['face_neighbor']
    face_centers = face_data['face_centers']
    
    # Step 1: Get all boundary faces (neighbor == -1)
    boundary_mask = (face_neighbor == -1)
    boundary_indices = np.where(boundary_mask)[0]
    
    # Step 2: Filter out z=0 and z=extrusion_length faces (extruded caps)
    boundary_face_centers = face_centers[boundary_indices]
    z_coords = boundary_face_centers[:, 2]
    side_face_mask = (z_coords > z_tol) & (z_coords < extrusion_length - z_tol)
    side_boundary_indices = boundary_indices[side_face_mask]
    side_face_centers = face_centers[side_boundary_indices]
    
    # Step 3: Classify boundary faces using marker-derived point coordinates
    # Build KD-trees for each boundary type (using only x,y)
    inflow_xy = coords_inflow[:, :2] if coords_inflow.shape[1] > 2 else coords_inflow
    outflow_xy = coords_outflow[:, :2] if coords_outflow.shape[1] > 2 else coords_outflow
    noslip_xy = coords_noslip[:, :2] if coords_noslip.shape[1] > 2 else coords_noslip
    
    tree_inflow = cKDTree(inflow_xy)
    tree_outflow = cKDTree(outflow_xy)
    tree_noslip = cKDTree(noslip_xy)
    
    # For each side boundary face, find distance to nearest point of each boundary type
    face_xy = side_face_centers[:, :2]
    dist_inflow, _ = tree_inflow.query(face_xy)
    dist_outflow, _ = tree_outflow.query(face_xy)
    dist_noslip, _ = tree_noslip.query(face_xy)
    
    # Classify: face belongs to whichever boundary type it's CLOSEST to
    # This correctly handles all three boundary types
    inflow_mask = (dist_inflow <= dist_outflow) & (dist_inflow <= dist_noslip)
    outflow_mask = (dist_outflow < dist_inflow) & (dist_outflow <= dist_noslip)
    noslip_mask = (dist_noslip < dist_inflow) & (dist_noslip < dist_outflow)
    
    inflow_face_indices = side_boundary_indices[inflow_mask]
    outflow_face_indices = side_boundary_indices[outflow_mask]
    noslip_face_indices = side_boundary_indices[noslip_mask]
    
    print(f"  Boundary faces: {len(boundary_indices)} total, {len(side_boundary_indices)} side faces")
    print(f"  Inflow: {len(inflow_face_indices)}, Outflow (ignored): {len(outflow_face_indices)}, Noslip: {len(noslip_face_indices)}")
    
    # Debug: print x-coordinate ranges for each boundary type
    if len(inflow_face_indices) > 0:
        inflow_x = face_centers[inflow_face_indices, 0]
        print(f"    Inflow face x-range: [{inflow_x.min():.4f}, {inflow_x.max():.4f}]")
    if len(outflow_face_indices) > 0:
        outflow_x = face_centers[outflow_face_indices, 0]
        print(f"    Outflow face x-range: [{outflow_x.min():.4f}, {outflow_x.max():.4f}]")
    if len(noslip_face_indices) > 0:
        noslip_x = face_centers[noslip_face_indices, 0]
        print(f"    Noslip face x-range: [{noslip_x.min():.4f}, {noslip_x.max():.4f}]")
    
    return inflow_face_indices, noslip_face_indices


class DNN(torch.nn.Module):
    """Custom PyTorch model with Fourier features."""

    def __init__(self, layers, fourier_features=64, zero_init_last=False):
        super().__init__()

        self.depth = len(layers) - 1
        self.fourier_features = fourier_features
        self.register_buffer("B", 10 * torch.randn((layers[0], fourier_features)))

        self.activation = torch.nn.GELU

        layer_list = list()
        for i in range(1, self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        # Final layer
        final_layer = torch.nn.Linear(layers[-2], layers[-1])
        if zero_init_last:
            # Zero-initialize final layer so output starts at ~0
            torch.nn.init.zeros_(final_layer.weight)
            torch.nn.init.zeros_(final_layer.bias)
        layer_list.append(("layer_%d" % (self.depth - 1), final_layer))
        
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.layers(x_proj)
        return out


class StokesFVMDeltaModel(torch.nn.Module):
    """
    Delta learning model for Stokes flow that predicts corrections (Δu, Δv, Δp)
    given (x, y) coordinates and GNN predictions (u_gnn, v_gnn, p_gnn).
    
    The final prediction is: u = u_gnn + α * Δu, v = v_gnn + α * Δv, p = p_gnn + α * Δp.
    
    Features to preserve initial GNN predictions:
    1. Zero-initialized final layer: Model starts with delta ≈ 0
    2. Learnable scaling factor (α): Controls magnitude of corrections, starts small
    3. Skip connection: GNN predictions are directly added to scaled deltas
    
    This approach removes the need for data loss since the GNN predictions are
    used as inputs, and the physics/BC losses guide the corrections.
    """

    def __init__(
        self, 
        layers=[5, 128, 128, 128, 128, 3], 
        fourier_features=64,
        zero_init=True,
        learnable_scale=True,
        initial_scale=0.1,
    ):
        super().__init__()
        # Input dimension is 5: (x, y, u_gnn, v_gnn, p_gnn)
        self.dnn = DNN(layers, fourier_features, zero_init_last=zero_init)
        
        # Learnable scaling factor for deltas (starts small to preserve GNN predictions)
        self.learnable_scale = learnable_scale
        if learnable_scale:
            # Separate scales for velocity and pressure (they have different magnitudes)
            self.alpha_vel = torch.nn.Parameter(torch.tensor(initial_scale))
            self.alpha_p = torch.nn.Parameter(torch.tensor(initial_scale))
        else:
            self.register_buffer("alpha_vel", torch.tensor(1.0))
            self.register_buffer("alpha_p", torch.tensor(1.0))
        
        self.zero_init = zero_init
        self.initial_scale = initial_scale

    def forward(self, coords_xy, gnn_uvp):
        """
        Args:
            coords_xy: [N, 2] tensor of (x, y) coordinates
            gnn_uvp: [N, 3] tensor of (u_gnn, v_gnn, p_gnn) predictions

        Returns:
            dict with 'u', 'v', 'p' tensors of shape [N, 1] (final predictions = GNN + scaled delta)
            Also returns 'delta_u', 'delta_v', 'delta_p' (raw), 'alpha_vel', 'alpha_p' for monitoring
        """
        # Concatenate coordinates and GNN predictions
        x = torch.cat([coords_xy, gnn_uvp], dim=1)  # [N, 5]
        delta_raw = self.dnn(x)  # [N, 3]
        
        # Extract raw deltas
        delta_u_raw = delta_raw[:, 0:1]
        delta_v_raw = delta_raw[:, 1:2]
        delta_p_raw = delta_raw[:, 2:3]
        
        # Scale deltas (separate scales for velocity and pressure)
        delta_u = self.alpha_vel * delta_u_raw
        delta_v = self.alpha_vel * delta_v_raw
        delta_p = self.alpha_p * delta_p_raw
        
        # Final predictions = GNN + scaled delta (skip connection)
        u_final = gnn_uvp[:, 0:1] + delta_u
        v_final = gnn_uvp[:, 1:2] + delta_v
        p_final = gnn_uvp[:, 2:3] + delta_p
        
        return {
            "u": u_final,
            "v": v_final,
            "p": p_final,
            "delta_u": delta_u,
            "delta_v": delta_v,
            "delta_p": delta_p,
            "alpha_vel": self.alpha_vel,
            "alpha_p": self.alpha_p,
        }


def extrude_2d_to_3d(polydata_2d, extrusion_length=1.0):
    """
    Extrude a 2D PolyData mesh to 3D volumetric cells.

    Creates proper 3D volumetric cells for FVM:
    - Triangles (VTK_TRIANGLE) -> Wedges/Prisms (VTK_WEDGE)
    - Quads (VTK_QUAD) -> Hexahedra (VTK_HEXAHEDRON)

    Args:
        polydata_2d: PyVista PolyData object (2D mesh)
        extrusion_length: Length to extrude in z direction

    Returns:
        PyVista UnstructuredGrid (3D volumetric mesh)
    """
    import vtk
    from vtk.util import numpy_support

    # Get 2D points and add z=0 if needed
    points_2d = np.array(polydata_2d.points)
    n_points_2d = len(points_2d)

    # Ensure points are 3D with z=0
    if points_2d.shape[1] == 2:
        points_2d = np.column_stack([points_2d, np.zeros(n_points_2d)])
    else:
        points_2d[:, 2] = 0.0  # Set z to 0 for bottom layer

    # Create top layer points (z = extrusion_length)
    points_top = points_2d.copy()
    points_top[:, 2] = extrusion_length

    # Combine bottom and top points
    all_points = np.vstack([points_2d, points_top])

    # Create VTK points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(all_points.astype(np.float64)))

    # Get 2D cells and create 3D cells
    cells = vtk.vtkCellArray()
    cell_types = []

    # VTK cell type constants
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    VTK_WEDGE = 13
    VTK_HEXAHEDRON = 12

    # Iterate through 2D cells
    polys = polydata_2d.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        n_pts = id_list.GetNumberOfIds()

        if n_pts == 3:
            # Triangle -> Wedge (prism)
            # Bottom: p0, p1, p2 (CCW)
            # Top: p0+offset, p1+offset, p2+offset (CCW)
            p0 = id_list.GetId(0)
            p1 = id_list.GetId(1)
            p2 = id_list.GetId(2)

            # VTK Wedge ordering: bottom triangle (0,1,2), top triangle (3,4,5)
            # where top points are directly above bottom points
            cells.InsertNextCell(6)
            cells.InsertCellPoint(p0)  # bottom 0
            cells.InsertCellPoint(p1)  # bottom 1
            cells.InsertCellPoint(p2)  # bottom 2
            cells.InsertCellPoint(p0 + n_points_2d)  # top 0
            cells.InsertCellPoint(p1 + n_points_2d)  # top 1
            cells.InsertCellPoint(p2 + n_points_2d)  # top 2
            cell_types.append(VTK_WEDGE)

        elif n_pts == 4:
            # Quad -> Hexahedron
            p0 = id_list.GetId(0)
            p1 = id_list.GetId(1)
            p2 = id_list.GetId(2)
            p3 = id_list.GetId(3)

            # VTK Hexahedron ordering: bottom quad (0,1,2,3), top quad (4,5,6,7)
            cells.InsertNextCell(8)
            cells.InsertCellPoint(p0)  # bottom 0
            cells.InsertCellPoint(p1)  # bottom 1
            cells.InsertCellPoint(p2)  # bottom 2
            cells.InsertCellPoint(p3)  # bottom 3
            cells.InsertCellPoint(p0 + n_points_2d)  # top 0
            cells.InsertCellPoint(p1 + n_points_2d)  # top 1
            cells.InsertCellPoint(p2 + n_points_2d)  # top 2
            cells.InsertCellPoint(p3 + n_points_2d)  # top 3
            cell_types.append(VTK_HEXAHEDRON)

        else:
            raise ValueError(f"Unsupported 2D cell with {n_pts} points")

    # Create unstructured grid
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(cell_types, cells)

    # Copy point data (replicate to both layers)
    for i in range(polydata_2d.GetPointData().GetNumberOfArrays()):
        arr = polydata_2d.GetPointData().GetArray(i)
        arr_name = arr.GetName()
        arr_np = numpy_support.vtk_to_numpy(arr)

        # Replicate data for top layer (same values since z-invariant)
        arr_combined = np.concatenate([arr_np, arr_np])

        new_arr = numpy_support.numpy_to_vtk(arr_combined)
        new_arr.SetName(arr_name)
        ugrid.GetPointData().AddArray(new_arr)

    # Convert to PyVista UnstructuredGrid
    return pv.wrap(ugrid)


class PhysicsInformedFineTunerFVM:
    """
    Physics-informed fine-tuner using FVM residuals with delta learning.
    
    Instead of having a data loss, the GNN predictions are provided as input
    to the model. The model predicts corrections (deltas) that are added to
    the GNN predictions. Physics and BC losses are applied to the final result.
    
    Features to preserve initial GNN predictions:
    - Zero-initialized final layer (zero_init=True)
    - Learnable scaling factor for deltas (learnable_scale=True)
    - Delta regularization loss (delta_reg_weight > 0)
    """

    def __init__(
        self,
        device,
        gnn_u_cell,
        gnn_v_cell,
        gnn_p_cell,
        ref_u_cell,
        ref_v_cell,
        ref_p_cell,
        cell_centers,
        cell_volumes,
        inflow_face_indices,
        noslip_face_indices,
        face_data,
        n_cells,
        nu,
        lr=0.001,
        total_iters=1000,
        use_lr_scheduler=True,
        # Original point data (for validation only - not used in training)
        ref_u_point=None,
        ref_v_point=None,
        ref_p_point=None,
        coords_point=None,
        # GNN predictions at point locations (for validation with delta model)
        gnn_u_point=None,
        gnn_v_point=None,
        gnn_p_point=None,
        # Delta learning options to preserve initial predictions
        zero_init=True,           # Zero-initialize final layer (delta starts at ~0)
        learnable_scale=True,     # Learnable scaling factor for deltas
        initial_scale=0.1,        # Initial value for scaling factor
        delta_reg_weight=0.0,     # Weight for delta regularization (penalize large corrections)
    ):
        super().__init__()

        self.device = device
        self.nu = nu
        self.n_cells = n_cells
        self.lr = lr
        self.total_iters = total_iters
        self.use_lr_scheduler = use_lr_scheduler

        # Cell-centered reference data
        self.ref_u_cell = torch.tensor(ref_u_cell).float().to(self.device).reshape(-1, 1)
        self.ref_v_cell = torch.tensor(ref_v_cell).float().to(self.device).reshape(-1, 1)
        self.ref_p_cell = torch.tensor(ref_p_cell).float().to(self.device).reshape(-1, 1)

        # Cell-centered GNN predictions (used as MODEL INPUT for delta learning)
        self.gnn_u_cell = torch.tensor(gnn_u_cell).float().to(self.device).reshape(-1, 1)
        self.gnn_v_cell = torch.tensor(gnn_v_cell).float().to(self.device).reshape(-1, 1)
        self.gnn_p_cell = torch.tensor(gnn_p_cell).float().to(self.device).reshape(-1, 1)
        
        # Combined GNN predictions at cell centers for model input [n_cells, 3]
        self.gnn_uvp_cell = torch.cat(
            [self.gnn_u_cell, self.gnn_v_cell, self.gnn_p_cell], dim=1
        )

        # Cell centers - keep full 3D for FVM distance calculations
        self.cell_centers_3d = cell_centers.astype(np.float32)
        # Use only (x, y) for model input (z-invariant)
        self.cell_centers_xy = (
            torch.tensor(cell_centers[:, :2], requires_grad=False).float().to(self.device)
        )

        # Cell volumes for FVM
        self.cell_volumes_np = cell_volumes.astype(np.float32)

        # Boundary face centers (for applying BCs at face centers - more physical for FVM)
        face_centers_np = face_data['face_centers']
        self.coords_inflow_face = (
            torch.tensor(face_centers_np[inflow_face_indices, :2], requires_grad=False)
            .float().to(self.device)
        )
        self.coords_noslip_face = (
            torch.tensor(face_centers_np[noslip_face_indices, :2], requires_grad=False)
            .float().to(self.device)
        )
        
        # Interpolate GNN predictions from cell centers to boundary face centers
        # Using nearest neighbor interpolation based on (x, y) coordinates
        print("Interpolating GNN predictions to boundary face centers...")
        cell_xy = cell_centers[:, :2]  # [n_cells, 2]
        tree_cells = cKDTree(cell_xy)
        
        # Inflow faces: find nearest cell for each face center
        inflow_face_xy = face_centers_np[inflow_face_indices, :2]
        _, inflow_nearest_cells = tree_cells.query(inflow_face_xy)
        self.gnn_uvp_inflow = torch.tensor(
            np.column_stack([
                gnn_u_cell[inflow_nearest_cells],
                gnn_v_cell[inflow_nearest_cells],
                gnn_p_cell[inflow_nearest_cells],
            ]), requires_grad=False
        ).float().to(self.device)
        
        # Noslip faces: find nearest cell for each face center
        noslip_face_xy = face_centers_np[noslip_face_indices, :2]
        _, noslip_nearest_cells = tree_cells.query(noslip_face_xy)
        self.gnn_uvp_noslip = torch.tensor(
            np.column_stack([
                gnn_u_cell[noslip_nearest_cells],
                gnn_v_cell[noslip_nearest_cells],
                gnn_p_cell[noslip_nearest_cells],
            ]), requires_grad=False
        ).float().to(self.device)
        
        print(f"Boundary faces - Inflow: {len(inflow_face_indices)}, Noslip: {len(noslip_face_indices)}")
        
        # Point data for validation only (to match PINN validation exactly)
        self.has_point_data = coords_point is not None
        if self.has_point_data:
            self.coords_point = (
                torch.tensor(coords_point[:, :2], requires_grad=False).float().to(self.device)
            )
            self.ref_u_point = torch.tensor(ref_u_point).float().to(self.device).reshape(-1, 1)
            self.ref_v_point = torch.tensor(ref_v_point).float().to(self.device).reshape(-1, 1)
            self.ref_p_point = torch.tensor(ref_p_point).float().to(self.device).reshape(-1, 1)
            
            # GNN predictions at point locations (for delta model validation)
            if gnn_u_point is not None:
                self.gnn_uvp_point = torch.tensor(
                    np.column_stack([
                        np.asarray(gnn_u_point).flatten(),
                        np.asarray(gnn_v_point).flatten(),
                        np.asarray(gnn_p_point).flatten(),
                    ]), requires_grad=False
                ).float().to(self.device)
            else:
                # Fallback: interpolate from cell centers if point GNN data not provided
                print("Warning: GNN point data not provided, interpolating from cell centers")
                point_xy = coords_point[:, :2]
                _, point_nearest_cells = tree_cells.query(point_xy)
                self.gnn_uvp_point = torch.tensor(
                    np.column_stack([
                        gnn_u_cell[point_nearest_cells],
                        gnn_v_cell[point_nearest_cells],
                        gnn_p_cell[point_nearest_cells],
                    ]), requires_grad=False
                ).float().to(self.device)

        # Face connectivity for FVM
        self.face_data = face_data

        # All cell indices (for full mesh FVM computation)
        self.all_cell_indices = np.arange(n_cells, dtype=np.int32)

        # Mesh data dict for FVM functions
        self.mesh_data_for_fvm = {
            "cell_centers": self.cell_centers_3d,
            "cell_volumes": self.cell_volumes_np,
            "n_cells": n_cells,
        }

        # Store delta learning options
        self.delta_reg_weight = delta_reg_weight
        self.zero_init = zero_init
        self.learnable_scale = learnable_scale
        self.initial_scale = initial_scale

        # Delta model: takes (x, y, u_gnn, v_gnn, p_gnn), outputs corrections
        # Input dim = 5, output dim = 3
        # With options to preserve initial GNN predictions
        self.model = StokesFVMDeltaModel(
            layers=[5, 128, 128, 128, 128, 3],
            fourier_features=64,
            zero_init=zero_init,
            learnable_scale=learnable_scale,
            initial_scale=initial_scale,
        ).to(self.device)
        
        print(f"Delta model settings: zero_init={zero_init}, learnable_scale={learnable_scale}, "
              f"initial_scale={initial_scale}, delta_reg_weight={delta_reg_weight}")

        # Setup optimizer: Muon for 2D params (weight matrices), AdamW for others
        self._setup_optimizer()

        # Cosine Annealing LR scheduler: decay from lr (max) to 1e-6 (min) by end of training
        # LR follows cosine curve from lr_max to lr_min over T_max iterations
        self.scheduler = None
        if self.use_lr_scheduler:
            lr_min = 1e-6
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.total_iters, 
                eta_min=lr_min
            )
            print(f"Cosine Annealing LR: {self.lr:.6f} (max) -> {lr_min:.6f} (min) over {self.total_iters} iters")
        else:
            print(f"LR scheduler disabled, using constant LR: {self.lr:.6f}")

    def _setup_optimizer(self):
        """Setup optimizer with Muon for 2D params and AdamW for others."""
        # Separate 2D parameters (weight matrices) from others (biases, etc.)
        muon_params = [p for p in self.model.parameters() if p.ndim == 2]
        other_params = [p for p in self.model.parameters() if p.ndim != 2]

        print(f"Muon params (2D weights): {len(muon_params)}")
        print(f"Other params (biases, etc.): {len(other_params)}")

        if HAS_MUON and len(muon_params) > 0:
            # Use Muon for 2D weight matrices, AdamW for others
            base_opt = torch.optim.AdamW(
                other_params,
                lr=self.lr,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                eps=1.0e-8,
            ) if other_params else None

            muon_opt = torch.optim.Muon(
                muon_params,
                lr=self.lr,
                weight_decay=0.0,
            )

            if base_opt is not None:
                self.optimizer = CombinedOptimizer(optimizers=[muon_opt, base_opt])
            else:
                self.optimizer = muon_opt

            print("Using Muon optimizer for 2D weight matrices")
        else:
            # Fallback to Adam if Muon not available
            if not HAS_MUON:
                print("Muon optimizer not available (requires PyTorch 2.4+), using Adam")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                fused=True if torch.cuda.is_available() else False,
            )

    def parabolic_inflow(self, y, U_max=0.3):
        """Parabolic inflow velocity profile."""
        u = 4 * U_max * y * (0.4 - y) / (0.4**2)
        v = torch.zeros_like(y)
        return u, v

    def loss(self):
        # =================================================================
        # Boundary condition losses (at boundary face centers)
        # Delta model: takes (coords, gnn_uvp), outputs final predictions
        # =================================================================

        # Inflow BC (at inflow boundary face centers)
        results_inflow = self.model(self.coords_inflow_face, self.gnn_uvp_inflow)
        pred_u_in, pred_v_in = results_inflow["u"], results_inflow["v"]
        u_in, v_in = self.parabolic_inflow(self.coords_inflow_face[:, 1:2])
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        # No-slip BC (at noslip boundary face centers)
        results_noslip = self.model(self.coords_noslip_face, self.gnn_uvp_noslip)
        pred_u_noslip, pred_v_noslip = results_noslip["u"], results_noslip["v"]
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # =================================================================
        # Cell-centered predictions for FVM residuals
        # Delta model takes GNN predictions as input, no more data loss
        # =================================================================

        model_out_cell = self.model(self.cell_centers_xy, self.gnn_uvp_cell)
        pred_u_cell = model_out_cell["u"]  # [n_cells, 1] = gnn_u + delta_u
        pred_v_cell = model_out_cell["v"]  # [n_cells, 1] = gnn_v + delta_v
        pred_p_cell = model_out_cell["p"]  # [n_cells, 1] = gnn_p + delta_p
        
        # Track delta magnitudes for monitoring (detached, no gradient)
        delta_u = model_out_cell["delta_u"]
        delta_v = model_out_cell["delta_v"]
        delta_p = model_out_cell["delta_p"]
        delta_u_mag = torch.mean(torch.abs(delta_u)).detach()
        delta_v_mag = torch.mean(torch.abs(delta_v)).detach()
        delta_p_mag = torch.mean(torch.abs(delta_p)).detach()

        # =================================================================
        # FVM Physics Residuals (at cell centers)
        # Applied to final predictions (GNN + delta)
        # =================================================================

        # Construct velocity field [n_cells, 3] with w=0 enforced
        velocity_3d = torch.cat(
            [
                pred_u_cell,
                pred_v_cell,
                torch.zeros_like(pred_u_cell),  # w = 0 enforced
            ],
            dim=1,
        )  # [n_cells, 3]

        # Pressure [n_cells]
        pressure_1d = pred_p_cell.squeeze(-1)  # [n_cells]

        # nut = 0 (laminar Stokes)
        nut_1d = torch.zeros(self.n_cells, device=self.device, dtype=torch.float32)

        # Compute FVM residuals with autograd support (Stokes = no convection)
        continuity, momentum_x, momentum_y, momentum_z = compute_residuals_warp_batch_autograd(
            velocity_data=velocity_3d,
            pressure_data=pressure_1d,
            nut_data=nut_1d,
            mesh_data=self.mesh_data_for_fvm,
            face_data=self.face_data,
            batch_cell_indices=self.all_cell_indices,
            nu=self.nu,
            device=str(self.device),
            stokes_flow=True,  # Use Stokes equations (no convection term)
        )

        # Physics losses (FVM residuals should be zero)
        loss_cont = torch.mean(continuity**2)
        loss_mom_u = torch.mean(momentum_x**2)
        loss_mom_v = torch.mean(momentum_y**2)
        # momentum_z not included since w=0 is enforced
        
        # =================================================================
        # Delta regularization loss (optional - penalize large corrections)
        # Encourages the model to make minimal corrections to preserve GNN
        # =================================================================
        loss_delta_reg = torch.mean(delta_u**2 + delta_v**2 + delta_p**2)
        
        # Get alpha values for monitoring
        alpha_vel = model_out_cell["alpha_vel"]
        alpha_p = model_out_cell["alpha_p"]

        return (
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            loss_delta_reg,
            delta_u_mag,
            delta_v_mag,
            delta_p_mag,
            alpha_vel,
            alpha_p,
        )

    def train_step(self):
        """Single FVM-based fine-tuning step with delta learning."""
        self.model.train()

        (
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            loss_delta_reg,
            delta_u_mag,
            delta_v_mag,
            delta_p_mag,
            alpha_vel,
            alpha_p,
        ) = self.loss()

        # Weighted loss combination (no data loss - GNN predictions are inputs)
        # BC losses, physics losses, and optional delta regularization
        loss = (
            10000 * loss_u_in
            + 10000 * loss_v_in
            + 100 * loss_u_noslip
            + 100 * loss_v_noslip
            + 10000 * loss_mom_u
            + 10000 * loss_mom_v
            + 10000 * loss_cont
            + self.delta_reg_weight * loss_delta_reg  # Penalize large corrections
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            loss_delta_reg,
            delta_u_mag,
            delta_v_mag,
            delta_p_mag,
            alpha_vel,
            alpha_p,
        )

    def validation(self):
        """Validation during the FVM fine-tuning step (at original point locations).
        
        Uses point data to match the PINN validation exactly for fair comparison.
        Delta model takes GNN predictions as input and outputs corrections.
        """
        self.model.eval()
        with torch.no_grad():
            # Validate at original point locations (same as PINN)
            # Delta model: (coords, gnn_uvp) -> final predictions
            model_out = self.model(self.coords_point, self.gnn_uvp_point)
            pred_u, pred_v, pred_p = (
                model_out["u"],
                model_out["v"],
                model_out["p"],
            )
            error_u = torch.linalg.norm(self.ref_u_point - pred_u) / torch.linalg.norm(
                self.ref_u_point
            )
            error_v = torch.linalg.norm(self.ref_v_point - pred_v) / torch.linalg.norm(
                self.ref_v_point
            )
            error_p = torch.linalg.norm(self.ref_p_point - pred_p) / torch.linalg.norm(
                self.ref_p_point
            )
            wandb.log(
                {
                    "test_u_error (%)": error_u.detach().cpu().numpy(),
                    "test_v_error (%)": error_v.detach().cpu().numpy(),
                    "test_p_error (%)": error_p.detach().cpu().numpy(),
                }
            )
            return error_u, error_v, error_p


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize loggers
    initialize_wandb(
        project="PhysicsNeMo-Launch",
        entity="PhysicsNeMo",
        name="Stokes-FVM-Delta-Learning",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")
    logger.file_logging()

    # Get dataset path
    path = os.path.join(to_absolute_path(cfg.results_dir), cfg.graph_path)
    logger.info(f"Loading 2D mesh from: {path}")

    # =========================================================================
    # Step 1: Load 2D VTP and extract data
    # =========================================================================
    (
        ref_u,
        ref_v,
        ref_p,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_outflow,
        coords_wall,
        coords_polygon,
        nu,
    ) = get_dataset(path)
    coords_noslip = np.concatenate([coords_wall, coords_polygon], axis=0)

    logger.info(f"2D mesh: {len(coords)} points")
    logger.info(f"Boundary points - Inflow: {len(coords_inflow)}, Outflow: {len(coords_outflow)}, Noslip: {len(coords_noslip)}")

    # =========================================================================
    # Step 2: Load 2D mesh and add point data for extrusion
    # =========================================================================
    polydata_2d = pv.read(path)

    # Add GNN predictions and reference as point data
    polydata_2d.point_data["gnn_u"] = gnn_u.flatten()
    polydata_2d.point_data["gnn_v"] = gnn_v.flatten()
    polydata_2d.point_data["gnn_p"] = gnn_p.flatten()
    polydata_2d.point_data["ref_u"] = ref_u.flatten()
    polydata_2d.point_data["ref_v"] = ref_v.flatten()
    polydata_2d.point_data["ref_p"] = ref_p.flatten()

    # =========================================================================
    # Step 3: Extrude 2D mesh to 3D
    # =========================================================================
    logger.info("Extruding 2D mesh to 3D (z=0 to z=1)...")
    ugrid_3d = extrude_2d_to_3d(polydata_2d, extrusion_length=1)
    logger.info(f"3D mesh: {ugrid_3d.n_cells} cells, {ugrid_3d.n_points} points")

    # Save extruded mesh for reference
    extruded_path = path.replace(".vtp", "_extruded.vtu")
    ugrid_3d.save(extruded_path)
    logger.info(f"Saved extruded mesh to: {extruded_path}")

    # =========================================================================
    # Step 4: Convert point data to cell data
    # =========================================================================
    logger.info("Converting point data to cell data...")
    ugrid_3d_cell = ugrid_3d.point_data_to_cell_data(pass_point_data=True)

    # Extract cell-centered data
    gnn_u_cell = ugrid_3d_cell.cell_data["gnn_u"].astype(np.float32)
    gnn_v_cell = ugrid_3d_cell.cell_data["gnn_v"].astype(np.float32)
    gnn_p_cell = ugrid_3d_cell.cell_data["gnn_p"].astype(np.float32)
    ref_u_cell = ugrid_3d_cell.cell_data["ref_u"].astype(np.float32)
    ref_v_cell = ugrid_3d_cell.cell_data["ref_v"].astype(np.float32)
    ref_p_cell = ugrid_3d_cell.cell_data["ref_p"].astype(np.float32)

    logger.info(f"Cell-centered data shapes: gnn_u={gnn_u_cell.shape}, ref_u={ref_u_cell.shape}")

    # =========================================================================
    # Step 5: Compute cell centers and volumes
    # =========================================================================
    logger.info("Computing cell centers and volumes...")
    cell_centers = ugrid_3d_cell.cell_centers().points.astype(np.float32)

    sized = ugrid_3d_cell.compute_cell_sizes(length=False, area=False, volume=True)
    cell_volumes = sized.cell_data["Volume"].astype(np.float32)

    n_cells = ugrid_3d_cell.n_cells
    logger.info(f"Cell centers shape: {cell_centers.shape}, n_cells: {n_cells}")

    # =========================================================================
    # Step 6: Build face connectivity for FVM
    # =========================================================================
    logger.info("Building face connectivity for FVM...")
    face_data = build_face_connectivity(ugrid_3d_cell)
    logger.info(f"Face connectivity: {face_data['n_faces']} faces")

    # =========================================================================
    # Step 7: Identify boundary faces using marker-derived coordinates
    # =========================================================================
    logger.info("Identifying boundary faces using marker data...")
    extrusion_length = 1.0  # Must match the extrusion in Step 3
    inflow_face_indices, noslip_face_indices = identify_boundary_faces(
        face_data, 
        coords_inflow,
        coords_outflow,
        coords_noslip,
        extrusion_length=extrusion_length
    )
    logger.info(f"Boundary faces - Inflow: {len(inflow_face_indices)}, Noslip: {len(noslip_face_indices)}")

    # =========================================================================
    # Step 8: Initialize fine-tuner (delta learning approach)
    # =========================================================================
    logger.info("Initializing FVM-based fine-tuner with delta learning...")
    logger.info("GNN predictions will be used as MODEL INPUT, not as targets.")
    logger.info("The model learns corrections (deltas) to make predictions physics-consistent.")
    
    # Get LR scheduler flag from config (default to True if not specified)
    use_lr_scheduler = getattr(cfg, 'pi_use_lr_scheduler', True)
    
    # Get delta learning options from config (with sensible defaults)
    zero_init = getattr(cfg, 'delta_zero_init', True)
    learnable_scale = getattr(cfg, 'delta_learnable_scale', True)
    initial_scale = getattr(cfg, 'delta_initial_scale', 0.1)
    delta_reg_weight = getattr(cfg, 'delta_reg_weight', 0.0)
    
    pi_fine_tuner = PhysicsInformedFineTunerFVM(
        device=device,
        gnn_u_cell=gnn_u_cell,
        gnn_v_cell=gnn_v_cell,
        gnn_p_cell=gnn_p_cell,
        ref_u_cell=ref_u_cell,
        ref_v_cell=ref_v_cell,
        ref_p_cell=ref_p_cell,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        inflow_face_indices=inflow_face_indices,
        noslip_face_indices=noslip_face_indices,
        face_data=face_data,
        n_cells=n_cells,
        nu=nu,
        lr=cfg.pi_lr,
        total_iters=cfg.pi_iters,
        use_lr_scheduler=use_lr_scheduler,
        # Point data for validation (to match PINN validation)
        ref_u_point=ref_u,
        ref_v_point=ref_v,
        ref_p_point=ref_p,
        coords_point=coords,
        # GNN predictions at point locations (for delta model validation)
        gnn_u_point=gnn_u,
        gnn_v_point=gnn_v,
        gnn_p_point=gnn_p,
        # Delta learning options to preserve initial GNN predictions
        zero_init=zero_init,
        learnable_scale=learnable_scale,
        initial_scale=initial_scale,
        delta_reg_weight=delta_reg_weight,
    )

    # =========================================================================
    # Step 9: Training loop (delta learning - no data loss)
    # =========================================================================
    logger.info("Starting FVM-based physics-informed fine-tuning (delta learning)...")
    logger.info("No data loss - GNN predictions are inputs, model learns corrections.")

    for iters in range(cfg.pi_iters):
        start_iter_time = time.time()

        (
            loss_u_in,
            loss_v_in,
            loss_u_noslip,
            loss_v_noslip,
            loss_mom_u,
            loss_mom_v,
            loss_cont,
            loss_delta_reg,
            delta_u_mag,
            delta_v_mag,
            delta_p_mag,
            alpha_vel,
            alpha_p,
        ) = pi_fine_tuner.train_step()

        if iters % 100 == 0:
            error_u, error_v, error_p = pi_fine_tuner.validation()

            logger.info(f"Iteration: {iters}")
            logger.info(f"--- BC Losses ---")
            logger.info(f"Loss u_in: {loss_u_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v_in: {loss_v_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u noslip: {loss_u_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v noslip: {loss_v_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"--- Physics Losses (FVM) ---")
            logger.info(f"Loss momentum u: {loss_mom_u.detach().cpu().numpy():.3e}")
            logger.info(f"Loss momentum v: {loss_mom_v.detach().cpu().numpy():.3e}")
            logger.info(f"Loss continuity: {loss_cont.detach().cpu().numpy():.3e}")
            logger.info(f"Loss delta reg: {loss_delta_reg.detach().cpu().numpy():.3e}")
            logger.info(f"--- Delta Scaling Factors (learnable) ---")
            logger.info(f"α_vel: {alpha_vel.detach().cpu().numpy():.4f}")
            logger.info(f"α_p: {alpha_p.detach().cpu().numpy():.4f}")
            logger.info(f"--- Delta Magnitudes (scaled corrections) ---")
            logger.info(f"Mean |Δu|: {delta_u_mag.cpu().numpy():.3e}")
            logger.info(f"Mean |Δv|: {delta_v_mag.cpu().numpy():.3e}")
            logger.info(f"Mean |Δp|: {delta_p_mag.cpu().numpy():.3e}")
            logger.info(f"--- Validation Errors ---")
            logger.info(f"Error u: {error_u:.3e}")
            logger.info(f"Error v: {error_v:.3e}")
            logger.info(f"Error p: {error_p:.3e}")

            # Log current learning rate
            current_lr = pi_fine_tuner.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.3e}")

            end_iter_time = time.time()
            logger.info(f"This iteration took {end_iter_time - start_iter_time:.2f} seconds")
            logger.info("-" * 50)

    logger.info("FVM-based physics-informed fine-tuning (delta learning) completed!")

    # =========================================================================
    # Step 10: Save results (delta learning)
    # =========================================================================
    logger.info("Saving results...")

    with torch.no_grad():
        # Get predictions at cell centers (delta model: coords + gnn_uvp -> final)
        model_out = pi_fine_tuner.model(
            pi_fine_tuner.cell_centers_xy, 
            pi_fine_tuner.gnn_uvp_cell
        )
        pred_u_cell = model_out["u"]
        pred_v_cell = model_out["v"]
        pred_p_cell = model_out["p"]
        
        # Also save the deltas (corrections) for analysis
        delta_u = model_out["delta_u"]
        delta_v = model_out["delta_v"]
        delta_p = model_out["delta_p"]

        # Add filtered predictions as cell data
        ugrid_3d_cell.cell_data["filtered_u"] = pred_u_cell.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["filtered_v"] = pred_v_cell.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["filtered_p"] = pred_p_cell.cpu().numpy().flatten()
        
        # Save delta corrections for analysis
        ugrid_3d_cell.cell_data["delta_u"] = delta_u.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["delta_v"] = delta_v.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["delta_p"] = delta_p.cpu().numpy().flatten()
        
        logger.info(f"Delta statistics - Mean |Δu|: {torch.mean(torch.abs(delta_u)).item():.6e}, "
                    f"Mean |Δv|: {torch.mean(torch.abs(delta_v)).item():.6e}, "
                    f"Mean |Δp|: {torch.mean(torch.abs(delta_p)).item():.6e}")

        # =====================================================================
        # Compute and print improvement: GNN vs Refined predictions
        # Using POINT data for fair comparison (same as validation)
        # =====================================================================
        logger.info("=" * 60)
        logger.info("FINAL RESULTS: GNN vs Refined Predictions (wrt Ground Truth)")
        logger.info("=" * 60)
        
        # Get refined predictions at point locations
        model_out_point = pi_fine_tuner.model(
            pi_fine_tuner.coords_point, 
            pi_fine_tuner.gnn_uvp_point
        )
        pred_u_point = model_out_point["u"]
        pred_v_point = model_out_point["v"]
        pred_p_point = model_out_point["p"]
        
        # L2 errors for GNN predictions (at point locations)
        gnn_u_point = pi_fine_tuner.gnn_uvp_point[:, 0:1]
        gnn_v_point = pi_fine_tuner.gnn_uvp_point[:, 1:2]
        gnn_p_point = pi_fine_tuner.gnn_uvp_point[:, 2:3]
        
        gnn_u_error = torch.linalg.norm(pi_fine_tuner.ref_u_point - gnn_u_point) / torch.linalg.norm(pi_fine_tuner.ref_u_point)
        gnn_v_error = torch.linalg.norm(pi_fine_tuner.ref_v_point - gnn_v_point) / torch.linalg.norm(pi_fine_tuner.ref_v_point)
        gnn_p_error = torch.linalg.norm(pi_fine_tuner.ref_p_point - gnn_p_point) / torch.linalg.norm(pi_fine_tuner.ref_p_point)
        
        # L2 errors for refined predictions
        refined_u_error = torch.linalg.norm(pi_fine_tuner.ref_u_point - pred_u_point) / torch.linalg.norm(pi_fine_tuner.ref_u_point)
        refined_v_error = torch.linalg.norm(pi_fine_tuner.ref_v_point - pred_v_point) / torch.linalg.norm(pi_fine_tuner.ref_v_point)
        refined_p_error = torch.linalg.norm(pi_fine_tuner.ref_p_point - pred_p_point) / torch.linalg.norm(pi_fine_tuner.ref_p_point)
        
        # Compute improvement percentages
        u_improvement = ((gnn_u_error - refined_u_error) / gnn_u_error * 100).item()
        v_improvement = ((gnn_v_error - refined_v_error) / gnn_v_error * 100).item()
        p_improvement = ((gnn_p_error - refined_p_error) / gnn_p_error * 100).item()
        
        logger.info(f"--- Velocity u ---")
        logger.info(f"  GNN L2 Error:     {gnn_u_error.item():.6e} ({gnn_u_error.item()*100:.4f}%)")
        logger.info(f"  Refined L2 Error: {refined_u_error.item():.6e} ({refined_u_error.item()*100:.4f}%)")
        logger.info(f"  Improvement:      {u_improvement:+.2f}%")
        
        logger.info(f"--- Velocity v ---")
        logger.info(f"  GNN L2 Error:     {gnn_v_error.item():.6e} ({gnn_v_error.item()*100:.4f}%)")
        logger.info(f"  Refined L2 Error: {refined_v_error.item():.6e} ({refined_v_error.item()*100:.4f}%)")
        logger.info(f"  Improvement:      {v_improvement:+.2f}%")
        
        logger.info(f"--- Pressure p ---")
        logger.info(f"  GNN L2 Error:     {gnn_p_error.item():.6e} ({gnn_p_error.item()*100:.4f}%)")
        logger.info(f"  Refined L2 Error: {refined_p_error.item():.6e} ({refined_p_error.item()*100:.4f}%)")
        logger.info(f"  Improvement:      {p_improvement:+.2f}%")
        
        logger.info("=" * 60)

        # Compute final FVM residuals for saving
        velocity_3d = torch.cat(
            [pred_u_cell, pred_v_cell, torch.zeros_like(pred_u_cell)],
            dim=1,
        )
        pressure_1d = pred_p_cell.squeeze(-1)
        nut_1d = torch.zeros(pi_fine_tuner.n_cells, device=pi_fine_tuner.device, dtype=torch.float32)

        continuity, momentum_x, momentum_y, momentum_z = compute_residuals_warp_batch_autograd(
            velocity_data=velocity_3d,
            pressure_data=pressure_1d,
            nut_data=nut_1d,
            mesh_data=pi_fine_tuner.mesh_data_for_fvm,
            face_data=pi_fine_tuner.face_data,
            batch_cell_indices=pi_fine_tuner.all_cell_indices,
            nu=pi_fine_tuner.nu,
            device=str(pi_fine_tuner.device),
            stokes_flow=True,
        )

        # Save residuals as cell data
        ugrid_3d_cell.cell_data["residual_continuity"] = continuity.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_x"] = momentum_x.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_y"] = momentum_y.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_z"] = momentum_z.cpu().numpy().flatten()

        logger.info(f"Final residuals - Continuity: {torch.mean(continuity**2).item():.6e}, "
                    f"Mom_x: {torch.mean(momentum_x**2).item():.6e}, "
                    f"Mom_y: {torch.mean(momentum_y**2).item():.6e}")

        # Convert cell data back to point data for saving
        ugrid_3d_point = ugrid_3d_cell.cell_data_to_point_data()

        # Save as VTU
        output_path = path.replace(".vtp", "_fvm_delta_filtered.vtu")
        ugrid_3d_point.save(output_path)
        logger.info(f"Saved results to: {output_path}")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
