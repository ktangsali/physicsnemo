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
Physics-Informed Fine-Tuning using FVM Residuals.

Fine-tunes GNN predictions using Finite Volume Method (FVM) residuals
computed via Warp kernels with autodiff support.
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

HAS_MUON = hasattr(torch.optim, "Muon")


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface."""

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")
        self.optimizers = optimizers
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})
        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
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
from extrude_mesh import (
    extract_noslip_edges_2d,
    extrude_lines_to_surface,
)


def compute_sdf_at_points(polydata_2d, query_points_xy):
    """Compute signed distance field (SDF) from no-slip walls to query points."""
    noslip_edges, noslip_lines, edge_is_obstacle, obstacle_centroid = extract_noslip_edges_2d(polydata_2d)
    
    if noslip_lines is None or len(noslip_edges) == 0:
        print("Warning: No no-slip edges found for SDF computation!")
        return np.zeros(len(query_points_xy))
    
    points = np.array(polydata_2d.points)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    domain_centroid = np.mean(points[:, :2], axis=0)
    
    noslip_surface = extrude_lines_to_surface(
        noslip_lines, z_min=-0.5, z_max=0.5, n_layers=1,
        domain_centroid=domain_centroid, obstacle_centroid=obstacle_centroid,
        edge_is_obstacle=edge_is_obstacle
    )
    noslip_surface_tri = noslip_surface.triangulate()
    
    query_points_3d = np.column_stack([
        query_points_xy[:, 0],
        query_points_xy[:, 1],
        np.zeros(len(query_points_xy))
    ])
    query_polydata = pv.PolyData(query_points_3d)
    
    query_with_dist = query_polydata.compute_implicit_distance(noslip_surface_tri, inplace=False)
    sdf = np.tanh(60 * np.array(-1 * query_with_dist.point_data["implicit_distance"]))
    
    print(f"SDF computed: range [{sdf.min():.6f}, {sdf.max():.6f}]")
    return sdf


def identify_boundary_faces(
    face_data, 
    coords_inflow, 
    coords_outflow,
    coords_noslip,
    extrusion_length=1.0, 
    z_tol=0.01
):
    """Identify boundary face centers for applying BCs in the FVM framework."""
    face_neighbor = face_data['face_neighbor']
    face_centers = face_data['face_centers']
    
    boundary_mask = (face_neighbor == -1)
    boundary_indices = np.where(boundary_mask)[0]
    
    boundary_face_centers = face_centers[boundary_indices]
    z_coords = boundary_face_centers[:, 2]
    side_face_mask = (z_coords > z_tol) & (z_coords < extrusion_length - z_tol)
    side_boundary_indices = boundary_indices[side_face_mask]
    side_face_centers = face_centers[side_boundary_indices]
    
    inflow_xy = coords_inflow[:, :2] if coords_inflow.shape[1] > 2 else coords_inflow
    outflow_xy = coords_outflow[:, :2] if coords_outflow.shape[1] > 2 else coords_outflow
    noslip_xy = coords_noslip[:, :2] if coords_noslip.shape[1] > 2 else coords_noslip
    
    tree_inflow = cKDTree(inflow_xy)
    tree_outflow = cKDTree(outflow_xy)
    tree_noslip = cKDTree(noslip_xy)
    
    face_xy = side_face_centers[:, :2]
    dist_inflow, _ = tree_inflow.query(face_xy)
    dist_outflow, _ = tree_outflow.query(face_xy)
    dist_noslip, _ = tree_noslip.query(face_xy)
    
    inflow_mask = (dist_inflow <= dist_outflow) & (dist_inflow <= dist_noslip)
    noslip_mask = (dist_noslip < dist_inflow) & (dist_noslip < dist_outflow)
    
    inflow_face_indices = side_boundary_indices[inflow_mask]
    noslip_face_indices = side_boundary_indices[noslip_mask]
    
    return inflow_face_indices, noslip_face_indices


class DNN(torch.nn.Module):
    """MLP with Fourier feature encoding."""

    def __init__(self, layers, fourier_features=64):
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

        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.layers(x_proj)


class StokesFVMModel(torch.nn.Module):
    """Model for Stokes flow that outputs (u, v, p) given (x, y) coordinates."""

    def __init__(self, layers=[2, 128, 128, 128, 128, 3], fourier_features=64):
        super().__init__()
        self.dnn = DNN(layers, fourier_features)

    def forward(self, coords_xy):
        out = self.dnn(coords_xy)
        return {
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "p": out[:, 2:3],
        }


def extrude_2d_to_3d(polydata_2d, extrusion_length=1.0):
    """Extrude a 2D PolyData mesh to 3D volumetric cells (wedges/hexahedra)."""
    import vtk
    from vtk.util import numpy_support

    points_2d = np.array(polydata_2d.points)
    n_points_2d = len(points_2d)

    if points_2d.shape[1] == 2:
        points_2d = np.column_stack([points_2d, np.zeros(n_points_2d)])
    else:
        points_2d[:, 2] = 0.0

    points_top = points_2d.copy()
    points_top[:, 2] = extrusion_length
    all_points = np.vstack([points_2d, points_top])

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(all_points.astype(np.float64)))

    cells = vtk.vtkCellArray()
    cell_types = []

    VTK_WEDGE = 13
    VTK_HEXAHEDRON = 12

    polys = polydata_2d.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        n_pts = id_list.GetNumberOfIds()

        if n_pts == 3:
            p0, p1, p2 = id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)
            cells.InsertNextCell(6)
            cells.InsertCellPoint(p0)
            cells.InsertCellPoint(p1)
            cells.InsertCellPoint(p2)
            cells.InsertCellPoint(p0 + n_points_2d)
            cells.InsertCellPoint(p1 + n_points_2d)
            cells.InsertCellPoint(p2 + n_points_2d)
            cell_types.append(VTK_WEDGE)
        elif n_pts == 4:
            p0, p1, p2, p3 = id_list.GetId(0), id_list.GetId(1), id_list.GetId(2), id_list.GetId(3)
            cells.InsertNextCell(8)
            cells.InsertCellPoint(p0)
            cells.InsertCellPoint(p1)
            cells.InsertCellPoint(p2)
            cells.InsertCellPoint(p3)
            cells.InsertCellPoint(p0 + n_points_2d)
            cells.InsertCellPoint(p1 + n_points_2d)
            cells.InsertCellPoint(p2 + n_points_2d)
            cells.InsertCellPoint(p3 + n_points_2d)
            cell_types.append(VTK_HEXAHEDRON)
        else:
            raise ValueError(f"Unsupported 2D cell with {n_pts} points")

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(cell_types, cells)

    for i in range(polydata_2d.GetPointData().GetNumberOfArrays()):
        arr = polydata_2d.GetPointData().GetArray(i)
        arr_name = arr.GetName()
        arr_np = numpy_support.vtk_to_numpy(arr)
        arr_combined = np.concatenate([arr_np, arr_np])
        new_arr = numpy_support.numpy_to_vtk(arr_combined)
        new_arr.SetName(arr_name)
        ugrid.GetPointData().AddArray(new_arr)

    return pv.wrap(ugrid)


class PhysicsInformedFineTunerFVM:
    """Physics-informed fine-tuner using FVM residuals computed via Warp kernels."""

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
        ref_u_point=None,
        ref_v_point=None,
        ref_p_point=None,
        coords_point=None,
        gnn_u_point=None,
        gnn_v_point=None,
        gnn_p_point=None,
        sdf_cell=None,
        use_sdf_weighting=True,
    ):
        super().__init__()

        self.device = device
        self.nu = nu
        self.n_cells = n_cells
        self.lr = lr
        self.total_iters = total_iters
        self.use_lr_scheduler = use_lr_scheduler
        
        # SDF-based weighting for physics residuals
        self.use_sdf_weighting = use_sdf_weighting
        if sdf_cell is not None and use_sdf_weighting:
            sdf_tensor = torch.tensor(sdf_cell, dtype=torch.float32).to(device)
            sdf_abs = torch.abs(sdf_tensor)
            sdf_max = torch.max(sdf_abs) + 1e-8
            self.sdf_weights = sdf_abs / sdf_max
            print(f"SDF weighting enabled: weights in [{self.sdf_weights.min():.4f}, {self.sdf_weights.max():.4f}]")
        else:
            self.sdf_weights = None

        self.ref_u_cell = torch.tensor(ref_u_cell).float().to(self.device).reshape(-1, 1)
        self.ref_v_cell = torch.tensor(ref_v_cell).float().to(self.device).reshape(-1, 1)
        self.ref_p_cell = torch.tensor(ref_p_cell).float().to(self.device).reshape(-1, 1)

        self.gnn_u_cell = torch.tensor(gnn_u_cell).float().to(self.device).reshape(-1, 1)
        self.gnn_v_cell = torch.tensor(gnn_v_cell).float().to(self.device).reshape(-1, 1)
        self.gnn_p_cell = torch.tensor(gnn_p_cell).float().to(self.device).reshape(-1, 1)

        self.cell_centers_3d = cell_centers.astype(np.float32)
        self.cell_centers_xy = (
            torch.tensor(cell_centers[:, :2], requires_grad=False).float().to(self.device)
        )
        self.cell_volumes_np = cell_volumes.astype(np.float32)

        face_centers = face_data['face_centers']
        self.coords_inflow_face = (
            torch.tensor(face_centers[inflow_face_indices, :2], requires_grad=False)
            .float().to(self.device)
        )
        self.coords_noslip_face = (
            torch.tensor(face_centers[noslip_face_indices, :2], requires_grad=False)
            .float().to(self.device)
        )
        
        self.has_point_data = coords_point is not None
        if self.has_point_data:
            self.coords_point = (
                torch.tensor(coords_point[:, :2], requires_grad=False).float().to(self.device)
            )
            self.ref_u_point = torch.tensor(ref_u_point).float().to(self.device).reshape(-1, 1)
            self.ref_v_point = torch.tensor(ref_v_point).float().to(self.device).reshape(-1, 1)
            self.ref_p_point = torch.tensor(ref_p_point).float().to(self.device).reshape(-1, 1)
            
            if gnn_u_point is not None:
                self.gnn_u_point = torch.tensor(np.asarray(gnn_u_point).flatten()).float().to(self.device).reshape(-1, 1)
                self.gnn_v_point = torch.tensor(np.asarray(gnn_v_point).flatten()).float().to(self.device).reshape(-1, 1)
                self.gnn_p_point = torch.tensor(np.asarray(gnn_p_point).flatten()).float().to(self.device).reshape(-1, 1)
            else:
                self.gnn_u_point = None
                self.gnn_v_point = None
                self.gnn_p_point = None

        self.face_data = face_data
        self.all_cell_indices = np.arange(n_cells, dtype=np.int32)

        self.mesh_data_for_fvm = {
            "cell_centers": self.cell_centers_3d,
            "cell_volumes": self.cell_volumes_np,
            "n_cells": n_cells,
        }

        self.model = StokesFVMModel(
            layers=[2, 128, 128, 128, 128, 3],
            fourier_features=64,
        ).to(self.device)

        self._setup_optimizer()

        self.scheduler = None
        if self.use_lr_scheduler:
            lr_min = 1e-6
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_iters, eta_min=lr_min
            )

    def _setup_optimizer(self):
        muon_params = [p for p in self.model.parameters() if p.ndim == 2]
        other_params = [p for p in self.model.parameters() if p.ndim != 2]

        if HAS_MUON and len(muon_params) > 0:
            base_opt = torch.optim.AdamW(
                other_params, lr=self.lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1.0e-8,
            ) if other_params else None

            muon_opt = torch.optim.Muon(muon_params, lr=self.lr, weight_decay=0.0)

            if base_opt is not None:
                self.optimizer = CombinedOptimizer(optimizers=[muon_opt, base_opt])
            else:
                self.optimizer = muon_opt
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                fused=True if torch.cuda.is_available() else False,
            )

    def parabolic_inflow(self, y, U_max=0.3):
        u = 4 * U_max * y * (0.4 - y) / (0.4**2)
        v = torch.zeros_like(y)
        return u, v

    def loss(self):
        # Boundary conditions
        results_inflow = self.model(self.coords_inflow_face)
        pred_u_in, pred_v_in = results_inflow["u"], results_inflow["v"]
        u_in, v_in = self.parabolic_inflow(self.coords_inflow_face[:, 1:2])
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        results_noslip = self.model(self.coords_noslip_face)
        pred_u_noslip, pred_v_noslip = results_noslip["u"], results_noslip["v"]
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # Cell-centered predictions
        model_out_cell = self.model(self.cell_centers_xy)
        pred_u_cell = model_out_cell["u"]
        pred_v_cell = model_out_cell["v"]
        pred_p_cell = model_out_cell["p"]

        # Data loss
        loss_u = torch.mean((self.gnn_u_cell - pred_u_cell) ** 2)
        loss_v = torch.mean((self.gnn_v_cell - pred_v_cell) ** 2)
        loss_p = torch.mean((self.gnn_p_cell - pred_p_cell) ** 2)

        # FVM residuals
        velocity_3d = torch.cat([pred_u_cell, pred_v_cell, torch.zeros_like(pred_u_cell)], dim=1)
        pressure_1d = pred_p_cell.squeeze(-1)
        nut_1d = torch.zeros(self.n_cells, device=self.device, dtype=torch.float32)

        continuity, momentum_x, momentum_y, momentum_z = compute_residuals_warp_batch_autograd(
            velocity_data=velocity_3d,
            pressure_data=pressure_1d,
            nut_data=nut_1d,
            mesh_data=self.mesh_data_for_fvm,
            face_data=self.face_data,
            batch_cell_indices=self.all_cell_indices,
            nu=self.nu,
            device=str(self.device),
            stokes_flow=True,
        )

        if self.sdf_weights is not None:
            loss_cont = torch.mean(self.sdf_weights * continuity**2)
            loss_mom_u = torch.mean(self.sdf_weights * momentum_x**2)
            loss_mom_v = torch.mean(self.sdf_weights * momentum_y**2)
        else:
            loss_cont = torch.mean(continuity**2)
            loss_mom_u = torch.mean(momentum_x**2)
            loss_mom_v = torch.mean(momentum_y**2)

        return (
            loss_u, loss_v, loss_p,
            loss_u_in, loss_v_in,
            loss_u_noslip, loss_v_noslip,
            loss_mom_u, loss_mom_v, loss_cont,
        )

    def train_step(self):
        self.model.train()

        (
            loss_u, loss_v, loss_p,
            loss_u_in, loss_v_in,
            loss_u_noslip, loss_v_noslip,
            loss_mom_u, loss_mom_v, loss_cont,
        ) = self.loss()

        loss = (
            1 * loss_u + 1 * loss_v + 1 * loss_p
            + 100 * loss_u_in + 100 * loss_v_in
            + 1000 * loss_u_noslip + 1000 * loss_v_noslip
            + 1000000 * loss_mom_u + 1000000 * loss_mom_v + 1000000 * loss_cont
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss_u, loss_v, loss_p,
            loss_u_in, loss_v_in,
            loss_u_noslip, loss_v_noslip,
            loss_mom_u, loss_mom_v, loss_cont,
        )

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            model_out = self.model(self.coords_point)
            pred_u, pred_v, pred_p = model_out["u"], model_out["v"], model_out["p"]
            error_u = torch.linalg.norm(self.ref_u_point - pred_u) / torch.linalg.norm(self.ref_u_point)
            error_v = torch.linalg.norm(self.ref_v_point - pred_v) / torch.linalg.norm(self.ref_v_point)
            error_p = torch.linalg.norm(self.ref_p_point - pred_p) / torch.linalg.norm(self.ref_p_point)
            wandb.log({
                "test_u_error (%)": error_u.detach().cpu().numpy(),
                "test_v_error (%)": error_v.detach().cpu().numpy(),
                "test_p_error (%)": error_p.detach().cpu().numpy(),
            })
            return error_u, error_v, error_p


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initialize_wandb(
        project="PhysicsNeMo-Launch",
        entity="PhysicsNeMo",
        name="Stokes-FVM-Fine-Tuning",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")
    logger.file_logging()

    path = os.path.join(to_absolute_path(cfg.results_dir), cfg.graph_path)
    logger.info(f"Loading 2D mesh from: {path}")

    # Load data
    (
        ref_u, ref_v, ref_p, gnn_u, gnn_v, gnn_p,
        coords, coords_inflow, coords_outflow, coords_wall, coords_polygon, nu,
    ) = get_dataset(path)
    coords_noslip = np.concatenate([coords_wall, coords_polygon], axis=0)

    logger.info(f"2D mesh: {len(coords)} points")

    # Prepare mesh
    polydata_2d = pv.read(path)
    polydata_2d.point_data["gnn_u"] = gnn_u.flatten()
    polydata_2d.point_data["gnn_v"] = gnn_v.flatten()
    polydata_2d.point_data["gnn_p"] = gnn_p.flatten()
    polydata_2d.point_data["ref_u"] = ref_u.flatten()
    polydata_2d.point_data["ref_v"] = ref_v.flatten()
    polydata_2d.point_data["ref_p"] = ref_p.flatten()

    # Extrude to 3D
    logger.info("Extruding 2D mesh to 3D...")
    ugrid_3d = extrude_2d_to_3d(polydata_2d, extrusion_length=1)
    logger.info(f"3D mesh: {ugrid_3d.n_cells} cells, {ugrid_3d.n_points} points")

    extruded_path = path.replace(".vtp", "_extruded.vtu")
    ugrid_3d.save(extruded_path)

    # Convert to cell data
    ugrid_3d_cell = ugrid_3d.point_data_to_cell_data(pass_point_data=True)
    gnn_u_cell = ugrid_3d_cell.cell_data["gnn_u"].astype(np.float32)
    gnn_v_cell = ugrid_3d_cell.cell_data["gnn_v"].astype(np.float32)
    gnn_p_cell = ugrid_3d_cell.cell_data["gnn_p"].astype(np.float32)
    ref_u_cell = ugrid_3d_cell.cell_data["ref_u"].astype(np.float32)
    ref_v_cell = ugrid_3d_cell.cell_data["ref_v"].astype(np.float32)
    ref_p_cell = ugrid_3d_cell.cell_data["ref_p"].astype(np.float32)

    # Compute geometry
    cell_centers = ugrid_3d_cell.cell_centers().points.astype(np.float32)
    sized = ugrid_3d_cell.compute_cell_sizes(length=False, area=False, volume=True)
    cell_volumes = sized.cell_data["Volume"].astype(np.float32)
    n_cells = ugrid_3d_cell.n_cells

    # Build FVM connectivity
    face_data = build_face_connectivity(ugrid_3d_cell)
    inflow_face_indices, noslip_face_indices = identify_boundary_faces(
        face_data, coords_inflow, coords_outflow, coords_noslip, extrusion_length=1.0
    )

    # Compute SDF
    logger.info("Computing SDF at cell centers...")
    cell_centers_xy = cell_centers[:, :2]
    sdf_cell = compute_sdf_at_points(polydata_2d, cell_centers_xy)

    # Initialize fine-tuner
    use_lr_scheduler = getattr(cfg, 'pi_use_lr_scheduler', True)
    use_sdf_weighting = getattr(cfg, 'use_sdf_weighting', True)
    
    pi_fine_tuner = PhysicsInformedFineTunerFVM(
        device=device,
        gnn_u_cell=gnn_u_cell, gnn_v_cell=gnn_v_cell, gnn_p_cell=gnn_p_cell,
        ref_u_cell=ref_u_cell, ref_v_cell=ref_v_cell, ref_p_cell=ref_p_cell,
        cell_centers=cell_centers, cell_volumes=cell_volumes,
        inflow_face_indices=inflow_face_indices, noslip_face_indices=noslip_face_indices,
        face_data=face_data, n_cells=n_cells, nu=nu,
        lr=cfg.pi_lr, total_iters=cfg.pi_iters, use_lr_scheduler=use_lr_scheduler,
        ref_u_point=ref_u, ref_v_point=ref_v, ref_p_point=ref_p, coords_point=coords,
        gnn_u_point=gnn_u, gnn_v_point=gnn_v, gnn_p_point=gnn_p,
        sdf_cell=sdf_cell, use_sdf_weighting=use_sdf_weighting,
    )

    # Training loop
    logger.info("Starting physics-informed fine-tuning...")

    for iters in range(cfg.pi_iters):
        start_iter_time = time.time()

        (
            loss_u, loss_v, loss_p,
            loss_u_in, loss_v_in,
            loss_u_noslip, loss_v_noslip,
            loss_mom_u, loss_mom_v, loss_cont,
        ) = pi_fine_tuner.train_step()

        if iters % 100 == 0:
            error_u, error_v, error_p = pi_fine_tuner.validation()

            logger.info(f"Iteration: {iters}")
            logger.info(f"Loss u: {loss_u.detach().cpu().numpy():.3e}, v: {loss_v.detach().cpu().numpy():.3e}, p: {loss_p.detach().cpu().numpy():.3e}")
            logger.info(f"Loss inflow: u={loss_u_in.detach().cpu().numpy():.3e}, v={loss_v_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss noslip: u={loss_u_noslip.detach().cpu().numpy():.3e}, v={loss_v_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss FVM: mom_u={loss_mom_u.detach().cpu().numpy():.3e}, mom_v={loss_mom_v.detach().cpu().numpy():.3e}, cont={loss_cont.detach().cpu().numpy():.3e}")
            logger.info(f"Error: u={error_u:.3e}, v={error_v:.3e}, p={error_p:.3e}")
            logger.info(f"LR: {pi_fine_tuner.optimizer.param_groups[0]['lr']:.3e}, Time: {time.time() - start_iter_time:.2f}s")
            logger.info("-" * 50)

    logger.info("Fine-tuning completed!")

    # Save results
    with torch.no_grad():
        model_out = pi_fine_tuner.model(pi_fine_tuner.cell_centers_xy)
        pred_u_cell = model_out["u"]
        pred_v_cell = model_out["v"]
        pred_p_cell = model_out["p"]

        ugrid_3d_cell.cell_data["filtered_u"] = pred_u_cell.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["filtered_v"] = pred_v_cell.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["filtered_p"] = pred_p_cell.cpu().numpy().flatten()

        # Final comparison
        logger.info("=" * 60)
        logger.info("FINAL RESULTS: GNN vs Refined (wrt Ground Truth)")
        logger.info("=" * 60)
        
        model_out_point = pi_fine_tuner.model(pi_fine_tuner.coords_point)
        pred_u_point = model_out_point["u"]
        pred_v_point = model_out_point["v"]
        pred_p_point = model_out_point["p"]
        
        gnn_u_error = torch.linalg.norm(pi_fine_tuner.ref_u_point - pi_fine_tuner.gnn_u_point) / torch.linalg.norm(pi_fine_tuner.ref_u_point)
        gnn_v_error = torch.linalg.norm(pi_fine_tuner.ref_v_point - pi_fine_tuner.gnn_v_point) / torch.linalg.norm(pi_fine_tuner.ref_v_point)
        gnn_p_error = torch.linalg.norm(pi_fine_tuner.ref_p_point - pi_fine_tuner.gnn_p_point) / torch.linalg.norm(pi_fine_tuner.ref_p_point)
        
        refined_u_error = torch.linalg.norm(pi_fine_tuner.ref_u_point - pred_u_point) / torch.linalg.norm(pi_fine_tuner.ref_u_point)
        refined_v_error = torch.linalg.norm(pi_fine_tuner.ref_v_point - pred_v_point) / torch.linalg.norm(pi_fine_tuner.ref_v_point)
        refined_p_error = torch.linalg.norm(pi_fine_tuner.ref_p_point - pred_p_point) / torch.linalg.norm(pi_fine_tuner.ref_p_point)
        
        u_improvement = ((gnn_u_error - refined_u_error) / gnn_u_error * 100).item()
        v_improvement = ((gnn_v_error - refined_v_error) / gnn_v_error * 100).item()
        p_improvement = ((gnn_p_error - refined_p_error) / gnn_p_error * 100).item()
        
        logger.info(f"Velocity u: GNN={gnn_u_error.item():.6e}, Refined={refined_u_error.item():.6e}, Improvement={u_improvement:+.2f}%")
        logger.info(f"Velocity v: GNN={gnn_v_error.item():.6e}, Refined={refined_v_error.item():.6e}, Improvement={v_improvement:+.2f}%")
        logger.info(f"Pressure p: GNN={gnn_p_error.item():.6e}, Refined={refined_p_error.item():.6e}, Improvement={p_improvement:+.2f}%")
        logger.info("=" * 60)

        # Compute final residuals
        velocity_3d = torch.cat([pred_u_cell, pred_v_cell, torch.zeros_like(pred_u_cell)], dim=1)
        pressure_1d = pred_p_cell.squeeze(-1)
        nut_1d = torch.zeros(pi_fine_tuner.n_cells, device=pi_fine_tuner.device, dtype=torch.float32)

        continuity, momentum_x, momentum_y, momentum_z = compute_residuals_warp_batch_autograd(
            velocity_data=velocity_3d, pressure_data=pressure_1d, nut_data=nut_1d,
            mesh_data=pi_fine_tuner.mesh_data_for_fvm, face_data=pi_fine_tuner.face_data,
            batch_cell_indices=pi_fine_tuner.all_cell_indices, nu=pi_fine_tuner.nu,
            device=str(pi_fine_tuner.device), stokes_flow=True,
        )

        ugrid_3d_cell.cell_data["residual_continuity"] = continuity.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_x"] = momentum_x.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_y"] = momentum_y.cpu().numpy().flatten()
        ugrid_3d_cell.cell_data["residual_momentum_z"] = momentum_z.cpu().numpy().flatten()

        ugrid_3d_point = ugrid_3d_cell.cell_data_to_point_data()
        output_path = path.replace(".vtp", "_fvm_filtered.vtu")
        ugrid_3d_point.save(output_path)
        logger.info(f"Saved results to: {output_path}")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
