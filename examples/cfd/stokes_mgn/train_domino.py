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

import math
import time
import os

import hydra
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast

try:
    import apex
except:
    pass

from physicsnemo.launch.logging import (
    PythonLogger,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from physicsnemo.models.transolver import Transolver

from utils import relative_lp_error

import zarr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from zarr_io_example import load_fvm_data, compute_sdf_2d
from collections import OrderedDict
from scipy.spatial import cKDTree

from utils import relative_lp_error

from fvm_residuals import compute_residuals_warp_batch_autograd


def identify_boundary_faces_from_zarr(face_neighbor, face_centers, cell_centers, stl_coords, x_tol=0.01):
    """
    Identify inlet and no-slip boundary face centers from zarr FVM data.
    
    Args:
        face_neighbor: [n_faces] array, -1 for boundary faces
        face_centers: [n_faces, 3] face center coordinates
        cell_centers: [n_cells, 3] cell center coordinates  
        stl_coords: [n_stl, 3] STL obstacle coordinates
        x_tol: tolerance for identifying inlet (faces near x_min)
    
    Returns:
        inlet_face_centers: [n_inlet, 2] xy coordinates of inlet faces
        noslip_face_centers: [n_noslip, 2] xy coordinates of no-slip faces
    """
    # Find boundary faces
    boundary_mask = (face_neighbor == -1)
    boundary_face_centers = face_centers[boundary_mask]
    
    # Domain bounds
    x_min = cell_centers[:, 0].min()
    x_max = cell_centers[:, 0].max()
    y_min = cell_centers[:, 1].min()
    y_max = cell_centers[:, 1].max()
    
    # Classify boundary faces:
    # - Inlet: x near x_min
    # - Outlet: x near x_max (we don't constrain outlet)
    # - Top/bottom walls: y near y_min or y_max
    # - Obstacle: close to STL coordinates
    
    inlet_mask = boundary_face_centers[:, 0] < (x_min + x_tol)
    outlet_mask = boundary_face_centers[:, 0] > (x_max - x_tol)
    top_wall_mask = boundary_face_centers[:, 1] > (y_max - x_tol)
    bottom_wall_mask = boundary_face_centers[:, 1] < (y_min + x_tol)
    
    # Build KD-tree for STL obstacle proximity
    stl_2d = stl_coords[:, :2]
    stl_unique = np.unique(stl_2d, axis=0)
    tree = cKDTree(stl_unique)
    
    # Find faces close to obstacle
    boundary_xy = boundary_face_centers[:, :2]
    dist_to_stl, _ = tree.query(boundary_xy)
    obstacle_mask = dist_to_stl < x_tol * 5  # slightly larger tolerance for obstacle
    
    # No-slip = top wall + bottom wall + obstacle (excluding inlet/outlet)
    noslip_mask = (top_wall_mask | bottom_wall_mask | obstacle_mask) & ~inlet_mask & ~outlet_mask
    
    inlet_face_centers = boundary_face_centers[inlet_mask, :2]
    noslip_face_centers = boundary_face_centers[noslip_mask, :2]
    
    return inlet_face_centers, noslip_face_centers


def parabolic_inflow(y, y_min, y_max, U_max=0.3):
    """
    Compute parabolic velocity profile for channel inlet.
    
    Args:
        y: y-coordinates of inlet faces
        y_min: minimum y coordinate of channel
        y_max: maximum y coordinate of channel
        U_max: maximum velocity at center
    
    Returns:
        u, v: velocity components (v is zero for inlet)
    """
    H = y_max - y_min  # channel height
    y_norm = y - y_min  # shift to start at 0
    u = 4 * U_max * y_norm * (H - y_norm) / (H ** 2)
    v = torch.zeros_like(y)
    return u, v


class FVMZarrDataset(Dataset):
    def __init__(self, root_dir, max_samples=None):
        self.zarr_paths = sorted(
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if d.endswith(".zarr")
        )
        # Limit number of samples if specified
        if max_samples is not None and max_samples < len(self.zarr_paths):
            self.zarr_paths = self.zarr_paths[:max_samples]

    def __len__(self):
        return len(self.zarr_paths)

    def __getitem__(self, idx):
        data = load_fvm_data(self.zarr_paths[idx])

        # Compute SDF from STL coordinates
        sdf = compute_sdf_2d(data['cell_centers'], data['stl_coordinates'])
        data['sdf'] = sdf.astype(np.float32)
        
        # Identify boundary faces for BC losses
        inlet_face_centers, noslip_face_centers = identify_boundary_faces_from_zarr(
            data['face_neighbor'],
            data['face_centers'],
            data['cell_centers'],
            data['stl_coordinates'],
        )
        data['inlet_face_centers'] = inlet_face_centers.astype(np.float32)
        data['noslip_face_centers'] = noslip_face_centers.astype(np.float32)
        
        # Store domain bounds for parabolic profile
        data['y_min'] = float(data['cell_centers'][:, 1].min())
        data['y_max'] = float(data['cell_centers'][:, 1].max())

        # convert numpy -> torch
        data = {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }

        return data

def fvm_collate_fn(batch):
    # batch is a list of dicts
    # we do NOT stack — return as-is
    return batch

class TransolverWrapper(torch.nn.Module):
    """
    Wrapper around Transolver to handle (coords, sdf) input format.
    
    For irregular meshes, Transolver takes:
    - embedding: [B, N, embedding_dim] - positional embeddings (x, y coords)
    - functional_input: [B, N, functional_dim] - functional input (SDF)
    """

    def __init__(self, cfg):
        super().__init__()
        
        # For irregular mesh (unstructured data):
        # - embedding_dim=2 for (x, y) coordinates
        # - functional_dim=1 for SDF
        # - out_dim=3 for (u, v, p)
        self.transolver = Transolver(
            structured_shape=None,  # Irregular/unstructured mesh
            embedding_dim=2,        # (x, y) coordinates as positional embedding
            functional_dim=1,       # SDF as functional input
            out_dim=3,              # (u, v, p) output
            n_layers=cfg.model.n_layers,
            n_hidden=cfg.model.n_hidden,
            dropout=cfg.model.dropout,
            n_head=cfg.model.n_head,
            act=cfg.model.act,
            mlp_ratio=cfg.model.mlp_ratio,
            slice_num=cfg.model.slice_num,
            unified_pos=False,      # Must be False for irregular mesh
            ref=cfg.model.ref,
            use_te=False,           # Disable transformer engine for compatibility
        )

    def forward(self, coords, sdf=None):
        """
        Args:
            coords: [N, 2] or [N, 3] coordinates (only x, y used)
            sdf: [N] or [N, 1] signed distance field
        
        Returns:
            [N, 3] predictions for (u, v, p)
        """
        # Take only x, y coordinates as embedding
        if coords.shape[-1] == 3:
            coords = coords[:, :2]
        
        # Prepare SDF as functional input
        if sdf is not None:
            if sdf.dim() == 1:
                sdf = sdf.unsqueeze(-1)  # [N] -> [N, 1]
        else:
            sdf = torch.zeros(coords.shape[0], 1, device=coords.device)
        
        # Add batch dimension: [N, C] -> [1, N, C]
        embedding = coords.unsqueeze(0)      # [1, N, 2] - positional embedding
        functional = sdf.unsqueeze(0)        # [1, N, 1] - functional input
        
        # Forward through Transolver (embedding first, then functional)
        out = self.transolver(embedding, functional)  # [1, N, 3]
        
        # Remove batch dimension: [1, N, 3] -> [N, 3]
        out = out.squeeze(0)
        
        return out


def dict_to_device(sample, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in sample.items()
    }

@torch.no_grad()
def validation_step(model, dataloader):
    error_keys = ["u", "v", "p"]
    errors = {key: 0 for key in error_keys}

    model.eval()
    for batch in dataloader:
        sample = dict_to_device(batch[0], torch.device("cuda"))
        cell_centers = sample["cell_centers"]
        sdf = sample["sdf"]
        true = torch.stack([sample["u"], sample["v"], sample["p"]], axis=1)
        pred = model(cell_centers, sdf)
    
        for index, key in enumerate(error_keys):
            pred_val = pred[:, index]
            true_val = true[:, index]
            # Normalized RMSE: RMSE / range
            rmse = torch.sqrt(torch.mean((pred_val - true_val) ** 2))
            val_range = true_val.max() - true_val.min()
            nrmse = (rmse / val_range) * 100  # as percentage
            errors[key] += nrmse.item()

    for key in error_keys:
        errors[key] = errors[key] / len(dataloader)
        print(f"validation nRMSE_{key} (%): {errors[key]:.2f}")


@torch.no_grad()
def visualize_predictions(model, samples, epoch, output_path='validation_vis.png'):
    """
    Visualize predictions vs ground truth for 3 samples.
    
    Creates a 3x6 grid: rows are samples, columns are u, u_pred, v, v_pred, p, p_pred
    """
    model.eval()
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    field_names = ['u', 'v', 'p']
    
    for row, sample in enumerate(samples):
        cell_centers = sample["cell_centers"]
        sdf = sample["sdf"]
        
        # Get predictions
        pred = model(cell_centers, sdf)
        
        # Get coordinates for plotting
        x = cell_centers[:, 0].cpu().numpy()
        y = cell_centers[:, 1].cpu().numpy()
        
        for col_idx, field in enumerate(field_names):
            true_val = sample[field].cpu().numpy()
            pred_val = pred[:, col_idx].cpu().numpy()
            
            # Shared color limits between true and pred
            vmin = min(true_val.min(), pred_val.min())
            vmax = max(true_val.max(), pred_val.max())
            
            # True value
            ax = axes[row, col_idx * 2]
            sc = ax.scatter(x, y, c=true_val, cmap='viridis', s=1, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal')
            ax.set_title(f'Sample {row+1}: {field} (true)')
            plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Predicted value
            ax = axes[row, col_idx * 2 + 1]
            sc = ax.scatter(x, y, c=pred_val, cmap='viridis', s=1, vmin=vmin, vmax=vmax)
            ax.set_aspect('equal')
            ax.set_title(f'Sample {row+1}: {field} (pred)')
            plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05)
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config_domino")
def main(cfg: DictConfig) -> None:

    logger = PythonLogger("main")  # General python logger

    # Transolver model for unstructured point cloud data
    # Input: (x, y, sdf) -> 3 channels
    # Output: (u, v, p) -> 3 channels
    model = TransolverWrapper(cfg).to(torch.device("cuda"))

    # Get sample limits from config
    num_train = getattr(cfg, 'num_training_samples', None)
    num_val = getattr(cfg, 'num_validation_samples', None)
    num_test = getattr(cfg, 'num_test_samples', None)
    
    train_dataset = FVMZarrDataset(
        to_absolute_path("physics-curated/train/"), 
        max_samples=num_train
    )
    logger.info(f"Training samples: {len(train_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=fvm_collate_fn,
        pin_memory=True
    )
    
    test_dataset = FVMZarrDataset(
        to_absolute_path("physics-curated/test/"), 
        max_samples=num_test
    )
    logger.info(f"Test samples: {len(test_dataset)}")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=fvm_collate_fn,
        pin_memory=True
    )
    
    # Get 3 fixed samples for visualization (same samples every epoch)
    vis_samples = [
        dict_to_device(test_dataset[i], torch.device("cuda")) 
        for i in range(min(3, len(test_dataset)))
    ]

    optimizer = apex.optimizers.FusedAdam(
        model.parameters(), lr=cfg.lr
    )
    final_lr_multiplier = 0.01
    lr_decay_rate = math.pow(final_lr_multiplier, 1.0 / cfg.epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=lr_decay_rate,
    )

    # Get BC config with defaults
    add_bc_loss = getattr(cfg, 'add_bc_loss', False)
    inlet_weight = getattr(cfg, 'inlet_weight', 100.0)
    noslip_weight = getattr(cfg, 'noslip_weight', 1000.0)
    U_max = getattr(cfg, 'U_max', 0.3)
    
    # Get physics loss config with defaults
    add_physics_loss = getattr(cfg, 'add_physics_loss', False)
    continuity_weight = getattr(cfg, 'continuity_weight', 1e13)
    momentum_weight = getattr(cfg, 'momentum_weight', 1e13)
    nu = getattr(cfg, 'nu', 0.001)  # Kinematic viscosity for Stokes flow

    for epoch in range(cfg.epochs):
        # Accumulate losses for epoch average
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_cont_loss = 0.0
        epoch_mom_loss = 0.0
        epoch_inlet_loss = 0.0
        epoch_noslip_loss = 0.0
        n_batches = 0
        
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            sample = dict_to_device(batch[0], torch.device("cuda"))
            cell_centers = sample["cell_centers"]
            sdf = sample["sdf"]
            true = torch.stack([sample["u"], sample["v"], sample["p"]], axis=1)
            pred = model(cell_centers, sdf)
            data_loss = torch.mean((true - pred)**2)
            
            loss = data_loss
            
            # Physics loss (continuity + momentum via FVM)
            if add_physics_loss:
                # Prepare velocity [n_cells, 3] and pressure [n_cells]
                pred_velocity_3d = torch.stack([pred[:,0], pred[:,1], torch.zeros_like(pred[:,0])], dim=1)
                pred_pressure = pred[:, 2]
                nut_zeros = torch.zeros_like(pred_pressure)  # No turbulent viscosity for Stokes
                
                # Build mesh_data dict for FVM
                mesh_data = {
                    'cell_centers': sample["cell_centers"].cpu().numpy(),
                    'cell_volumes': sample["cell_volumes"].cpu().numpy(),
                    'n_cells': sample["n_cells"],
                }
                
                # Build face_data dict for FVM
                face_data = {
                    'face_owner': sample["face_owner"].cpu().numpy(),
                    'face_neighbor': sample["face_neighbor"].cpu().numpy(),
                    'face_area': sample["face_area"].cpu().numpy(),
                    'face_normal': sample["face_normal"].cpu().numpy(),
                    'face_centers': sample["face_centers"].cpu().numpy(),
                    'n_faces': sample["n_faces"],
                }
                
                # Compute all cells
                batch_cell_indices = np.arange(sample["n_cells"], dtype=np.int32)
                
                # Compute FVM residuals with autograd support
                continuity, momentum_x, momentum_y, momentum_z = compute_residuals_warp_batch_autograd(
                    velocity_data=pred_velocity_3d,
                    pressure_data=pred_pressure,
                    nut_data=nut_zeros,
                    mesh_data=mesh_data,
                    face_data=face_data,
                    batch_cell_indices=batch_cell_indices,
                    nu=nu,
                    device='cuda:0',
                    stokes_flow=True,  # Use Stokes equations (no convection)
                )
                
                # Compute losses
                continuity_loss = torch.mean(continuity**2) * continuity_weight
                momentum_loss = (torch.mean(momentum_x**2) + torch.mean(momentum_y**2)) * momentum_weight
                
                # Check physics gradients every 5th epoch
                if epoch % 5 == 0 and i == 0:
                    optimizer.zero_grad()
                    (continuity_loss + momentum_loss).backward(retain_graph=True)
                    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                    print(f"Physics grad norm: {grad_norm:.2e}")
                    optimizer.zero_grad()

                loss = loss + continuity_loss + momentum_loss
                epoch_cont_loss += continuity_loss.item()
                epoch_mom_loss += momentum_loss.item()
            
            # Boundary condition losses
            if add_bc_loss:
                inlet_coords = sample["inlet_face_centers"]  # [N, 2] already
                noslip_coords = sample["noslip_face_centers"]  # [N, 2] already
                y_min = sample["y_min"]
                y_max = sample["y_max"]
                stl_coords_np = sample["stl_coordinates"].cpu().numpy()
                
                # For inlet faces
                if len(inlet_coords) > 0:
                    # Compute SDF at inlet face centers (need 3D for compute_sdf_2d)
                    inlet_coords_3d_np = np.column_stack([
                        inlet_coords.cpu().numpy(),
                        np.zeros(len(inlet_coords))
                    ])
                    inlet_sdf = compute_sdf_2d(inlet_coords_3d_np, stl_coords_np)
                    inlet_sdf = torch.from_numpy(inlet_sdf.astype(np.float32)).to(inlet_coords.device)
                    
                    # Model now takes 2D coords directly
                    pred_inlet = model(inlet_coords, inlet_sdf)
                    pred_u_in, pred_v_in = pred_inlet[:, 0], pred_inlet[:, 1]
                    
                    # Target parabolic profile
                    u_in, v_in = parabolic_inflow(inlet_coords[:, 1], y_min, y_max, U_max=U_max)
                    
                    inlet_loss = (torch.mean((u_in - pred_u_in)**2) + torch.mean((v_in - pred_v_in)**2)) * inlet_weight
                    loss = loss + inlet_loss
                    epoch_inlet_loss += inlet_loss.item()
                
                # For no-slip faces
                if len(noslip_coords) > 0:
                    # Compute SDF at no-slip face centers
                    noslip_coords_3d_np = np.column_stack([
                        noslip_coords.cpu().numpy(),
                        np.zeros(len(noslip_coords))
                    ])
                    noslip_sdf = compute_sdf_2d(noslip_coords_3d_np, stl_coords_np)
                    noslip_sdf = torch.from_numpy(noslip_sdf.astype(np.float32)).to(noslip_coords.device)
                    
                    # Model now takes 2D coords directly
                    pred_noslip = model(noslip_coords, noslip_sdf)
                    pred_u_noslip, pred_v_noslip = pred_noslip[:, 0], pred_noslip[:, 1]
                    
                    # No-slip: u = v = 0
                    noslip_loss = (torch.mean(pred_u_noslip**2) + torch.mean(pred_v_noslip**2)) * noslip_weight
                    loss = loss + noslip_loss
                    epoch_noslip_loss += noslip_loss.item()
            
            epoch_loss += loss.item()
            epoch_data_loss += data_loss.item()
            n_batches += 1
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Print epoch averages
        avg_loss = epoch_loss / n_batches
        avg_data = epoch_data_loss / n_batches
        loss_str = f"Epoch: {epoch}, Loss: {avg_loss:.2e}, Data: {avg_data:.2e}"
        
        if add_physics_loss:
            avg_cont = epoch_cont_loss / n_batches
            avg_mom = epoch_mom_loss / n_batches
            loss_str += f", Cont: {avg_cont:.2e}, Mom: {avg_mom:.2e}"
        
        if add_bc_loss:
            avg_inlet = epoch_inlet_loss / n_batches
            avg_noslip = epoch_noslip_loss / n_batches
            loss_str += f", Inlet: {avg_inlet:.2e}, NoSlip: {avg_noslip:.2e}"
        
        print(loss_str)
        print("Validation Errors")
        validation_step(model, test_dataloader)
        
        # Visualize predictions every 5 epochs
        if epoch % 5 == 0:
            visualize_predictions(model, vis_samples, epoch, cfg.validation_image)

if __name__ == "__main__":
    main()