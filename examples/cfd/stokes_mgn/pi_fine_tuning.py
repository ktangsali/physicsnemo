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
Physics-Informed Fine-Tuning with Delta Learning (PINN-based).

This script fine-tunes GNN predictions using PINN-style autodiff residuals.

**Delta Learning Approach:**
Instead of having a data loss, the GNN predictions are provided as INPUT to the model.
The model predicts corrections (Δu, Δv, Δp) that are added to the GNN predictions.
Physics and boundary condition losses are applied to the final result (GNN + delta).

This approach removes the need to balance data loss with physics loss - the physics
constraints directly guide the corrections needed to make GNN predictions physically
consistent.
"""

import os
import time

import hydra
import numpy as np
import torch
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
from typing import Dict

from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.arch import Arch
from sympy import Function, Number, Symbol

from utils import get_dataset, relative_lp_error


class Stokes(PDE):
    """Incompressible Stokes flow"""

    def __init__(self, nu, dim=3):
        # set params
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 2:
            input_variables.pop("z")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # set equations
        self.equations = {}
        self.equations["continuity"] = u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = +p.diff(x) - nu * (
            u.diff(x).diff(x) + u.diff(y).diff(y) + u.diff(z).diff(z)
        )
        self.equations["momentum_y"] = +p.diff(y) - nu * (
            v.diff(x).diff(x) + v.diff(y).diff(y) + v.diff(z).diff(z)
        )
        self.equations["momentum_z"] = +p.diff(z) - nu * (
            w.diff(x).diff(x) + w.diff(y).diff(y) + w.diff(z).diff(z)
        )

        if self.dim == 2:
            self.equations.pop("momentum_z")


class DNN(torch.nn.Module):
    """
    Custom PyTorch model with Fourier features.
    Supports zero-initialization of the final layer for delta learning.
    """

    def __init__(self, layers, fourier_features=64, zero_init_last=False):
        super().__init__()

        # parameters
        self.depth = len(layers) - 1

        # Fourier features
        self.fourier_features = fourier_features
        self.register_buffer(
            "B", 10 * torch.randn((layers[0], fourier_features))
        )  # Random matrix

        # set up layer order dict
        self.activation = torch.nn.GELU

        layer_list = list()
        for i in range(1, self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))

        # Final layer with optional zero initialization
        final_layer = torch.nn.Linear(layers[-2], layers[-1])
        if zero_init_last:
            # Zero-initialize final layer so output starts at ~0 (for delta learning)
            torch.nn.init.zeros_(final_layer.weight)
            torch.nn.init.zeros_(final_layer.bias)
        layer_list.append(("layer_%d" % (self.depth - 1), final_layer))
        
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        # Add Fourier features
        x_proj = torch.matmul(x, self.B)
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # Pass through layers
        out = self.layers(x_proj)
        return out


class MdlsSymDNN(Arch):
    """
    Wrapper model to convert PyTorch model to PhysicsNeMo-Sym model.

    PhysicsNeMo Sym relies on the inputs/outputs of the model being dictionary of tensors.
    This wrapper converts the input dictionary of tensors to a single tensor by
    concatenating them along appropriate dimension before passing them as an input to
    the pytorch model. During the output, the process is reversed,
    the output tensor from pytorch model is split across appropriate dimensions and then
    converted to a dictionary with appropriate keys to produce the final output.

    The model arguments thus become a list of `Key` objects that informs the model
    about the input and output dimensionality of the pytorch model.

    For more details on PhysicsNeMo Sym models, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/simple_training_example.html#using-custom-models-in-physicsnemo
    For more details on Key class, refer:
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.html#module-physicsnemo.sym.key
    """

    def __init__(
        self,
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        layers=[2, 128, 128, 128, 128, 3],
        fourier_features=64,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        self.mdls_model = DNN(layers, fourier_features)

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        # Use concat_input method of the Arch class to convert dict of tensors to
        # a single multi-dimensional tensor. Ref: https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/models/arch.py#L251
        x = self.concat_input(
            dict_tensor,
            self.input_key_dict,
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        out = self.mdls_model(x)
        # Use split_output method of the Arch class to convert a single muli-dimensional
        # tensor to a dict of tensors. Ref: https://github.com/NVIDIA/physicsnemo-sym/blob/main/physicsnemo/sym/models/arch.py#L381
        return self.split_output(out, self.output_key_dict, dim=1)


class MdlsSymDeltaDNN(Arch):
    """
    Delta learning model wrapper for PhysicsNeMo-Sym.
    
    Takes (x, y, u_gnn, v_gnn, p_gnn) as input and predicts corrections (Δu, Δv, Δp).
    The final prediction is: u = u_gnn + α * Δu, v = v_gnn + α * Δv, p = p_gnn + α * Δp.
    
    Features to preserve initial GNN predictions:
    1. Zero-initialized final layer: Model starts with delta ≈ 0
    2. Learnable scaling factor (α): Controls magnitude of corrections, starts small
    3. Skip connection: GNN predictions are directly added to scaled deltas
    """

    def __init__(
        self,
        input_keys=[Key("x"), Key("y"), Key("u_gnn"), Key("v_gnn"), Key("p_gnn")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        layers=[5, 128, 128, 128, 128, 3],  # Input is 5: x, y, u_gnn, v_gnn, p_gnn
        fourier_features=64,
        zero_init=True,
        learnable_scale=True,
        initial_scale=0.1,
    ):
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
        )

        self.mdls_model = DNN(layers, fourier_features, zero_init_last=zero_init)
        
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

    def forward(self, dict_tensor: Dict[str, torch.Tensor]):
        """
        Args:
            dict_tensor: Dictionary with keys 'x', 'y', 'u_gnn', 'v_gnn', 'p_gnn'
            
        Returns:
            Dictionary with 'u', 'v', 'p' (final predictions = GNN + scaled delta)
            Also includes 'delta_u', 'delta_v', 'delta_p', 'alpha_vel', 'alpha_p' for monitoring
        """
        # Concatenate all inputs
        x = self.concat_input(
            dict_tensor,
            self.input_key_dict,
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        
        # Get raw deltas from the network
        delta_raw = self.mdls_model(x)  # [N, 3]
        
        # Extract raw deltas
        delta_u_raw = delta_raw[:, 0:1]
        delta_v_raw = delta_raw[:, 1:2]
        delta_p_raw = delta_raw[:, 2:3]
        
        # Scale deltas (separate scales for velocity and pressure)
        delta_u = self.alpha_vel * delta_u_raw
        delta_v = self.alpha_vel * delta_v_raw
        delta_p = self.alpha_p * delta_p_raw
        
        # Final predictions = GNN + scaled delta (skip connection)
        u_final = dict_tensor["u_gnn"] + delta_u
        v_final = dict_tensor["v_gnn"] + delta_v
        p_final = dict_tensor["p_gnn"] + delta_p
        
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


class PhysicsInformedFineTuner:
    """
    Physics-informed fine-tuner with delta learning.
    
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
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_noslip,
        nu,
        ref_u,
        ref_v,
        ref_p,
        lr=0.001,
        # Delta learning options
        zero_init=True,
        learnable_scale=True,
        initial_scale=0.1,
        delta_reg_weight=0.0,
    ):
        super().__init__()

        self.device = device
        self.nu = nu
        self.delta_reg_weight = delta_reg_weight

        # Reference data for validation
        self.ref_u = torch.tensor(ref_u).float().to(self.device)
        self.ref_v = torch.tensor(ref_v).float().to(self.device)
        self.ref_p = torch.tensor(ref_p).float().to(self.device)

        # GNN predictions - used as MODEL INPUT (not targets)
        self.gnn_u = torch.tensor(gnn_u).float().to(self.device)
        self.gnn_v = torch.tensor(gnn_v).float().to(self.device)
        self.gnn_p = torch.tensor(gnn_p).float().to(self.device)

        self.coords = torch.tensor(coords, requires_grad=True).float().to(self.device)
        self.coords_inflow = (
            torch.tensor(coords_inflow, requires_grad=True).float().to(self.device)
        )
        self.coords_noslip = (
            torch.tensor(coords_noslip, requires_grad=True).float().to(self.device)
        )
        
        # Interpolate GNN predictions to boundary points (inflow, noslip)
        # Using nearest neighbor interpolation
        from scipy.spatial import cKDTree
        coords_np = coords[:, :2] if coords.shape[1] > 2 else coords
        tree = cKDTree(coords_np)
        
        # Inflow boundary GNN values
        inflow_np = coords_inflow[:, :2] if coords_inflow.shape[1] > 2 else coords_inflow
        _, inflow_nearest = tree.query(inflow_np)
        self.gnn_u_inflow = self.gnn_u[inflow_nearest]
        self.gnn_v_inflow = self.gnn_v[inflow_nearest]
        self.gnn_p_inflow = self.gnn_p[inflow_nearest]
        
        # Noslip boundary GNN values
        noslip_np = coords_noslip[:, :2] if coords_noslip.shape[1] > 2 else coords_noslip
        _, noslip_nearest = tree.query(noslip_np)
        self.gnn_u_noslip = self.gnn_u[noslip_nearest]
        self.gnn_v_noslip = self.gnn_v[noslip_nearest]
        self.gnn_p_noslip = self.gnn_p[noslip_nearest]

        # Delta model: takes (x, y, u_gnn, v_gnn, p_gnn), outputs corrections
        self.model = MdlsSymDeltaDNN(
            input_keys=[Key("x"), Key("y"), Key("u_gnn"), Key("v_gnn"), Key("p_gnn")],
            output_keys=[Key("u"), Key("v"), Key("p")],
            layers=[5, 128, 128, 128, 128, 3],
            fourier_features=64,
            zero_init=zero_init,
            learnable_scale=learnable_scale,
            initial_scale=initial_scale,
        ).to(self.device)
        
        print(f"Delta model settings: zero_init={zero_init}, learnable_scale={learnable_scale}, "
              f"initial_scale={initial_scale}, delta_reg_weight={delta_reg_weight}")

        self.node_pde = Stokes(nu=self.nu, dim=2)

        # note: this example uses the PhysicsInformer class from PhysicsNeMo Sym to
        # construct the computational graph. This allows you to leverage PhysicsNeMo Sym's
        # optimized derivative backend to compute the derivatives, along with other
        # benefits like symbolic definition of PDEs and leveraging the PDEs from PhysicsNeMo
        # Sym's PDE module.

        self.phy_informer = PhysicsInformer(
            required_outputs=["continuity", "momentum_x", "momentum_y"],
            equations=self.node_pde,
            grad_method="autodiff",
            device=self.device,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            fused=True if torch.cuda.is_available() else False,
        )

    def parabolic_inflow(self, y, U_max=0.3):
        u = 4 * U_max * y * (0.4 - y) / (0.4**2)
        v = torch.zeros_like(y)
        return u, v

    def loss(self):
        # =================================================================
        # Boundary condition losses (delta model takes GNN predictions as input)
        # =================================================================
        
        # Inflow points
        x_in, y_in = self.coords_inflow[:, 0:1], self.coords_inflow[:, 1:2]
        results_inflow = self.model({
            "x": x_in, 
            "y": y_in,
            "u_gnn": self.gnn_u_inflow,
            "v_gnn": self.gnn_v_inflow,
            "p_gnn": self.gnn_p_inflow,
        })
        pred_u_in, pred_v_in = results_inflow["u"], results_inflow["v"]

        # No-slip points
        x_no_slip, y_no_slip = self.coords_noslip[:, 0:1], self.coords_noslip[:, 1:2]
        results_noslip = self.model({
            "x": x_no_slip, 
            "y": y_no_slip,
            "u_gnn": self.gnn_u_noslip,
            "v_gnn": self.gnn_v_noslip,
            "p_gnn": self.gnn_p_noslip,
        })
        pred_u_noslip, pred_v_noslip = results_noslip["u"], results_noslip["v"]

        # =================================================================
        # Interior points for PDE residuals
        # =================================================================
        x_int, y_int = self.coords[:, 0:1], self.coords[:, 1:2]
        model_out = self.model({
            "x": x_int, 
            "y": y_int,
            "u_gnn": self.gnn_u,
            "v_gnn": self.gnn_v,
            "p_gnn": self.gnn_p,
        })
        
        # Compute PDE residuals on final predictions (GNN + delta)
        results_int = self.phy_informer.forward(
            {
                "coordinates": self.coords,
                "u": model_out["u"],
                "v": model_out["v"],
                "p": model_out["p"],
            }
        )
        pred_mom_u, pred_mom_v, pred_cont = (
            results_int["momentum_x"],
            results_int["momentum_y"],
            results_int["continuity"],
        )
        
        # Track delta magnitudes for monitoring
        delta_u = model_out["delta_u"]
        delta_v = model_out["delta_v"]
        delta_p = model_out["delta_p"]
        delta_u_mag = torch.mean(torch.abs(delta_u)).detach()
        delta_v_mag = torch.mean(torch.abs(delta_v)).detach()
        delta_p_mag = torch.mean(torch.abs(delta_p)).detach()
        
        # Get alpha values for monitoring
        alpha_vel = model_out["alpha_vel"]
        alpha_p = model_out["alpha_p"]

        u_in, v_in = self.parabolic_inflow(self.coords_inflow[:, 1:2])

        # =================================================================
        # Compute losses (NO DATA LOSS - GNN predictions are inputs)
        # =================================================================

        # Inflow boundary condition loss
        loss_u_in = torch.mean((u_in - pred_u_in) ** 2)
        loss_v_in = torch.mean((v_in - pred_v_in) ** 2)

        # Noslip boundary condition loss
        loss_u_noslip = torch.mean(pred_u_noslip**2)
        loss_v_noslip = torch.mean(pred_v_noslip**2)

        # PDE loss
        loss_mom_u = torch.mean(pred_mom_u**2)
        loss_mom_v = torch.mean(pred_mom_v**2)
        loss_cont = torch.mean(pred_cont**2)
        
        # Delta regularization loss (penalize large corrections)
        loss_delta_reg = torch.mean(delta_u**2 + delta_v**2 + delta_p**2)

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

    def train(self):
        """PINN based fine-tuning with delta learning (no data loss)."""
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

        # Weighted loss combination (NO DATA LOSS - GNN predictions are inputs)
        # BC losses, physics losses, and optional delta regularization
        loss = (
            10000 * loss_u_in
            + 10000 * loss_v_in
            + 100 * loss_u_noslip
            + 100 * loss_v_noslip
            + 10000 * loss_mom_u
            + 10000 * loss_mom_v
            + 10000 * loss_cont
            + self.delta_reg_weight * loss_delta_reg
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        """Validation during the PINN fine-tuning step (delta model)."""
        self.model.eval()
        with torch.no_grad():
            x_int, y_int = self.coords[:, 0:1], self.coords[:, 1:2]
            model_out = self.model({
                "x": x_int, 
                "y": y_int,
                "u_gnn": self.gnn_u,
                "v_gnn": self.gnn_v,
                "p_gnn": self.gnn_p,
            })
            pred_u, pred_v, pred_p = (
                model_out["u"],
                model_out["v"],
                model_out["p"],
            )
            error_u = torch.linalg.norm(self.ref_u - pred_u) / torch.linalg.norm(
                self.ref_u
            )
            error_v = torch.linalg.norm(self.ref_v - pred_v) / torch.linalg.norm(
                self.ref_v
            )
            error_p = torch.linalg.norm(self.ref_p - pred_p) / torch.linalg.norm(
                self.ref_p
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

    # initialize loggers
    initialize_wandb(
        project="PhysicsNeMo-Launch",
        entity="PhysicsNeMo",
        name="Stokes-PINN-Delta-Learning",
        group="Stokes-DDP-Group",
        mode=cfg.wandb_mode,
    )

    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    # Get dataset
    path = os.path.join(to_absolute_path(cfg.results_dir), cfg.graph_path)

    # get_dataset() function here provides the true values (ref_*) and the gnn
    # predictions (gnn_*) along with other data required for the PINN training.
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

    # Get delta learning options from config (with sensible defaults)
    zero_init = getattr(cfg, 'delta_zero_init', True)
    learnable_scale = getattr(cfg, 'delta_learnable_scale', False)
    initial_scale = getattr(cfg, 'delta_initial_scale', 0.1)
    delta_reg_weight = getattr(cfg, 'delta_reg_weight', 0.1)

    logger.info("Initializing PINN fine-tuner with delta learning...")
    logger.info("GNN predictions will be used as MODEL INPUT, not as targets.")
    logger.info("The model learns corrections (deltas) to make predictions physics-consistent.")

    # Initialize model
    pi_fine_tuner = PhysicsInformedFineTuner(
        device,
        gnn_u,
        gnn_v,
        gnn_p,
        coords,
        coords_inflow,
        coords_noslip,
        nu,
        ref_u,
        ref_v,
        ref_p,
        lr=cfg.pi_lr,
        zero_init=zero_init,
        learnable_scale=learnable_scale,
        initial_scale=initial_scale,
        delta_reg_weight=delta_reg_weight,
    )

    logger.info("Starting PINN-based physics-informed fine-tuning (delta learning)...")
    logger.info("No data loss - GNN predictions are inputs, model learns corrections.")
    
    for iters in range(cfg.pi_iters):
        # Start timing the iteration
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
        ) = pi_fine_tuner.train()

        if iters % 100 == 0:
            error_u, error_v, error_p = pi_fine_tuner.validation()

            # Print losses
            logger.info(f"Iteration: {iters}")
            logger.info(f"--- BC Losses ---")
            logger.info(f"Loss u_in: {loss_u_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v_in: {loss_v_in.detach().cpu().numpy():.3e}")
            logger.info(f"Loss u noslip: {loss_u_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"Loss v noslip: {loss_v_noslip.detach().cpu().numpy():.3e}")
            logger.info(f"--- Physics Losses (PINN) ---")
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

            # Print iteration time
            end_iter_time = time.time()
            logger.info(
                f"This iteration took {end_iter_time - start_iter_time:.2f} seconds"
            )
            logger.info("-" * 50)  # Add a separator for clarity

    logger.info("PINN-based physics-informed fine-tuning (delta learning) completed!")

    # Save results
    # Final inference call after fine-tuning predictions using the delta model
    with torch.no_grad():
        x_int_inf, y_int_inf = (
            pi_fine_tuner.coords[:, 0:1],
            pi_fine_tuner.coords[:, 1:2],
        )
        results_int_inf = pi_fine_tuner.model({
            "x": x_int_inf, 
            "y": y_int_inf,
            "u_gnn": pi_fine_tuner.gnn_u,
            "v_gnn": pi_fine_tuner.gnn_v,
            "p_gnn": pi_fine_tuner.gnn_p,
        })
        pred_u_inf, pred_v_inf, pred_p_inf = (
            results_int_inf["u"],
            results_int_inf["v"],
            results_int_inf["p"],
        )
        
        # Also get deltas for saving
        delta_u_inf = results_int_inf["delta_u"]
        delta_v_inf = results_int_inf["delta_v"]
        delta_p_inf = results_int_inf["delta_p"]

        # =====================================================================
        # Compute and print improvement: GNN vs Refined predictions
        # =====================================================================
        logger.info("=" * 60)
        logger.info("FINAL RESULTS: GNN vs Refined Predictions (wrt Ground Truth)")
        logger.info("=" * 60)
        
        # L2 errors for GNN predictions
        gnn_u_error = torch.linalg.norm(pi_fine_tuner.ref_u - pi_fine_tuner.gnn_u) / torch.linalg.norm(pi_fine_tuner.ref_u)
        gnn_v_error = torch.linalg.norm(pi_fine_tuner.ref_v - pi_fine_tuner.gnn_v) / torch.linalg.norm(pi_fine_tuner.ref_v)
        gnn_p_error = torch.linalg.norm(pi_fine_tuner.ref_p - pi_fine_tuner.gnn_p) / torch.linalg.norm(pi_fine_tuner.ref_p)
        
        # L2 errors for refined predictions
        refined_u_error = torch.linalg.norm(pi_fine_tuner.ref_u - pred_u_inf) / torch.linalg.norm(pi_fine_tuner.ref_u)
        refined_v_error = torch.linalg.norm(pi_fine_tuner.ref_v - pred_v_inf) / torch.linalg.norm(pi_fine_tuner.ref_v)
        refined_p_error = torch.linalg.norm(pi_fine_tuner.ref_p - pred_p_inf) / torch.linalg.norm(pi_fine_tuner.ref_p)
        
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

        pred_u_inf = pred_u_inf.detach().cpu().numpy()
        pred_v_inf = pred_v_inf.detach().cpu().numpy()
        pred_p_inf = pred_p_inf.detach().cpu().numpy()
        delta_u_inf = delta_u_inf.detach().cpu().numpy()
        delta_v_inf = delta_v_inf.detach().cpu().numpy()
        delta_p_inf = delta_p_inf.detach().cpu().numpy()
        
        logger.info(f"Delta statistics - Mean |Δu|: {np.mean(np.abs(delta_u_inf)):.6e}, "
                    f"Mean |Δv|: {np.mean(np.abs(delta_v_inf)):.6e}, "
                    f"Mean |Δp|: {np.mean(np.abs(delta_p_inf)):.6e}")

        polydata = pv.read(path)
        polydata["filtered_u"] = pred_u_inf
        polydata["filtered_v"] = pred_v_inf
        polydata["filtered_p"] = pred_p_inf
        polydata["delta_u"] = delta_u_inf
        polydata["delta_v"] = delta_v_inf
        polydata["delta_p"] = delta_p_inf
        
        # Save to new file with delta suffix
        output_path = path.replace(".vtp", "_delta_filtered.vtp")
        polydata.save(output_path)
        logger.info(f"Saved results to: {output_path}")

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
