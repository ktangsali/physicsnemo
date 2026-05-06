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

import os
import urllib.request

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sympy import Function, Number, Symbol
from torch.optim import Adam, lr_scheduler

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from physicsnemo.utils.logging import PythonLogger


class NavierStokes(PDE):
    """Incompressible Navier-Stokes (steady, 2D, rho=1).

    When ``nu`` is a string, it becomes a symbolic field that the network must
    supply -- used for inverse problems where viscosity is unknown.
    """

    def __init__(self, nu=0.01):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        p = Function("p")(x, y)
        if isinstance(nu, str):
            nu = Function(nu)(x, y)
        else:
            nu = Number(nu)

        self.equations = {
            "continuity": u.diff(x) + v.diff(y),
            "momentum_x": (
                u * u.diff(x)
                + v * u.diff(y)
                + p.diff(x)
                - nu * (u.diff(x, 2) + u.diff(y, 2))
            ),
            "momentum_y": (
                u * v.diff(x)
                + v * v.diff(y)
                + p.diff(y)
                - nu * (v.diff(x, 2) + v.diff(y, 2))
            ),
        }


class AdvectionDiffusion(PDE):
    """Advection-diffusion equation (steady, 2D, rho=1).

    When ``D`` is a string, it becomes a symbolic field for inversion.
    """

    def __init__(self, D=0.1):
        self.dim = 2
        x, y = Symbol("x"), Symbol("y")
        c = Function("c")(x, y)
        u = Function("u")(x, y)
        v = Function("v")(x, y)
        if isinstance(D, str):
            D = Function(D)(x, y)
        else:
            D = Number(D)
        self.equations = {
            "advection_diffusion": (
                u * c.diff(x)
                + v * c.diff(y)
                - D * (c.diff(x, 2) + c.diff(y, 2))
            ),
        }


def _download_data(csv_file: str, url: str, log: PythonLogger) -> None:
    """Download CSV if it does not already exist."""
    if not os.path.exists(csv_file):
        log.info(f"Downloading {csv_file} ...")
        urllib.request.urlretrieve(url, csv_file)
    else:
        log.info(f"{csv_file} already exists, skipping download.")


def _load_data(csv_file: str, base_temp: float):
    """Load and normalise the OpenFOAM heat-sink data."""
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    x = data["Points0"].reshape(-1, 1)
    y = data["Points1"].reshape(-1, 1)
    u = data["U0"].reshape(-1, 1)
    v = data["U1"].reshape(-1, 1)
    p = data["p"].reshape(-1, 1)
    c = data["T"].reshape(-1, 1) / base_temp - 1.0
    coords = np.concatenate([x, y], axis=1)
    flow_fields = np.concatenate([u, v, p], axis=1)
    return coords, flow_fields, c


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Inverse PINN: infer viscosity (nu) and diffusivity (D) from data."""
    DistributedManager.initialize()
    dist = DistributedManager()

    log = PythonLogger(name="inverse_pinn")
    log.file_logging()

    # Load data
    _download_data(cfg.data.csv_file, cfg.data.csv_url, log)
    coords, flow_fields, c = _load_data(cfg.data.csv_file, cfg.data.base_temp)
    log.info(f"Loaded {len(coords)} data points")

    # Generate networks
    flow_net = FullyConnected(in_features=2, out_features=3).to(dist.device)
    heat_net = FullyConnected(in_features=2, out_features=1).to(dist.device)
    invert_net_nu = FullyConnected(in_features=2, out_features=1).to(dist.device)
    invert_net_D = FullyConnected(in_features=2, out_features=1).to(dist.device)

    # Define the PDEs using string for coefficients (these will be inferred)
    navier_stokes = NavierStokes(nu="nu")
    advection_diffusion = AdvectionDiffusion(D="D")

    # Define the PhysicsInformer objects
    pi_ns = PhysicsInformer(
        required_outputs=["continuity", "momentum_x", "momentum_y"],
        equations=navier_stokes,
        grad_method="autodiff",
        device=dist.device,
        detach_names=[
            "u", "u__x", "u__x__x", "u__y", "u__y__y",
            "v", "v__x", "v__x__x", "v__y", "v__y__y",
            "p", "p__x", "p__y",
        ],
    )

    pi_ad = PhysicsInformer(
        required_outputs=["advection_diffusion"],
        equations=advection_diffusion,
        grad_method="autodiff",
        device=dist.device,
        detach_names=["u", "v", "c", "c__x", "c__y", "c__x__x", "c__y__y"],
    )

    # Optimizer and Learning Rate Scheduler
    all_params = (
        list(flow_net.parameters())
        + list(heat_net.parameters())
        + list(invert_net_nu.parameters())
        + list(invert_net_D.parameters())
    )
    optimizer = Adam(all_params, lr=cfg.scheduler.initial_lr)

    decay_rate = cfg.scheduler.decay_rate
    decay_steps = cfg.scheduler.decay_steps
    per_step_gamma = decay_rate ** (1.0 / decay_steps)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=per_step_gamma)

    # Training loop
    max_steps = cfg.training.max_steps
    log_freq = cfg.training.log_freq
    batch_size = cfg.training.batch_size

    for step in range(max_steps):
        optimizer.zero_grad()

        idx = np.random.choice(len(coords), size=batch_size, replace=False)
        coords_batch = torch.tensor(
            coords[idx], dtype=torch.float32, device=dist.device
        )
        coords_batch.requires_grad_(True)
        flow_fields_batch = torch.tensor(
            flow_fields[idx], dtype=torch.float32, device=dist.device
        )
        c_batch = torch.tensor(c[idx], dtype=torch.float32, device=dist.device)

        flow_pred = flow_net(coords_batch)
        c_pred = heat_net(coords_batch)
        nu_pred = invert_net_nu(coords_batch)
        D_pred = invert_net_D(coords_batch)

        # Data-fitting loss
        data_loss = torch.nn.functional.mse_loss(
            flow_pred, flow_fields_batch
        ) + torch.nn.functional.mse_loss(c_pred, c_batch)

        # Physics residual loss
        ns_residuals = pi_ns.forward(
            {
                "u": flow_pred[:, 0:1],
                "v": flow_pred[:, 1:2],
                "p": flow_pred[:, 2:3],
                "nu": nu_pred,
                "coordinates": coords_batch,
            }
        )
        ad_residuals = pi_ad.forward(
            {
                "u": flow_pred[:, 0:1],
                "v": flow_pred[:, 1:2],
                "c": c_pred,
                "D": D_pred,
                "coordinates": coords_batch,
            }
        )

        phy_loss = (
            ns_residuals["continuity"] ** 2
            + ns_residuals["momentum_x"] ** 2
            + ns_residuals["momentum_y"] ** 2
            + ad_residuals["advection_diffusion"] ** 2
        ).mean()

        loss = data_loss + phy_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_freq == 0 or step == max_steps - 1:
            mean_nu = nu_pred.mean().item()
            mean_D = D_pred.mean().item()
            log.info(
                f"step {step:6d} | loss={loss.item():.6e} "
                f"data={data_loss.item():.6e} phy={phy_loss.item():.6e} "
                f"| mean_nu={mean_nu:.6f} mean_D={mean_D:.6f} "
                f"| lr={scheduler.get_last_lr()[0]:.6e}"
            )


if __name__ == "__main__":
    main()
