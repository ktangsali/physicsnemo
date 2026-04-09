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

"""Variational Gaussian Process head for uncertainty-aware scalar regression.

Provides :class:`VariationalGPHead`, a module that can be attached to any
neural-network encoder to produce calibrated uncertainty estimates for a
scalar target.  Built on GPyTorch's variational inference machinery with
inducing points.

Key design choices
------------------
* **Float64 GP internals** — Short lengthscales on L2-normalised embeddings
  make the inducing-point covariance matrix K_uu ill-conditioned in float32.
  All GP computations (kernel, variational strategy, likelihood) run in
  float64; inputs are upcast on entry and outputs are downcast on exit so
  gradients flow through the encoder seamlessly.
* **Optional DKL feature extractor** — A small MLP can be inserted between
  the encoder embedding and the GP kernel (Deep Kernel Learning).  The MLP
  runs in float32 for speed.
* **Matérn-5/2 ARD kernel** — A smooth, twice-differentiable kernel with
  per-dimension lengthscales (Automatic Relevance Determination).

Requires ``gpytorch`` — install via ``pip install gpytorch`` or use the
``uq-extras`` optional dependency group.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )
    from gpytorch.mlls import VariationalELBO

    _GPYTORCH_AVAILABLE = True
except ImportError:
    _GPYTORCH_AVAILABLE = False


def _require_gpytorch() -> None:
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "VariationalGPHead requires gpytorch.  "
            "Install it with: pip install gpytorch"
        )


class _VariationalGPLayer(ApproximateGP):
    """Low-level variational GP with Matérn-5/2 ARD kernel.

    This is an internal building block used by :class:`VariationalGPHead`.
    Users should not need to instantiate it directly.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Initial inducing point locations of shape ``(M, D)``.
    input_dim : int
        Dimensionality of each input (must match last dim of
        *inducing_points*).
    lengthscale_range : tuple[float, float]
        Hard interval constraint on per-dimension lengthscales.
    lengthscale_prior : tuple[float, float] | None
        ``(concentration, rate)`` for a Gamma prior on lengthscales.
    outputscale_prior : tuple[float, float] | None
        ``(concentration, rate)`` for a Gamma prior on the output scale.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        input_dim: int = 32,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
    ) -> None:
        _require_gpytorch()
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        ls_constraint = gpytorch.constraints.Interval(*lengthscale_range)
        ls_prior_obj = None
        if lengthscale_prior is not None:
            ls_prior_obj = gpytorch.priors.GammaPrior(*lengthscale_prior)

        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=input_dim,
            lengthscale_constraint=ls_constraint,
            lengthscale_prior=ls_prior_obj,
        )

        os_prior_obj = None
        if outputscale_prior is not None:
            os_prior_obj = gpytorch.priors.GammaPrior(*outputscale_prior)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=os_prior_obj,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class VariationalGPHead(nn.Module):
    """Variational GP head that operates entirely in float64.

    Attach this module to any encoder that produces fixed-size embeddings to
    obtain calibrated uncertainty estimates for a scalar regression target.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input embedding vector. Default is 32.
    n_inducing : int, optional
        Number of inducing points for the variational approximation.
        Default is 64.
    n_train : int
        Total number of training examples (required for the ELBO
        normalisation constant).
    inducing_points : torch.Tensor | None, optional
        Initial inducing point locations ``(M, D)``.  If *None*, random
        normal points are used. Default is ``None``.
    lengthscale_range : tuple[float, float], optional
        Hard interval constraint ``[lo, hi]`` on per-dimension ARD
        lengthscales. Default is ``(0.01, 10.0)``.
    lengthscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on lengthscales,
        e.g. ``(3.0, 6.0)`` gives mean 0.5. Default is ``None``.
    outputscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on the output scale,
        e.g. ``(2.0, 0.5)`` gives mean 4.0. Default is ``None``.
    mlp_hidden : list[int] | None, optional
        Hidden layer sizes for an optional DKL feature extractor MLP
        inserted before the GP kernel.  ``None`` means the embedding
        feeds the GP directly. Default is ``None``.

    Attributes
    ----------
    gp_layer : _VariationalGPLayer
        The variational GP (float64).
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Observation noise model (float64).
    mll : gpytorch.mlls.VariationalELBO
        Marginal log-likelihood objective.
    feature_extractor : nn.Sequential | None
        Optional DKL MLP (float32).

    Examples
    --------
    >>> head = VariationalGPHead(input_dim=32, n_inducing=128, n_train=3200)
    >>> emb = torch.randn(4, 32)
    >>> mean, var, lo95, hi95 = head.predict(emb)
    >>> mean.shape
    torch.Size([4])
    """

    def __init__(
        self,
        input_dim: int = 32,
        n_inducing: int = 64,
        n_train: int | None = None,
        inducing_points: torch.Tensor | None = None,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
        mlp_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        _require_gpytorch()
        if n_train is None:
            raise ValueError("n_train is required for the ELBO normalisation constant")

        if mlp_hidden:
            layers: list[nn.Module] = []
            in_dim = input_dim
            for h in mlp_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature_extractor = nn.Sequential(*layers)
            gp_input_dim = mlp_hidden[-1]
        else:
            self.feature_extractor = None
            gp_input_dim = input_dim

        if inducing_points is None:
            inducing_points = torch.randn(n_inducing, gp_input_dim)
        elif (
            inducing_points.shape[-1] != gp_input_dim
            and self.feature_extractor is not None
        ):
            with torch.no_grad():
                inducing_points = self.feature_extractor(inducing_points)

        self.gp_layer = _VariationalGPLayer(
            inducing_points,
            gp_input_dim,
            lengthscale_range=lengthscale_range,
            lengthscale_prior=lengthscale_prior,
            outputscale_prior=outputscale_prior,
        ).double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.mll = VariationalELBO(self.likelihood, self.gp_layer, num_data=n_train)

    @staticmethod
    def _gp_context():
        """Safety-net jitter in case float64 K_uu is still marginal."""
        return gpytorch.settings.cholesky_jitter(
            float_value=1e-3, double_value=1e-4
        )

    def _apply_fe(self, embedding: torch.Tensor) -> torch.Tensor:
        """Run optional feature extractor (float32), return float64 for GP."""
        if self.feature_extractor is not None:
            embedding = self.feature_extractor(embedding)
        return embedding.double()

    def forward(
        self, embedding: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass returning a GP predictive distribution.

        Parameters
        ----------
        embedding : torch.Tensor
            Global embedding of shape ``(B, D)`` from the encoder.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Predictive distribution in the caller's original dtype.
        """
        orig_dtype = embedding.dtype
        with self._gp_context():
            dist = self.gp_layer(self._apply_fe(embedding))
        return gpytorch.distributions.MultivariateNormal(
            dist.mean.to(orig_dtype),
            dist.lazy_covariance_matrix.to_dense().to(orig_dtype),
        )

    def forward_and_loss(
        self,
        embedding: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both the predictive mean and ELBO loss.

        Parameters
        ----------
        embedding : torch.Tensor
            Global embedding of shape ``(B, D)``.
        target : torch.Tensor
            Scalar target values of shape ``(B,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(mean, neg_elbo)`` — predictive mean ``(B,)`` and negative
            ELBO loss (scalar), both in the caller's dtype.
        """
        orig_dtype = embedding.dtype
        with self._gp_context():
            dist = self.gp_layer(self._apply_fe(embedding))
            neg_elbo = -self.mll(dist, target.double())
        return dist.mean.to(orig_dtype), neg_elbo.to(orig_dtype)

    def loss(self, embedding: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the negative ELBO loss.

        Parameters
        ----------
        embedding : torch.Tensor
            Global embedding of shape ``(B, D)``.
        target : torch.Tensor
            Scalar target values of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar negative ELBO loss in the caller's dtype.
        """
        _, neg_elbo = self.forward_and_loss(embedding, target)
        return neg_elbo

    @torch.no_grad()
    def predict(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce predictions with calibrated uncertainty intervals.

        Parameters
        ----------
        embedding : torch.Tensor
            Global embedding of shape ``(B, D)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(mean, variance, lower_95, upper_95)`` — all ``(B,)`` tensors
            in the caller's dtype.  The 95 % interval is
            ``mean ± 1.96 * sqrt(variance)``.
        """
        orig_dtype = embedding.dtype
        self.eval()
        self.likelihood.eval()
        with self._gp_context(), gpytorch.settings.fast_pred_var():
            dist = self.gp_layer(self._apply_fe(embedding))
            pred = self.likelihood(dist)
            mean = pred.mean
            var = pred.variance
            lower = mean - 1.96 * var.sqrt()
            upper = mean + 1.96 * var.sqrt()
        return (
            mean.to(orig_dtype),
            var.to(orig_dtype),
            lower.to(orig_dtype),
            upper.to(orig_dtype),
        )
