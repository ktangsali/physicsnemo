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

"""Pointwise multitask variational Gaussian Process head for field regression.

Provides :class:`FieldVariationalGPHead`, a module that can be attached to any
backbone which exposes *per-point* features to produce calibrated, per-point
uncertainty estimates over a multi-channel field (e.g. surface pressure +
wall-shear-stress).

This is the *field* member of a two-head family.  Both are variational GPs with
inducing points, a Matern-5/2 ARD kernel and a variational ELBO; they differ in
what a "data point" is:

============  ==================================  ==========================
Head          Input                               Output
============  ==================================  ==========================
`VariationalGPHead`       one pooled embedding    one scalar per geometry
                          ``(B, D)``              ``(B,)``
`FieldVariationalGPHead`  per-point features      ``num_tasks`` channels per
                          ``(..., D)``            point ``(..., num_tasks)``
============  ==================================  ==========================

The posterior mean is the field prediction; the posterior variance is the
per-point uncertainty, which grows as a point's feature moves away from the
learned inducing points (a distance-aware, single-pass UQ signal that needs no
ensembling or MC-Dropout).

Attaching to a backbone
-----------------------
The head is deliberately model-agnostic: it consumes a feature tensor and
nothing else.  There is no dependency on any particular backbone, no assumption
about mesh topology, and coordinates are *not* required as a separate input
(any positional information the backbone encodes simply arrives inside the
features).  The only contract is:

1. the backbone emits per-point features whose last dimension is
   ``input_dim`` — any leading batch/point dims are flattened internally, so
   ``(B, N, D)``, ``(N, D)`` and ``(B, T, N, D)`` all work;
2. targets are supplied as ``(..., num_tasks)`` matching those leading dims.

So any point-wise encoder works — GeoTransolver, DoMINO, MeshGraphNet — by
exposing whatever it already computes before its final projection::

    head = FieldVariationalGPHead(input_dim=feat_dim, num_tasks=4,
                                  n_train=n_points_per_epoch)

    feats = backbone.encode(batch)        # (B, N, feat_dim)
    mean, neg_elbo = head.forward_and_loss(feats, targets, beta=beta)
    loss = neg_elbo + lambda_mse * mse(mean, targets)

At inference, :meth:`FieldVariationalGPHead.predict` returns the mean plus the
epistemic/total variance split in a single forward pass.

Key design choices
------------------
* **Independent multitask GP** — Each of the ``num_tasks`` output channels has
  its own variational GP (shared inducing-point structure, independent kernels
  and variational parameters), built with GPyTorch's
  ``IndependentMultitaskVariationalStrategy``.
* **Float64 GP internals (default)** — Short lengthscales on the inducing-point
  covariance make ``K_uu`` ill-conditioned in float32.  GP internals run in
  float64 by default; inputs are upcast on entry and outputs downcast on exit so
  gradients flow through the backbone seamlessly.  Controlled by *use_double*.
* **Optional DKL feature extractor** — A small pointwise MLP can be inserted
  between the backbone features and the GP kernel (Deep Kernel Learning),
  reducing a wide feature vector to a compact, well-conditioned kernel input.
* **Matern-5/2 ARD kernel** — Smooth, twice-differentiable, with per-dimension
  lengthscales (Automatic Relevance Determination).
* **Optional heteroscedastic noise** — The observation noise can be made a
  function of the features rather than one learned scalar per channel; see
  :meth:`FieldVariationalGPHead._hetero_neg_elbo`.

Requires ``gpytorch`` — install via ``pip install gpytorch`` or use the
``uq-extras`` optional dependency group.
"""

from __future__ import annotations

import importlib
import math
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec

_GPYTORCH_AVAILABLE = check_version_spec("gpytorch", hard_fail=False)

if _GPYTORCH_AVAILABLE:
    gpytorch = importlib.import_module("gpytorch")
    _ApproximateGP = gpytorch.models.ApproximateGP
    CholeskyVariationalDistribution = (
        gpytorch.variational.CholeskyVariationalDistribution
    )
    VariationalStrategy = gpytorch.variational.VariationalStrategy
    IndependentMultitaskVariationalStrategy = (
        gpytorch.variational.IndependentMultitaskVariationalStrategy
    )
    VariationalELBO = gpytorch.mlls.VariationalELBO
else:
    _ApproximateGP = nn.Module


def _require_gpytorch() -> None:
    if not _GPYTORCH_AVAILABLE:
        raise ImportError(
            "physicsnemo.experimental.uq.FieldVariationalGPHead requires gpytorch. "
            "Install it with: pip install gpytorch  "
            "(or: pip install nvidia-physicsnemo[uq-extras])"
        )


class _MultitaskVariationalGPLayer(_ApproximateGP):
    """Low-level independent multitask variational GP with Matern-5/2 ARD kernels.

    This is an internal building block used by :class:`FieldVariationalGPHead`.  Users
    should not need to instantiate it directly.

    Parameters
    ----------
    inducing_points : torch.Tensor
        Initial inducing point locations of shape ``(num_tasks, M, D)``.
    input_dim : int
        Dimensionality of each input (must match last dim of *inducing_points*).
    num_tasks : int
        Number of output channels / independent GPs.
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
        input_dim: int = 16,
        num_tasks: int = 4,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
    ) -> None:
        _require_gpytorch()
        batch_shape = torch.Size([num_tasks])

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=batch_shape,
        )
        base_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            base_strategy,
            num_tasks=num_tasks,
        )
        super().__init__(variational_strategy)

        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        ls_constraint = gpytorch.constraints.Interval(*lengthscale_range)
        ls_prior_obj = None
        if lengthscale_prior is not None:
            ls_prior_obj = gpytorch.priors.GammaPrior(*lengthscale_prior)

        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=input_dim,
            batch_shape=batch_shape,
            lengthscale_constraint=ls_constraint,
            lengthscale_prior=ls_prior_obj,
        )

        os_prior_obj = None
        if outputscale_prior is not None:
            os_prior_obj = gpytorch.priors.GammaPrior(*outputscale_prior)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            batch_shape=batch_shape,
            outputscale_prior=os_prior_obj,
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FieldVariationalGPPrediction(NamedTuple):
    """Structured output of :meth:`FieldVariationalGPHead.predict`.

    Attributes
    ----------
    mean : torch.Tensor
        Predictive mean, shape ``(..., num_tasks)``.
    variance : torch.Tensor
        Total predictive variance (epistemic + aleatoric/observation noise),
        shape ``(..., num_tasks)``.
    lower : torch.Tensor
        Lower bound of the confidence interval, shape ``(..., num_tasks)``.
    upper : torch.Tensor
        Upper bound of the confidence interval, shape ``(..., num_tasks)``.
    epistemic_variance : torch.Tensor
        Latent GP function variance *only* (the reducible / model uncertainty,
        excluding the constant likelihood noise floor), shape
        ``(..., num_tasks)``.  This is the signal to use for active learning and
        for "where is the model uncertain?" maps — it has far more spatial
        contrast than the noise-dominated total ``variance``.
    """

    mean: torch.Tensor
    variance: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    epistemic_variance: torch.Tensor


class FieldVariationalGPHead(nn.Module):
    r"""Pointwise independent multitask variational GP head for field UQ.

    Attach this module to any backbone that produces per-point features to
    obtain calibrated, per-point uncertainty estimates over a multi-channel
    field.  The posterior mean is the field prediction; the posterior variance
    is the per-point uncertainty.

    Inputs of shape ``(..., D)`` are accepted (e.g. ``(B, N, D)`` or
    ``(N, D)``); all leading dimensions are flattened into the point dimension,
    the GP is evaluated, and outputs are reshaped back to ``(..., num_tasks)``.

    Parameters
    ----------
    input_dim : int
        Dimension of each per-point feature vector from the backbone.
    num_tasks : int, optional
        Number of output channels (independent GPs). Default is 4.
    n_inducing : int, optional
        Number of inducing points per task. Default is 256.
    n_train : int
        Total number of *training points* (across all geometries) — used for
        the ELBO normalisation constant so the data term and KL term are
        balanced when minibatching at the point level.
    inducing_points : torch.Tensor | None, optional
        Initial inducing locations, either ``(M, gp_dim)`` (shared init,
        broadcast across tasks) or ``(num_tasks, M, gp_dim)``.  If *None*,
        random normal points are used. Default is ``None``.
    lengthscale_range : tuple[float, float], optional
        Hard interval constraint ``[lo, hi]`` on per-dimension ARD
        lengthscales. Default is ``(0.01, 10.0)``.
    lengthscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on lengthscales.
        Default is ``None``.
    outputscale_prior : tuple[float, float] | None, optional
        ``(concentration, rate)`` for a Gamma prior on the output scale.
        Default is ``None``.
    mlp_hidden : list[int] | None, optional
        Hidden layer sizes for an optional pointwise DKL feature extractor MLP
        inserted before the GP kernel.  ``None`` feeds the features directly to
        the GP. Default is ``None``.
    feature_norm : {"none", "l2_radial"}, optional
        Normalisation applied to the GP-input features.  ``"none"`` passes them
        through.  ``"l2_radial"`` splits each feature into its unit direction
        plus its (batch-standardised) magnitude, appended as one extra ARD
        dimension — so ``gp_input_dim`` becomes ``mlp_hidden[-1] + 1``.  This
        pins the feature scale, preventing the DKL map from shrinking distances
        to circumvent the lengthscale constraint, while keeping the radial
        out-of-distribution cue that a pure unit-sphere projection discards.
        Default is ``"none"``.
    use_double : bool, optional
        If ``True``, GP internals run in float64 for numerical stability of the
        Cholesky decomposition on ``K_uu``. Default is ``True``.
    jitter : tuple[float, float], optional
        ``(float_value, double_value)`` passed to
        ``gpytorch.settings.cholesky_jitter``. Default is ``(1e-3, 1e-4)``.
    confidence_z : float, optional
        Z-score multiplier for the confidence interval returned by
        :meth:`predict`.  Default is ``1.96`` (95 % interval).
    noise_mlp_hidden : list[int] | None, optional
        Hidden sizes of an observation-noise MLP over the GP-input features.
        ``None`` (default) keeps the standard homoscedastic
        ``MultitaskGaussianLikelihood`` (one noise scalar per channel).  When
        set, the observation noise becomes input-dependent, which makes the
        *total* predictive std informative for per-point error ranking — with a
        constant noise floor the total std ranks points identically to the
        epistemic std, so all ranking signal comes from the (typically <1 %)
        epistemic share of the variance.  The epistemic/aleatoric split is
        retained; only the aleatoric part gains spatial structure.
    noise_std_range : tuple[float, float], optional
        Hard clamp ``(lo, hi)`` on the per-point noise std, as a safety net
        against a degenerate zero-noise solution.  Default ``(1e-3, 10.0)``.

    Attributes
    ----------
    gp_layer : _MultitaskVariationalGPLayer
        The independent multitask variational GP.
    likelihood : gpytorch.likelihoods.MultitaskGaussianLikelihood
        Homoscedastic observation-noise model.  Retained even when
        *noise_mlp_hidden* is set (the heteroscedastic path computes its own
        noise), because it holds the learned per-channel noise floor.
    mll : gpytorch.mlls.VariationalELBO
        Marginal log-likelihood objective (its ``beta`` can be annealed).
    feature_extractor : nn.Sequential | None
        Optional DKL MLP.
    gp_input_dim : int
        Width of the kernel input, after the DKL MLP and *feature_norm*.

    See Also
    --------
    physicsnemo.experimental.uq.VariationalGPHead
        The scalar counterpart: pools a geometry to one embedding and predicts a
        single value per geometry rather than a field.

    Examples
    --------
    >>> head = FieldVariationalGPHead(
    ...     input_dim=448, num_tasks=4, n_inducing=256,
    ...     n_train=51200 * 100, mlp_hidden=[128],
    ... )
    >>> feats = torch.randn(1, 4096, 448)
    >>> pred = head.predict(feats)
    >>> pred.mean.shape
    torch.Size([1, 4096, 4])
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int = 4,
        n_inducing: int = 256,
        n_train: int | None = None,
        inducing_points: torch.Tensor | None = None,
        lengthscale_range: tuple[float, float] = (0.01, 10.0),
        lengthscale_prior: tuple[float, float] | None = None,
        outputscale_prior: tuple[float, float] | None = None,
        mlp_hidden: list[int] | None = None,
        feature_norm: str = "none",
        use_double: bool = True,
        jitter: tuple[float, float] = (1e-3, 1e-4),
        confidence_z: float = 1.96,
        noise_mlp_hidden: list[int] | None = None,
        noise_std_range: tuple[float, float] = (1e-3, 10.0),
    ) -> None:
        super().__init__()
        _require_gpytorch()
        if n_train is None:
            raise ValueError("n_train is required for the ELBO normalisation constant")

        if feature_norm not in ("none", "l2_radial"):
            raise ValueError(
                f"feature_norm must be 'none' or 'l2_radial', got {feature_norm!r}"
            )

        self.num_tasks = num_tasks
        self._use_double = use_double
        self._jitter = jitter
        self._confidence_z = confidence_z
        self._feature_norm = feature_norm

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

        # 'l2_radial' keeps the L2-normalised direction AND appends the pre-norm
        # feature magnitude as one extra ARD dimension.  Projecting onto the unit
        # sphere alone makes out-of-distribution geometries indistinguishable in
        # norm, collapsing the OOD/in-distribution std ratio towards 1.0.  The
        # magnitude is standardised by a (non-affine) BatchNorm tracking the
        # training distribution, so OOD magnitudes land in the tails -> larger
        # kernel distance -> higher posterior variance.
        if feature_norm == "l2_radial":
            self._radial_bn = nn.BatchNorm1d(1, affine=False)
            gp_input_dim += 1
        else:
            self._radial_bn = None
        self.gp_input_dim = gp_input_dim

        inducing_points = self._init_inducing(
            inducing_points, n_inducing, gp_input_dim, num_tasks
        )

        gp_layer = _MultitaskVariationalGPLayer(
            inducing_points,
            gp_input_dim,
            num_tasks=num_tasks,
            lengthscale_range=lengthscale_range,
            lengthscale_prior=lengthscale_prior,
            outputscale_prior=outputscale_prior,
        )
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        )

        if use_double:
            gp_layer = gp_layer.double()
            likelihood = likelihood.double()

        self.gp_layer = gp_layer
        # Kept even in the heteroscedastic case: it is the homoscedastic noise
        # model (unused then), and eval scripts call ``head.likelihood.eval()``.
        self.likelihood = likelihood
        self.mll = VariationalELBO(self.likelihood, self.gp_layer, num_data=n_train)
        self._num_data = int(n_train)

        # ---- Optional input-dependent (heteroscedastic) observation noise ----
        # The default MultitaskGaussianLikelihood learns ONE noise scalar per
        # channel. That constant dominates the total predictive variance (on
        # DrivAerStar surface pressure it is ~99.6% of it) and, being constant,
        # adds zero information to the per-point ranking of |error| — the total
        # std ranks points exactly like the epistemic std. Making the noise a
        # function of the GP-input features lets the other ~99% of the variance
        # carry ranking signal, and is physically right (a wake has higher
        # irreducible variance than the hood). The epistemic/aleatoric split is
        # preserved — and sharpened, since the aleatoric part now varies too.
        self._noise_range = (float(noise_std_range[0]), float(noise_std_range[1]))
        if noise_mlp_hidden:
            layers: list[nn.Module] = []
            in_dim = gp_input_dim
            for h in noise_mlp_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, num_tasks))
            noise_head = nn.Sequential(*layers)
            # Zero-init the output layer so the modulation starts at exactly 1x
            # and training begins from the homoscedastic solution rather than
            # from a random (possibly tiny or huge) per-point noise.
            nn.init.zeros_(noise_head[-1].weight)
            nn.init.zeros_(noise_head[-1].bias)
            base = torch.zeros(num_tasks)
            if use_double:
                noise_head = noise_head.double()
                base = base.double()
            self.noise_head = noise_head
            # Per-task log base std; the MLP only supplies a bounded
            # multiplicative deviation around it.
            self.log_base_noise = nn.Parameter(base)
        else:
            self.noise_head = None
            self.log_base_noise = None

    @staticmethod
    def _init_inducing(
        inducing_points: torch.Tensor | None,
        n_inducing: int,
        gp_input_dim: int,
        num_tasks: int,
    ) -> torch.Tensor:
        """Return inducing points of shape ``(num_tasks, M, gp_input_dim)``."""
        if inducing_points is None:
            return torch.randn(num_tasks, n_inducing, gp_input_dim)
        if inducing_points.dim() == 2:
            return inducing_points.unsqueeze(0).expand(num_tasks, -1, -1).contiguous()
        if inducing_points.dim() == 3:
            return inducing_points
        raise ValueError(
            "inducing_points must be (M, D) or (num_tasks, M, D), "
            f"got shape {tuple(inducing_points.shape)}"
        )

    def _gp_context(self):
        """Safety-net jitter for near-singular covariance matrices."""
        return gpytorch.settings.cholesky_jitter(
            float_value=self._jitter[0], double_value=self._jitter[1]
        )

    @property
    def heteroscedastic(self) -> bool:
        """Whether an input-dependent observation-noise head is active."""
        return self.noise_head is not None

    def _pointwise_noise_var(
        self, gp_in: torch.Tensor
    ) -> Float[torch.Tensor, "points tasks"]:
        """Per-point, per-task observation-noise variance.

        ``sigma_t(x) = exp(log_base_noise_t + clamp(g_t(x), -3, 3))`` — a learned
        per-task base scale times a bounded multiplicative modulation from the
        noise MLP.  The clamp keeps the modulation within ~20x either way so a
        bad step cannot drive the noise to 0 (infinite log-likelihood) or blow
        it up; the final clamp to ``noise_std_range`` is a hard safety net.
        """
        log_mod = self.noise_head(gp_in).clamp(-3.0, 3.0)
        std = torch.exp(self.log_base_noise.to(log_mod.dtype) + log_mod)
        std = std.clamp(self._noise_range[0], self._noise_range[1])
        return std.square()

    def _hetero_neg_elbo(
        self,
        dist: "gpytorch.distributions.MultitaskMultivariateNormal",
        gp_target: torch.Tensor,
        gp_in: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        r"""Negative ELBO with diagonal, input-dependent Gaussian noise.

        The expected log-likelihood under a heteroscedastic Gaussian is the
        variational analogue of the attenuated regression loss of Kendall & Gal
        (*What Uncertainties Do We Need in Bayesian Deep Learning for Computer
        Vision?*, NeurIPS 2017),

        .. math::
            -\log p(y \mid x) \;\simeq\;
            \frac{\lVert y - \mu(x) \rVert^2 + \operatorname{Var}[f(x)]}
                 {2\,\sigma^2(x)}
            + \tfrac{1}{2}\log \sigma^2(x),

        differing only in that the GP contributes the extra
        :math:`\operatorname{Var}[f(x)]` term (the latent posterior variance),
        which their deterministic network does not have.  Each point is weighted
        by :math:`1/\sigma^2(x)`, so the noise head learns to down-weight
        genuinely noisy regions instead of forcing the mean to fit them.

        Mirrors ``gpytorch.mlls.VariationalELBO``'s normalisation exactly — the
        expected log-likelihood is summed over points and tasks then divided by
        the number of points, and the KL is divided by ``num_data / beta`` — so
        the loss is on the same scale as the homoscedastic path and the existing
        learning rates and beta/NLL warmup schedules carry over unchanged.
        """
        mu = dist.mean
        latent_var = dist.variance
        noise_var = self._pointwise_noise_var(gp_in)
        # E_q[log N(y | f, sigma^2)] for a diagonal Gaussian, where the
        # E_q[(y - f)^2] term contributes the latent variance.
        ll = -0.5 * (
            math.log(2.0 * math.pi)
            + noise_var.log()
            + ((gp_target - mu).square() + latent_var) / noise_var
        )
        log_lik = ll.sum() / gp_target.shape[0]
        kl = self.gp_layer.variational_strategy.kl_divergence().sum()
        kl = kl / (self._num_data / max(float(beta), 1e-8))
        return -(log_lik - kl)

    def _transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Run optional DKL extractor then the (scale-fixing) feature norm.

        The normalisation pins the GP-input feature scale so the kernel
        lengthscale must do the smoothing work; without it the DKL map can
        shrink feature distances to defeat a lengthscale constraint.
        ``l2_radial`` keeps the L2-normalised direction but appends the
        (standardised) pre-norm magnitude as an extra dimension, retaining the
        radial out-of-distribution cue that a pure unit-sphere projection
        discards.
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(features)
        if self._feature_norm == "l2_radial":
            # Split into direction (unit sphere) + standardised magnitude.
            magnitude = features.norm(dim=-1, keepdim=True)
            direction = features / magnitude.clamp_min(1e-12)
            # BatchNorm1d expects (N, C); flatten all leading dims into N.
            lead_shape = magnitude.shape[:-1]
            mag_std = self._radial_bn(magnitude.reshape(-1, 1)).reshape(*lead_shape, 1)
            features = torch.cat([direction, mag_std], dim=-1)
        return features

    def _apply_fe(self, features: torch.Tensor) -> torch.Tensor:
        """Run feature transform (DKL + norm), then cast to GP precision."""
        features = self._transform_features(features)
        if self._use_double:
            return features.double()
        return features

    def transform_features(self, features: torch.Tensor) -> torch.Tensor:
        """Public wrapper for the DKL + feature-norm transform (GP-input space).

        Returns the features the kernel actually sees (same space as the
        inducing points), *without* the double-precision cast, so callers can
        compute auxiliary losses (e.g. a distance penalty) on the GP-input
        geometry while keeping gradients flowing back into the backbone. Pass
        the result to :meth:`forward_and_loss` with ``pretransformed=True`` to
        avoid re-running the transform (and its BatchNorm) a second time.
        """
        return self._transform_features(features)

    @staticmethod
    def _flatten_points(features: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        """Flatten all leading dims into a single point dimension."""
        lead = features.shape[:-1]
        return features.reshape(-1, features.shape[-1]), lead

    @torch.no_grad()
    def set_inducing_points(self, points: torch.Tensor) -> None:
        """Re-seed inducing locations from collected features.

        Accepts ``(M, D)`` (shared across tasks) or ``(num_tasks, M, D)`` in
        the *raw feature* space (the DKL extractor, if any, is applied here).
        The variational mean is zeroed and the variational covariance reset to
        a small identity so GP-side optimisation restarts cleanly.
        """
        base = self.gp_layer.variational_strategy.base_variational_strategy
        device = base.inducing_points.device

        # Apply the same DKL + feature-norm transform used at inference so the
        # inducing points live in the same (normalised) GP-input space.
        if self.feature_extractor is not None:
            fe_device = next(self.feature_extractor.parameters()).device
            points = points.to(fe_device)
        points = self._transform_features(points)
        points = self._init_inducing(
            points, points.shape[-2], self.gp_input_dim, self.num_tasks
        )
        if self._use_double:
            points = points.double()
        points = points.to(device)

        base.inducing_points.data.copy_(points)
        vd = base._variational_distribution
        vd.variational_mean.data.zero_()
        m = points.shape[-2]
        eye = torch.eye(m, device=device, dtype=vd.chol_variational_covar.dtype)
        vd.chol_variational_covar.data.copy_(
            (eye * 0.01).expand_as(vd.chol_variational_covar)
        )

    def forward(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        r"""Forward pass returning the per-point multitask predictive distribution.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features from the backbone; any leading dims are
            flattened into the point dimension.

        Returns
        -------
        gpytorch.distributions.MultitaskMultivariateNormal
            Predictive distribution over ``(P, num_tasks)`` in float (the GP's
            working precision); reshape via :meth:`predict` for original dtype.
        """
        flat, _ = self._flatten_points(features)
        with self._gp_context():
            return self.gp_layer(self._apply_fe(flat))

    def forward_and_loss(
        self,
        features: Float[torch.Tensor, "... dim"],
        target: Float[torch.Tensor, "... tasks"],
        beta: float = 1.0,
        pretransformed: bool = False,
        return_variance: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        r"""Forward pass returning the predictive mean and negative ELBO.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features of shape ``(..., D)``.
        target : Float[torch.Tensor, "... tasks"]
            Per-point field targets of shape ``(..., num_tasks)``.
        beta : float, optional
            KL-term weight for the ELBO (for KL annealing). Default ``1.0``.
        pretransformed : bool, optional
            If ``True``, ``features`` are assumed to already be in the GP-input
            space (i.e. the output of :meth:`transform_features`) so the DKL +
            feature-norm transform is skipped and only the precision cast is
            applied. Default ``False``.
        return_variance : bool, optional
            If ``True``, also return the per-point *latent* (epistemic) variance
            of shape ``target`` (with gradient), for auxiliary losses such as a
            within-sample concordance penalty. Default ``False``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(mean, neg_elbo)`` — predictive mean reshaped to ``target``'s
            shape and the negative ELBO (scalar), both in the caller's dtype. If
            ``return_variance`` is ``True``, a third element is the per-point
            latent variance reshaped to ``target``'s shape.
        """
        orig_dtype = features.dtype
        flat, lead = self._flatten_points(features)
        flat_target = target.reshape(-1, self.num_tasks)
        gp_target = flat_target.double() if self._use_double else flat_target

        # GPyTorch's VariationalELBO scales the KL term by ``num_data / beta``,
        # so beta must stay strictly positive. Floor it to a tiny value so that
        # ``beta -> 0`` (KL annealing warmup) effectively removes the KL term
        # without a division-by-zero.
        self.mll.beta = max(float(beta), 1e-8)
        if pretransformed:
            gp_in = flat.double() if self._use_double else flat
        else:
            gp_in = self._apply_fe(flat)
        with self._gp_context():
            dist = self.gp_layer(gp_in)
            if self.heteroscedastic:
                neg_elbo = self._hetero_neg_elbo(dist, gp_target, gp_in, beta)
            else:
                neg_elbo = -self.mll(dist, gp_target)
        mean = dist.mean.to(orig_dtype).reshape(*lead, self.num_tasks)
        if return_variance:
            var = dist.variance.to(orig_dtype).reshape(*lead, self.num_tasks)
            return mean, neg_elbo.to(orig_dtype), var
        return mean, neg_elbo.to(orig_dtype)

    def loss(
        self,
        features: Float[torch.Tensor, "... dim"],
        target: Float[torch.Tensor, "... tasks"],
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Compute the (beta-weighted) negative ELBO loss."""
        _, neg_elbo = self.forward_and_loss(features, target, beta=beta)
        return neg_elbo

    @torch.no_grad()
    def predict(
        self, features: Float[torch.Tensor, "... dim"]
    ) -> FieldVariationalGPPrediction:
        r"""Produce per-point predictions with calibrated uncertainty.

        Parameters
        ----------
        features : Float[torch.Tensor, "... dim"]
            Per-point features of shape ``(..., D)``.

        Returns
        -------
        FieldVariationalGPPrediction
            Named tuple ``(mean, variance, lower, upper, epistemic_variance)`` —
            all of shape ``(..., num_tasks)`` in the caller's dtype.  The
            confidence interval is ``mean +/- confidence_z * sqrt(variance)``.
            ``variance`` is the total (epistemic + observation noise) predictive
            variance; ``epistemic_variance`` is the latent GP term alone, which
            is the signal to use for active learning and "where is the model
            uncertain?" maps.
        """
        orig_dtype = features.dtype
        flat, lead = self._flatten_points(features)
        was_training = self.training
        self.eval()
        self.likelihood.eval()
        try:
            with self._gp_context(), gpytorch.settings.fast_pred_var():
                gp_in = self._apply_fe(flat)
                dist = self.gp_layer(gp_in)
                # Latent (epistemic) variance, before the observation-noise
                # floor is added.
                epistemic_var = dist.variance
                if self.heteroscedastic:
                    # Same decomposition, but the aleatoric term now varies
                    # per point instead of being one scalar per channel.
                    mean = dist.mean
                    var = epistemic_var + self._pointwise_noise_var(gp_in)
                else:
                    pred = self.likelihood(dist)
                    mean = pred.mean
                    var = pred.variance
                z = self._confidence_z
                lower = mean - z * var.sqrt()
                upper = mean + z * var.sqrt()
            return FieldVariationalGPPrediction(
                mean=mean.to(orig_dtype).reshape(*lead, self.num_tasks),
                variance=var.to(orig_dtype).reshape(*lead, self.num_tasks),
                lower=lower.to(orig_dtype).reshape(*lead, self.num_tasks),
                upper=upper.to(orig_dtype).reshape(*lead, self.num_tasks),
                epistemic_variance=epistemic_var.to(orig_dtype).reshape(
                    *lead, self.num_tasks
                ),
            )
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Backwards-compatible aliases.
#
# This head shipped as ``FieldGPHead`` before being renamed to sit alongside
# the scalar ``VariationalGPHead``. These aliases keep existing imports (e.g.
# the physicsnemo-cfd evaluation wrapper) working. Note that they do NOT
# preserve checkpoint filenames: ``save_checkpoint`` derives the file stem from
# ``type(model).__name__``, which resolves to the new name through an alias, so
# new runs write ``FieldVariationalGPHead.0.<tag>.pt``. Older
# ``FieldGPHead.0.<tag>.pt`` files remain loadable by explicit path.
# ---------------------------------------------------------------------------
FieldGPHead = FieldVariationalGPHead
FieldGPPrediction = FieldVariationalGPPrediction
