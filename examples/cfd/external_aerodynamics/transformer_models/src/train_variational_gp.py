# Core python imports:
import os
import time
from pathlib import Path
from typing import Literal, Any, Callable, Sequence
import collections
from contextlib import nullcontext

from collections.abc import Sequence

# Configuration:
import hydra
import omegaconf
from omegaconf import DictConfig

# Pytorch imports:
import torch
from torch import nn
from torch.optim import Optimizer
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

# For metrics and model printouts:
from tabulate import tabulate
import torchinfo

# For loading dataset stats:
import numpy as np

# Physicsnemo imports ...
import fsspec
import physicsnemo
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.checkpoint import (
    _get_checkpoint_filename,
    _unique_model_names,
    checkpoint_logging,
)
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.profiling import profile, Profiler
from physicsnemo.datapipes.cae.transolver_datapipe import (
    create_transolver_dataset,
    TransolverDataPipe,
)
from physicsnemo.models.domino.utils import unstandardize

# Local folder imports for this example
from metrics import metrics_fn

# tensorwise is to handle single-point-cloud or multi-point-cloud running.
# it's a decorator that will automatically unzip one or more of a list of tensors,
# run the funtcion, and rezip the results.
from utils import tensorwise

# Special import, if transformer engine is available:
from physicsnemo.core.version_check import check_version_spec

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
else:
    te, Format, DelayedScaling = None, None, None

# GPyTorch related stuff

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO


# Allowlist OmegaConf and related types for torch.load(weights_only=True) when loading
# checkpoints that were saved with config/metadata containing these types (PyTorch 2.6+).
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([Any])
torch.serialization.add_safe_globals([list])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])
torch.serialization.add_safe_globals([int])
torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
torch.serialization.add_safe_globals([omegaconf.base.Metadata])


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.

    Note:
        This will get upstreamed to physicsnemo shortly.  Don't count on this
        class existing here in the future!

        In other words, this is already marked for deprecation!
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


def update_model_params_for_fp8(cfg, logger) -> tuple | None:
    """
    Adjusts model configuration parameters to ensure compatibility with FP8 computations.

    The output shape will be padded to a multiple of 16.  The input shape
    is padded dynamically in the forward pass, but that is printed here
    for information.

    Args:
        cfg: Configuration object with model and training attributes.
        logger: Logger object for info messages.

    Returns:
        tuple: (cfg, output_pad_size) if precision is "float8", where output_pad_size is the amount
               of padding added to the output dimension (or None if no padding was needed).
    """
    # we have to manipulate the output shape
    # to enable fp8 computations with transformer_engine.
    # need the input and output to be divisible by 16.
    # if (cfg.model.embedding_dim + cfg.model.functional_dim) % 16 != 0:

    output_pad_size = None
    if cfg.precision == "float8":
        if cfg.model.out_dim % 16 != 0:
            # pad the output:
            output_pad_size = 16 - (cfg.model.out_dim % 16)
            cfg.model.out_dim += output_pad_size
            logger.info(
                f"Padding output dimension to {cfg.model.out_dim} for fp8 autocast"
            )

        # This part is informational only:
        if (cfg.model.functional_dim + cfg.model.embedding_dim) % 16 != 0:
            input_pad_size = 16 - (
                (cfg.model.functional_dim + cfg.model.embedding_dim) % 16
            )
            cfg.model.functional_dim += input_pad_size
            logger.info(
                f"Padding input dimension to {cfg.model.functional_dim} and {cfg.model.embedding_dim} for fp8 autocast"
            )

    return cfg, output_pad_size


def load_pretrained_model_only(
    model: torch.nn.Module,
    path: str,
    epoch: int | None = None,
) -> bool:
    """Load only the model state from a checkpoint path (e.g. pretrained GeoTransolver).
    Does not load optimizer/scheduler or training-state .pt file.

    Returns:
        True if at least one model was loaded, False otherwise.
    """
    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
    if not fs.exists(path):
        checkpoint_logging.warning(
            f"Pretrained checkpoint path does not exist: {path}, skipping load"
        )
        return False
    models_dict = _unique_model_names([model], loading=True)
    loaded_any = False
    for name, m in models_dict.items():
        if not isinstance(m, physicsnemo.core.Module):
            continue
        file_name = _get_checkpoint_filename(
            path, base_name=name, index=epoch, model_type="mdlus"
        )
        if fs.exists(file_name):
            m.load(file_name)
            checkpoint_logging.success(
                f"Loaded pretrained model state: {file_name}"
            )
            loaded_any = True
        else:
            checkpoint_logging.warning(
                f"Could not find pretrained model file: {file_name}, skipping"
            )
    return loaded_any


@tensorwise
def cast_precisions(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """
    Casts the tensors to the specified precision.

    We are careful to take either a tensor or list of tensors, and return the same format.
    """

    match precision:
        case "float16":
            dtype = torch.float16
        case "bfloat16":
            dtype = torch.bfloat16
        case _:
            dtype = None

    if dtype is not None:
        return tensor.to(dtype)
    else:
        return tensor


# Fixed reference for drag coefficient: A (m²), U (m/s), ρ (kg/m³)
# coeff = 2 / (A * ρ * U²) for dimensionless force coefficient
FRONTAL_AREA = 1.85
REFERENCE_VELOCITY = 40.0
REFERENCE_DENSITY = 1.225
DRAG_COEFF_SCALE = 0.35  # scale drag target for GP (target = Cd / DRAG_COEFF_SCALE)


def compute_force_coefficients_torch(
    normals: torch.Tensor,
    area: torch.Tensor,
    coeff: float,
    p: torch.Tensor,
    wss: torch.Tensor,
    force_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes force coefficients (e.g. drag) from surface pressure and wall shear stress.
    All tensors on same device; normals (N, 3), area (N,) or (N,1), p (N,), wss (N, 3).
    Returns c_total, c_p, c_f (scalars).
    """
    if force_direction is None:
        force_direction = torch.tensor(
            [1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype
        )
    area = area.view(-1)
    # (N, 3) @ (3,) -> (N,) for normals · force_direction
    n_dot_f = (normals * force_direction).sum(dim=-1)
    c_p = coeff * (n_dot_f * area * p).sum()
    wss_dot_f = (wss * force_direction).sum(dim=-1)
    c_f = -coeff * (wss_dot_f * area).sum()
    c_total = c_p + c_f
    return c_total, c_p, c_f


def compute_drag_target_from_batch(
    batch: dict,
    surface_factors: dict,
    device: torch.device,
    drag_scale: float = DRAG_COEFF_SCALE,
) -> torch.Tensor:
    """
    From dataloader batch (normalized fields, surface_normals, surface_areas),
    unnormalize fields, compute drag coefficient, return target scaled for GP (Cd / drag_scale).
    Uses full-mesh fields (batch["fields_full"]) when present so drag is computed on the
    entire surface; otherwise uses batch["fields"] (subsampled, shapes must match normals/areas).
    Returns tensor of shape (1,) on device for use as GP target.
    """
    # Prefer full-mesh fields for drag so normals/areas (full mesh) match
    if "fields_full" in batch:
        fields = batch["fields_full"]
    else:
        fields = batch["fields"]
    if isinstance(fields, list):
        fields = fields[0]
    # (B, N, 4) -> unstandardize per channel
    mean = surface_factors["mean"]
    std = surface_factors["std"]
    fields_phys = unstandardize(fields, mean, std)
    # Single sample: (1, N, 4)
    fields_phys = fields_phys.squeeze(0)
    p = fields_phys[:, 0]
    wss = fields_phys[:, 1:4]

    normals = batch["surface_normals"].squeeze(0).to(device, dtype=fields_phys.dtype)
    area = batch["surface_areas"].squeeze(0).to(device, dtype=fields_phys.dtype)
    p = p.to(device)
    wss = wss.to(device)

    coeff = 2.0 / (
        FRONTAL_AREA * REFERENCE_DENSITY * (REFERENCE_VELOCITY ** 2)
    )
    c_total, _c_p, _c_f = compute_force_coefficients_torch(
        normals, area, coeff, p, wss
    )
    # Scale for GP target
    target = (c_total / drag_scale).unsqueeze(0)
    return target


def spectral_norm_wrapper(layer: nn.Module, use_sn: bool) -> nn.Module:
    """Wrap a layer with spectral normalization if requested."""
    if use_sn:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


class AttentionPooling(nn.Module):
    """
    Learns per-point importance weights before aggregating.
    """
    def __init__(
        self, feat_dim=256, embed_dim=32, hidden=128,
        spectral_norm=False, normalize=False, target_scale=1.0,
    ):
        super().__init__()
        sn = spectral_norm
        self.attention = nn.Sequential(
            spectral_norm_wrapper(nn.Linear(feat_dim, hidden), sn), nn.Tanh(),
            spectral_norm_wrapper(nn.Linear(hidden, 1), sn),
        )
        self.projector = nn.Sequential(
            spectral_norm_wrapper(nn.Linear(feat_dim, 256), sn), nn.ReLU(), nn.LayerNorm(256),
            spectral_norm_wrapper(nn.Linear(256, 128), sn), nn.ReLU(), nn.LayerNorm(128),
            spectral_norm_wrapper(nn.Linear(128, embed_dim), sn),
        )
        self.normalize = normalize
        self.target_scale = target_scale

    def forward(self, point_feats):
        # point_feats: (B, N, feat_dim)  e.g. (B, H*S, context_dim) for embedding_states
        attn_scores = self.attention(point_feats)              # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)      # (B, N, 1)
        weighted_sum = (attn_weights * point_feats).sum(dim=1) # (B, feat_dim)
        out = self.projector(weighted_sum)                     # (B, 32)
        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1) * self.target_scale
        return out


class MeanPooling(nn.Module):
    """
    Mean pooling over the spatial (point) dimension, then a linear projection to embed_dim.
    """
    def __init__(
        self, feat_dim=256, embed_dim=32, spectral_norm=False,
        normalize=False, target_scale=1.0,
    ):
        super().__init__()
        self.projector = spectral_norm_wrapper(nn.Linear(feat_dim, embed_dim), spectral_norm)
        self.normalize = normalize
        self.target_scale = target_scale

    def forward(self, point_feats):
        # point_feats: (B, N, feat_dim) -> (B, embed_dim)
        pooled = point_feats.mean(dim=1)
        out = self.projector(pooled)
        if self.normalize:
            out = torch.nn.functional.normalize(out, dim=-1) * self.target_scale
        return out


def create_embedding_reduction(
    pooling: Literal["attention", "mean"],
    feat_dim: int = 256,
    embed_dim: int = 32,
    spectral_norm: bool = False,
    normalize: bool = False,
    target_scale: float = 1.0,
    **kwargs: Any,
) -> nn.Module:
    """Create embedding reduction module from config."""
    if pooling == "attention":
        return AttentionPooling(
            feat_dim=feat_dim, embed_dim=embed_dim,
            spectral_norm=spectral_norm,
            normalize=normalize, target_scale=target_scale,
            **kwargs,
        )
    if pooling == "mean":
        return MeanPooling(
            feat_dim=feat_dim, embed_dim=embed_dim,
            spectral_norm=spectral_norm,
            normalize=normalize, target_scale=target_scale,
        )
    raise ValueError(f"Unknown embedding_pooling: {pooling}. Use 'attention' or 'mean'.")

class DKLGPLayer(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        embed_dim=32,
        lengthscale_range=(0.01, 10.0),
        lengthscale_prior=None,
        outputscale_prior=None,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()

        ls_constraint = gpytorch.constraints.Interval(*lengthscale_range)
        ls_prior_obj = None
        if lengthscale_prior is not None:
            # (concentration, rate) for Gamma — e.g. (3.0, 6.0) → mean 0.5
            ls_prior_obj = gpytorch.priors.GammaPrior(*lengthscale_prior)

        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=embed_dim,
            lengthscale_constraint=ls_constraint,
            lengthscale_prior=ls_prior_obj,
        )

        os_prior_obj = None
        if outputscale_prior is not None:
            # (concentration, rate) for Gamma — e.g. (2.0, 0.5) → mean 4.0
            os_prior_obj = gpytorch.priors.GammaPrior(*outputscale_prior)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=os_prior_obj,
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DragGP(nn.Module):
    """Variational GP head that operates entirely in float64.

    Short lengthscales on L2-normalised embeddings make the inducing-point
    covariance matrix K_uu ill-conditioned in float32.  Running the GP
    (kernel, variational strategy, likelihood) in float64 eliminates the
    precision issue at the source.  The parameter overhead is negligible
    (a few thousand doubles vs millions of float32 encoder weights).

    Inputs are cast to float64 on entry; outputs are cast back to the
    caller's dtype so gradients flow through seamlessly.

    When *mlp_hidden* is provided (e.g. ``[128, 64]``), a small feature
    extractor is inserted before the GP kernel (Deep Kernel Learning).
    The MLP runs in float32 for speed; its output is cast to float64 for
    the GP.
    """

    def __init__(
        self, embed_dim=32, n_inducing=64, n_train=None, inducing_points=None,
        lengthscale_range=(0.01, 10.0),
        lengthscale_prior=None,
        outputscale_prior=None,
        mlp_hidden=None,
    ):
        super().__init__()
        assert n_train is not None, "Must specify total number of training geometries"

        # Optional DKL feature extractor (float32)
        if mlp_hidden:
            layers = []
            in_dim = embed_dim
            for h in mlp_hidden:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature_extractor = nn.Sequential(*layers)
            gp_input_dim = mlp_hidden[-1]
        else:
            self.feature_extractor = None
            gp_input_dim = embed_dim

        if inducing_points is None:
            inducing_points = torch.randn(n_inducing, gp_input_dim)
        elif inducing_points.shape[-1] != gp_input_dim and self.feature_extractor is not None:
            with torch.no_grad():
                inducing_points = self.feature_extractor(inducing_points)
        self.gp_layer = DKLGPLayer(
            inducing_points, gp_input_dim,
            lengthscale_range=lengthscale_range,
            lengthscale_prior=lengthscale_prior,
            outputscale_prior=outputscale_prior,
        ).double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.mll = VariationalELBO(self.likelihood, self.gp_layer, num_data=n_train)

    @staticmethod
    def _gp_context():
        """Safety-net jitter in case float64 K_uu is still marginal."""
        return gpytorch.settings.cholesky_jitter(float_value=1e-3, double_value=1e-4)

    def _apply_fe(self, embedding):
        """Run optional feature extractor (float32), return float64 for GP."""
        if self.feature_extractor is not None:
            embedding = self.feature_extractor(embedding)
        return embedding.double()

    def forward(self, embedding):
        """
        Args:
            embedding: (B, D) global embedding from encoder (any dtype)
        Returns:
            dist: MultivariateNormal in the caller's original dtype
        """
        orig_dtype = embedding.dtype
        with self._gp_context():
            dist = self.gp_layer(self._apply_fe(embedding))
        return gpytorch.distributions.MultivariateNormal(
            dist.mean.to(orig_dtype),
            dist.lazy_covariance_matrix.to_dense().to(orig_dtype),
        )

    def forward_and_loss(self, embedding, drag_target):
        """Single forward pass returning both the predictive mean and ELBO loss.

        Args:
            embedding:   (B, D) global embedding
            drag_target: (B,) scalar drag values
        Returns:
            mean:     (B,) predictive mean in caller's dtype
            neg_elbo: scalar loss in caller's dtype
        """
        orig_dtype = embedding.dtype
        with self._gp_context():
            dist = self.gp_layer(self._apply_fe(embedding))
            neg_elbo = -self.mll(dist, drag_target.double())
        return dist.mean.to(orig_dtype), neg_elbo.to(orig_dtype)

    def loss(self, embedding, drag_target):
        """
        Args:
            embedding:   (B, D) global embedding
            drag_target: (B,) scalar drag values
        Returns:
            neg_elbo: scalar loss (in caller's dtype) to backprop through encoder + GP
        """
        _, neg_elbo = self.forward_and_loss(embedding, drag_target)
        return neg_elbo

    @torch.no_grad()
    def predict(self, embedding):
        """
        Args:
            embedding: (B, D)
        Returns:
            mean, variance, lower_95, upper_95  — all (B,) in caller's dtype
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
            mean.to(orig_dtype), var.to(orig_dtype),
            lower.to(orig_dtype), upper.to(orig_dtype),
        )



def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    # Set checkpoint directory - defaults to output_dir if not specified
    checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    if checkpoint_dir is None:
        checkpoint_dir = cfg.output_dir

    # train.py writes to checkpoints/; this script writes to checkpoints_gp/
    pretrained_ckpt_path = getattr(
        cfg, "pretrained_checkpoint_path", None
    ) or f"{checkpoint_dir}/{cfg.run_id}/checkpoints"
    gp_ckpt_path = f"{checkpoint_dir}/{cfg.run_id}/checkpoints_gp"

    if dist_manager.rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pretrained_ckpt_path, exist_ok=True)
        os.makedirs(gp_ckpt_path, exist_ok=True)
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/train",
            )
        )
        val_writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.output_dir + "/" + cfg.run_id + "/val",
            )
        )
    else:
        writer = None
        val_writer = None

    logger.info(f"Config:\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Output directory: {cfg.output_dir}/{cfg.run_id}")
    logger.info(
        f"Pretrained GeoTransolver (load only): {pretrained_ckpt_path}"
    )
    logger.info(f"GP checkpoint (save/load): {gp_ckpt_path}")

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    # Set up model
    # (Using partial convert to get lists, etc., instead of ListConfigs.)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")

    model.to(dist_manager.device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_manager.local_rank],
        output_device=dist_manager.device,
    )

    # Load pretrained GeoTransolver from train.py checkpoint dir (checkpoints/), not from checkpoints_gp
    pretrained_loaded = load_pretrained_model_only(
        model, pretrained_ckpt_path
    )
    for p in model.parameters():
        p.requires_grad = False
    if pretrained_loaded:
        logger.info("GeoTransolver loaded and frozen (weights will not be updated)")
    else:
        logger.warning(
            "Pretrained GeoTransolver was not loaded; running with randomly initialized weights"
        )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters (GeoTransolver, frozen): {num_params}")

    # Load the normalization file from configured directory (defaults to current dir)
    norm_dir = getattr(cfg.data, "normalization_dir", ".")
    if cfg.data.mode == "surface" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "surface_fields_normalization.npz")
        norm_data = np.load(norm_file)
        surface_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        surface_factors = None

    if cfg.data.mode == "volume" or cfg.data.mode == "combined":
        norm_file = str(Path(norm_dir) / "volume_fields_normalization.npz")
        norm_data = np.load(norm_file)
        volume_factors = {
            "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
            "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
        }
    else:
        volume_factors = None

    # Training dataset
    train_dataloader = create_transolver_dataset(
        cfg.data,
        phase="train",
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    # Validation dataset

    val_dataloader = create_transolver_dataset(
        cfg.data,
        phase="val",
        surface_factors=surface_factors,
        volume_factors=volume_factors,
    )

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataloader,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    precision = cfg.precision

    # Embedding reduction and GP (only these are trained; GeoTransolver is frozen)
    pooling_type = cfg.get("embedding_pooling", "attention")
    logger.info(f"Embedding pooling: {pooling_type}")
    embedding_reduction_model = create_embedding_reduction(
        pooling=pooling_type,
        feat_dim=256,
        embed_dim=32,
    )
    embedding_reduction_model.to(dist_manager.device)

    n_inducing = 128
    embed_dim = 32
    logger.info(
        f"Collecting {n_inducing} inducing-point embeddings from training data..."
    )
    init_embeddings = []
    model.eval()
    embedding_reduction_model.eval()
    with torch.no_grad():
        for i_init, batch_init in enumerate(train_dataloader):
            if len(init_embeddings) >= n_inducing:
                break
            features_init = cast_precisions(batch_init["fx"], precision)
            embeddings_init = cast_precisions(batch_init["embeddings"], precision)
            geometry_init = (
                cast_precisions(batch_init["geometry"], precision)
                if "geometry" in batch_init
                else None
            )
            local_positions_init = embeddings_init[:, :, :3]
            _, _, emb_states_init = model(
                global_embedding=features_init,
                local_embedding=embeddings_init,
                geometry=geometry_init,
                local_positions=local_positions_init,
            )
            reduced_init = embedding_reduction_model(emb_states_init.flatten(1, 2))
            init_embeddings.append(reduced_init.cpu())
    init_embeddings = torch.cat(init_embeddings, dim=0)[:n_inducing]
    logger.info(
        f"Inducing points initialised from data: shape {init_embeddings.shape}, "
        f"norm range [{init_embeddings.norm(dim=1).min():.4f}, "
        f"{init_embeddings.norm(dim=1).max():.4f}]"
    )

    gp = DragGP(
        embed_dim=embed_dim,
        n_inducing=n_inducing,
        n_train=len(train_dataloader),
        inducing_points=init_embeddings,
    )
    gp.to(dist_manager.device)

    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_reduction_model.parameters(), "lr": 1e-3},
            {"params": gp.gp_layer.variational_parameters(), "lr": 1e-2},
            {"params": gp.gp_layer.hyperparameters(), "lr": 1e-2},
            {"params": gp.likelihood.parameters(), "lr": 1e-2},
        ],
        weight_decay=1e-4,
    )

    # Set up learning rate scheduler based on config
    scheduler_cfg = cfg.training.scheduler
    scheduler_name = scheduler_cfg.name
    scheduler_params = dict(scheduler_cfg.params)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    scaler = GradScaler() if precision == "float16" else None

    if precision == "float8" and not TE_AVAILABLE:
        raise ImportError(
            "TransformerEngine is not installed.  Please install it to use float8 precision."
        )

    # Save/load only GP and embedding reduction to checkpoints_gp/ (GeoTransolver stays in checkpoints/)
    ckpt_args = {
        "path": gp_ckpt_path,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": [embedding_reduction_model, gp],
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)




    # Training loop
    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        # Set the epoch in the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_indices = list(train_sampler)
        val_indices = list(val_sampler)
        train_dataloader.dataset.set_indices(train_indices)
        val_dataloader.dataset.set_indices(val_indices)

        epoch_len = len(train_indices)
        val_epoch_len = len(val_indices)
        total_gp_loss = 0.0
        total_train_mse = 0.0
        accumulation_steps = getattr(
            cfg.training, "gradient_accumulation_steps", 1
        )

        model.eval()
        embedding_reduction_model.train()
        for i, batch in enumerate(train_dataloader):
            features = batch["fx"]
            embeddings = batch["embeddings"]
            targets = batch["fields"]

            # Cast precisions:
            features = cast_precisions(features, precision=precision)
            embeddings = cast_precisions(embeddings, precision=precision)
            if "geometry" in batch.keys():
                geometry = cast_precisions(batch["geometry"], precision=precision)
            else:
                geometry = None

            local_positions = embeddings[:, :, :3]

            with torch.no_grad():
                outputs, _, embedding_states = model(
                    global_embedding=features,
                    local_embedding=embeddings,
                    geometry=geometry,
                    local_positions=local_positions,
                )

            reduced_embeddings = embedding_reduction_model(embedding_states.flatten(1, 2))

            # True drag from batch: unnormalize fields, compute Cd, scale for GP (Cd / 0.35)
            drag_target = compute_drag_target_from_batch(
                batch, surface_factors, dist_manager.device
            ).to(reduced_embeddings.dtype)
            gp.train()
            gp.likelihood.train()

            gp_loss = gp.loss(reduced_embeddings, drag_target)

            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            (gp_loss / accumulation_steps).backward()

            this_loss = gp_loss.detach().item()
            total_gp_loss += this_loss

            with torch.no_grad():
                pred_mean = gp.gp_layer(reduced_embeddings).mean
                train_mse = torch.nn.functional.mse_loss(
                    pred_mean, drag_target
                ).item()
            total_train_mse += train_mse

            if (i + 1) % accumulation_steps == 0 or (i + 1) == epoch_len:
                optimizer.step()

            logger.info(
                f"Epoch {epoch} [{i}/{epoch_len}] GP Loss: {this_loss:.6f}  Train MSE: {train_mse:.6f}"
            )
            if dist_manager.rank == 0 and writer is not None:
                writer.add_scalar(
                    "batch/gp_loss", this_loss, i + epoch_len * epoch
                )

        avg_gp_loss = total_gp_loss / epoch_len
        avg_train_mse = total_train_mse / epoch_len
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] Avg Train GP Loss: {avg_gp_loss:.6f}  Avg Train MSE: {avg_train_mse:.6f}"
        )

        ls = gp.gp_layer.covar_module.base_kernel.lengthscale.detach().cpu()
        os_ = gp.gp_layer.covar_module.outputscale.detach().cpu().item()
        noise = gp.likelihood.noise.detach().cpu().item()
        logger.info(
            f"  GP hypers — lengthscale: min={ls.min():.4f} max={ls.max():.4f} "
            f"mean={ls.mean():.4f} | outputscale={os_:.6f} | noise={noise:.6f}"
        )
        logger.info(
            f"  last-batch embedding norm: {reduced_embeddings.detach().norm(dim=1).mean():.4f}"
        )

        if dist_manager.rank == 0 and writer is not None:
            writer.add_scalar("epoch/gp_loss", avg_gp_loss, epoch)
            writer.add_scalar("epoch/train_mse", avg_train_mse, epoch)
            writer.add_scalar("epoch/gp_lengthscale_mean", ls.mean().item(), epoch)
            writer.add_scalar("epoch/gp_outputscale", os_, epoch)
            writer.add_scalar("epoch/gp_noise", noise, epoch)

        # Validation: predict with GP and compute MSE vs true drag over full val set
        model.eval()
        embedding_reduction_model.eval()
        gp.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                features = batch["fx"]
                embeddings = batch["embeddings"]
                features = cast_precisions(features, precision=precision)
                embeddings = cast_precisions(embeddings, precision=precision)
                geometry = (
                    cast_precisions(batch["geometry"], precision=precision)
                    if "geometry" in batch
                    else None
                )
                local_positions = embeddings[:, :, :3]

                outputs, _, embedding_states = model(
                    global_embedding=features,
                    local_embedding=embeddings,
                    geometry=geometry,
                    local_positions=local_positions,
                )
                reduced_embeddings = embedding_reduction_model(
                    embedding_states.flatten(1, 2)
                )

                drag_target = compute_drag_target_from_batch(
                    batch, surface_factors, dist_manager.device
                ).to(reduced_embeddings.dtype)

                pred_mean, pred_var, lower_95, upper_95 = gp.predict(
                    reduced_embeddings
                )
                # Validation loss: MSE(predicted mean, true target)
                val_loss_batch = torch.nn.functional.mse_loss(
                    pred_mean, drag_target
                ).item()
                total_val_loss += val_loss_batch

                logger.info(
                    f"Val [{i}/{val_epoch_len}] GP MSE: {val_loss_batch:.6f}"
                )

        avg_val_loss = total_val_loss / val_epoch_len
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] Avg Val GP MSE: {avg_val_loss:.6f}"
        )
        if dist_manager.rank == 0 and val_writer is not None:
            val_writer.add_scalar("epoch/gp_mse", avg_val_loss, epoch)

        if epoch % getattr(cfg.training, "save_interval", 1) == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch + 1)

        scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig):
    main(cfg)

if __name__ == "__main__":
    launch()


