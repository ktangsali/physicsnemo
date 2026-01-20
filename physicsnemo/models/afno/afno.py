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

from dataclasses import dataclass
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

import physicsnemo  # noqa: F401 for docs
import physicsnemo.nn.fft as fft
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

Tensor = torch.Tensor


class AFNOMlp(nn.Module):
    r"""Fully-connected Multi-layer perception used inside AFNO.

    Parameters
    ----------
    in_features : int
        Input feature size.
    latent_features : int
        Latent feature size.
    out_features : int
        Output feature size.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function.
    drop : float, optional, default=0.0
        Drop out rate.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(*, D_{in})` where :math:`D_{in}` is
        ``in_features``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(*, D_{out})` where :math:`D_{out}` is
        ``out_features``.
    """

    def __init__(
        self,
        in_features: int,
        latent_features: int,
        out_features: int,
        activation_fn: nn.Module = nn.GELU(),
        drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, latent_features)
        self.act = activation_fn
        self.fc2 = nn.Linear(latent_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2DLayer(nn.Module):
    r"""AFNO spectral convolution layer.

    This layer performs spectral mixing using block-diagonal weight matrices
    in the Fourier domain with soft shrinkage for sparsity.

    Parameters
    ----------
    hidden_size : int
        Feature dimensionality.
    num_blocks : int, optional, default=8
        Number of blocks used in the block diagonal weight matrix.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1
        Threshold for limiting number of modes used, in range ``[0, 1]``.
    hidden_size_factor : int, optional, default=1
        Factor to increase spectral features by after weight multiplication.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, C)` where :math:`B` is batch size,
        :math:`H, W` are spatial dimensions, and :math:`C` is ``hidden_size``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, H, W, C)`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1,
        hidden_size_factor: int = 1,
    ):
        super().__init__()
        if not (hidden_size % num_blocks == 0):
            raise ValueError(
                f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"
            )

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size)
        )

    def forward(
        self, x: Float[Tensor, "batch height width channels"]
    ) -> Float[Tensor, "batch height width channels"]:
        r"""Forward pass of the AFNO spectral layer."""
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        # Apply 2D FFT in the spatial dimensions
        x = fft.rfft2(x, dim=(1, 2), norm="ortho")
        x_real, x_imag = fft.real(x), fft.imag(x)
        x_real = x_real.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)
        x_imag = x_imag.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                H,
                W // 2 + 1,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2 = torch.zeros(x_real.shape + (2,), device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_real[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[0],
                )
                - torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_imag[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[1],
                )
                + self.b1[0]
            )
        )

        o1_imag[:, total_modes - kept_modes : total_modes + kept_modes, :kept_modes] = (
            F.relu(
                torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_imag[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[0],
                )
                + torch.einsum(
                    "nyxbi,bio->nyxbo",
                    x_real[
                        :,
                        total_modes - kept_modes : total_modes + kept_modes,
                        :kept_modes,
                    ],
                    self.w1[1],
                )
                + self.b1[1]
            )
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 0
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            - torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2[
            :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, ..., 1
        ] = (
            torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_imag[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[0],
            )
            + torch.einsum(
                "nyxbi,bio->nyxbo",
                o1_real[
                    :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes
                ],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = fft.view_as_complex(x)
        # TODO(akamenev): replace the following branching with
        # a one-liner, something like x.reshape(..., -1).squeeze(-1),
        # but this currently fails during ONNX export.
        if torch.onnx.is_in_onnx_export():
            x = x.reshape(B, H, W // 2 + 1, C, 2)
        else:
            x = x.reshape(B, H, W // 2 + 1, C)
        # Using ONNX friendly FFT functions
        x = fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    r"""AFNO block consisting of spectral convolution and MLP.

    Parameters
    ----------
    embed_dim : int
        Embedded feature dimensionality.
    num_blocks : int, optional, default=8
        Number of blocks used in the block diagonal weight matrix.
    mlp_ratio : float, optional, default=4.0
        Ratio of MLP latent variable size to input feature size.
    drop : float, optional, default=0.0
        Drop out rate in MLP.
    activation_fn : nn.Module, optional, default=nn.GELU()
        Activation function used in MLP.
    norm_layer : nn.Module, optional, default=nn.LayerNorm
        Normalization function.
    double_skip : bool, optional, default=True
        Whether to use double skip connections.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1.0
        Threshold for limiting number of modes used, in range ``[0, 1]``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, H, W, C)` where :math:`B` is batch size,
        :math:`H, W` are spatial dimensions, and :math:`C` is ``embed_dim``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, H, W, C)`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        activation_fn: nn.Module = nn.GELU(),
        norm_layer: nn.Module = nn.LayerNorm,
        double_skip: bool = True,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.filter = AFNO2DLayer(
            embed_dim, num_blocks, sparsity_threshold, hard_thresholding_fraction
        )
        self.norm2 = norm_layer(embed_dim)
        mlp_latent_dim = int(embed_dim * mlp_ratio)
        self.mlp = AFNOMlp(
            in_features=embed_dim,
            latent_features=mlp_latent_dim,
            out_features=embed_dim,
            activation_fn=activation_fn,
            drop=drop,
        )
        self.double_skip = double_skip

    def forward(
        self, x: Float[Tensor, "batch height width channels"]
    ) -> Float[Tensor, "batch height width channels"]:
        r"""Forward pass of the AFNO block."""
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class PatchEmbed(nn.Module):
    r"""Patch embedding layer.

    Converts 2D patches into a 1D vector sequence for input to AFNO.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions as ``[height, width]``.
    in_channels : int
        Number of input channels.
    patch_size : List[int], optional, default=[16, 16]
        Size of image patches as ``[patch_height, patch_width]``.
    embed_dim : int, optional, default=256
        Embedded channel size.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`H, W` are
        spatial dimensions matching ``inp_shape``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, N, D)` where :math:`N` is the number of
        patches and :math:`D` is ``embed_dim``.
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        num_patches = (inp_shape[1] // patch_size[1]) * (inp_shape[0] // patch_size[0])
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch num_patches embed_dim"]:
        r"""Forward pass of patch embedding."""
        # Input validation
        if not torch.compiler.is_compiling():
            B, C, H, W = x.shape
            if not (H == self.inp_shape[0] and W == self.inp_shape[1]):
                raise ValueError(
                    f"Input image size ({H}*{W}) doesn't match model "
                    f"({self.inp_shape[0]}*{self.inp_shape[1]})."
                )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class AFNO(Module):
    r"""Adaptive Fourier neural operator (AFNO) model.

    AFNO is a model that is designed for 2D images only. It combines patch
    embedding with spectral convolution blocks in the Fourier domain.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions as ``[height, width]``.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    patch_size : List[int], optional, default=[16, 16]
        Size of image patches as ``[patch_height, patch_width]``.
    embed_dim : int, optional, default=256
        Embedded channel size.
    depth : int, optional, default=4
        Number of AFNO layers.
    mlp_ratio : float, optional, default=4.0
        Ratio of layer MLP latent variable size to input feature size.
    drop_rate : float, optional, default=0.0
        Drop out rate in layer MLPs.
    num_blocks : int, optional, default=16
        Number of blocks in the block-diag frequency weight matrices.
    sparsity_threshold : float, optional, default=0.01
        Sparsity threshold (softshrink) of spectral features.
    hard_thresholding_fraction : float, optional, default=1.0
        Threshold for limiting number of modes used, in range ``[0, 1]``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)` where :math:`B` is batch
        size, :math:`C_{in}` is the number of input channels, and :math:`H, W` are
        spatial dimensions matching ``inp_shape``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)` where :math:`C_{out}` is
        ``out_channels``.

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32)  # (N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """

    def __init__(
        self,
        inp_shape: List[int],
        in_channels: int,
        out_channels: int,
        patch_size: List[int] = [16, 16],
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__(meta=MetaData())
        if len(inp_shape) != 2:
            raise ValueError("inp_shape should be a list of length 2")
        if len(patch_size) != 2:
            raise ValueError("patch_size should be a list of length 2")

        if not (
            inp_shape[0] % patch_size[0] == 0 and inp_shape[1] % patch_size[1] == 0
        ):
            raise ValueError(
                f"input shape {inp_shape} should be divisible by patch_size {patch_size}"
            )

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.inp_shape = inp_shape
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            inp_shape=inp_shape,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = inp_shape[0] // self.patch_size[0]
        self.w = inp_shape[1] // self.patch_size[1]

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        r"""Initialize model weights.

        Parameters
        ----------
        m : nn.Module
            Module to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch h w embed_dim"]:
        r"""Forward pass of core AFNO feature extraction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(B, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Features of shape :math:`(B, h, w, D)` where :math:`h, w` are patch
            grid dimensions and :math:`D` is ``embed_dim``.
        """
        B = x.shape[0]

        # Embed patches and add positional encoding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Reshape to 2D grid and apply blocks
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        r"""Forward pass of the AFNO model."""
        # Input validation
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected 4D input tensor (B, C, H, W), got {x.ndim}D tensor "
                    f"with shape {tuple(x.shape)}"
                )
            B, C, H, W = x.shape
            if H != self.inp_shape[0] or W != self.inp_shape[1]:
                raise ValueError(
                    f"Expected input spatial dimensions {self.inp_shape}, "
                    f"got ({H}, {W})"
                )

        # Extract features through AFNO blocks
        x = self.forward_features(x)

        # Project to output channels
        x = self.head(x)

        # Reshape tensor back into [B, C, H, W]
        out = x.view(list(x.shape[:-1]) + [self.patch_size[0], self.patch_size[1], -1])
        out = torch.permute(out, (0, 5, 1, 3, 2, 4))
        out = out.reshape(list(out.shape[:2]) + [self.inp_shape[0], self.inp_shape[1]])

        return out
