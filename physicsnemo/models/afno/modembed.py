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

from typing import Type

import torch
from jaxtyping import Float
from torch import Tensor, nn

from physicsnemo.core.module import Module


class PositionalEmbedding(Module):
    r"""Module for generating sinusoidal positional embeddings based on timesteps.

    Parameters
    ----------
    num_channels : int
        Number of channels for the embedding. Should be even.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B,)` or :math:`(B, 1)` containing timesteps.
        Type hint ``B ...`` accepts both shapes.

    Outputs
    -------
    torch.Tensor
        Positional embedding of shape :math:`(B, D)` where :math:`D` is
        ``num_channels``.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        freqs = torch.pi * torch.arange(
            start=1, end=self.num_channels // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing sinusoidal embeddings."""
        x = x.view(-1).outer(self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class OneHotEmbedding(Module):
    r"""Module for generating soft one-hot embeddings based on timesteps.

    The embedding uses a soft one-hot encoding where the value at each position
    is based on the distance to the timestep.

    Parameters
    ----------
    num_channels : int
        Number of channels for the embedding.

    Forward
    -------
    t : torch.Tensor
        Input tensor of shape :math:`(B,)` or :math:`(B, 1)` containing normalized
        timesteps in range ``[0, 1]``.

    Outputs
    -------
    torch.Tensor
        Soft one-hot embedding of shape :math:`(B, D)` where :math:`D` is
        ``num_channels``. Type hint ``B ...`` accepts :math:`(B,)` or :math:`(B, 1)`.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        ind = torch.arange(num_channels)
        ind = ind.view(1, len(ind))
        self.register_buffer("indices", ind)

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing soft one-hot embeddings."""
        ind = t * (self.num_channels - 1)
        return torch.clamp(1 - torch.abs(ind - self.indices), min=0)


class ModEmbedNet(Module):
    r"""Network that generates a timestep embedding and processes it with an MLP.

    Parameters
    ----------
    max_time : float, optional, default=1.0
        Maximum input time. The inputs to ``forward`` should be in the range
        ``[0, max_time]``.
    dim : int, optional, default=64
        The dimensionality of the time embedding.
    depth : int, optional, default=1
        The number of layers in the MLP.
    activation_fn : Type[nn.Module], optional, default=nn.GELU
        The activation function class.
    method : str, optional, default="sinusoidal"
        The embedding method. Either ``"sinusoidal"`` or ``"onehot"``.

    Forward
    -------
    t : torch.Tensor
        Input tensor of shape :math:`(B,)` or :math:`(B, 1)` containing timesteps
        in range ``[0, max_time]``.

    Outputs
    -------
    torch.Tensor
        Embedding of shape :math:`(B, D)` where :math:`D` is ``dim``.

    Examples
    --------
    >>> import torch
    >>> embed_net = ModEmbedNet(max_time=1.0, dim=64, depth=2)
    >>> t = torch.tensor([0.0, 0.5, 1.0])
    >>> embedding = embed_net(t)
    >>> embedding.shape
    torch.Size([3, 64])

    See Also
    --------
    :mod:`~physicsnemo.nn.module.embedding_layers` :
        Other embedding layers (e.g. :class:`~physicsnemo.nn.module.embedding_layers.FourierEmbedding`,
        :class:`~physicsnemo.nn.module.embedding_layers.PositionalEmbedding`).
    """

    def __init__(
        self,
        max_time: float = 1.0,
        dim: int = 64,
        depth: int = 1,
        activation_fn: Type[nn.Module] = nn.GELU,
        method: str = "sinusoidal",
    ):
        super().__init__()
        self.max_time = max_time
        self.method = method
        if method == "onehot":
            self.onehot_embed = OneHotEmbedding(dim)
        elif method == "sinusoidal":
            self.sinusoid_embed = PositionalEmbedding(dim)
        else:
            raise ValueError(f"Embedding '{method}' not supported")

        self.dim = dim

        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: Float[Tensor, "B ..."]) -> Float[Tensor, "B D"]:
        r"""Forward pass computing the modulation embedding."""
        # Normalize time to [0, 1]
        t = t / self.max_time

        # Compute base embedding
        if self.method == "onehot":
            emb = self.onehot_embed(t)
        elif self.method == "sinusoidal":
            emb = self.sinusoid_embed(t)

        # Process through MLP
        return self.mlp(emb)
