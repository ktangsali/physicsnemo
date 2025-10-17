# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch

from physicsnemo.distributed.shard_tensor import ShardTensor
from physicsnemo.utils.profiling import profile


@profile
def preprocess_surface_data(
    batch: dict,
    norm_factors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Preprocess the surface data.  The functional input
    is the air density and stream velocity.  The embeddings
    are the surface mesh centers and normals.  The targets are
    normalized to mean of 0, std 1.  We cache the mean and std
    to de-normalize when computing the metrics.
    """

    mesh_centers = batch["surface_mesh_centers"]
    normals = batch["surface_normals"]
    targets = batch["surface_fields"]
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"], batch["angle_of_attack"]], dim=-1
    ).to(torch.float32)

    # Normalize the surface fields:
    targets = (targets - norm_factors["mean"]) / norm_factors["std"]

    # If you want to use this, be sure to update the
    # functional_dim value in your configuration

    # fourier_sin_features = [
    #     torch.sin(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(4)
    # ]
    # fourier_cos_features = [
    #     torch.cos(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(4)
    # ]

    # Calculate center of mass
    sizes = batch["stl_areas"]
    centers = batch["stl_centers"]

    total_weighted_position = torch.einsum("ki,kij->kj", sizes, centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size

    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    embeddings = torch.cat(
        [
            mesh_centers,
            normals,
            # *fourier_sin_features,
            # *fourier_cos_features
        ],
        dim=-1,
    )

    others = {
        "surface_areas": sizes,
        "surface_normals": normals,
        "stream_velocity": batch["stream_velocity"],
        "air_density": batch["air_density"],
        "angle_of_attack": batch["angle_of_attack"],
    }

    return node_features, embeddings, targets, others

@profile
def preprocess_volume_data(
    batch: dict,
    norm_factors: dict[str, torch.Tensor],
    ignore_w_component: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Preprocess the volume data.  The functional input
    is the air density and stream velocity. The embeddings
    are the volume mesh centers. The targets are
    normalized to mean of 0, std 1.  We cache the mean and std
    to de-normalize when computing the metrics.
    
    Args:
        batch: Batch of data
        norm_factors: Normalization factors
        ignore_w_component: If True, exclude w (U_z) component from targets (for 2D sims)
    """

    mesh_centers = batch["volume_mesh_centers"]
    volume_fields = batch["volume_fields"]
    
    # Extract implicit_distance (last column) as SDF
    # volume_fields: [U_x, U_y, U_z, p, nut, implicit_distance]
    sdf = volume_fields[..., -1:]  # Keep as (N, 1)
    targets = volume_fields[..., :-1]  # Everything except implicit_distance
    
    # Exclude w (U_z) component if specified (index 2: U_x=0, U_y=1, U_z=2, p=3, nut=4)
    if ignore_w_component:
        targets = torch.cat([targets[..., :2], targets[..., 3:]], dim=-1)
    
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"], batch["angle_of_attack"]], dim=-1
    ).to(torch.float32)

    # Normalize the volume fields:
    targets = (targets - norm_factors["mean"]) / norm_factors["std"]

    # Calculate center of mass
    sizes = batch["stl_areas"]
    centers = batch["stl_centers"]
    total_weighted_position = torch.einsum("ki,kij->kj", sizes, centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size
    
    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    # Create Fourier features for positional encoding
    # fourier_sin_features = [
    #     torch.sin(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(2)
    # ]
    # fourier_cos_features = [
    #     torch.cos(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(2)
    # ]

    embeddings = torch.cat(
        [
            mesh_centers,
            sdf,
            # *fourier_sin_features,
            # *fourier_cos_features
        ],
        dim=-1,
    )

    # print(batch["stream_velocity"], batch["air_density"])
    # exit()
    others = {
        "stream_velocity": batch["stream_velocity"],
        "air_density": batch["air_density"],
        "angle_of_attack": batch["angle_of_attack"],
    }

    return node_features, embeddings, targets, others



@profile
def downsample_surface(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_keep=1024,
):
    if num_keep == -1:
        features = features.unsqueeze(1).expand(1, embeddings.shape[1], -1)
        return features, embeddings, targets

    """
    Downsample the surface data. We generate one set of indices, and
    use it to sample the same points from the features, embeddings,
    and targets.  Using torch.multinomial to sample without replacement.
    """

    num_samples = embeddings.shape[1]
    # Generate random indices to keep (faster for large num_samples)
    indices = torch.multinomial(
        torch.ones(num_samples, device=features.device), num_keep, replacement=False
    )

    # Use the same indices to downsample all tensors
    downsampled_embeddings = embeddings[:, indices]
    downsampled_targets = targets[:, indices]
    # This unsqueezes the features (air density and stream velocity) to
    # the same shape as the embeddings
    downsampled_features = features.unsqueeze(1).expand(
        1, downsampled_embeddings.shape[1], -1
    )

    return downsampled_features, downsampled_embeddings, downsampled_targets


@profile
def downsample_volume(
    features: torch.Tensor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_keep=300_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Downsample the volume data. We generate one set of indices, and
    use it to sample the same points from the features, embeddings,
    and targets.  Using torch.multinomial to sample without replacement.
    
    Args:
        features: Input features of shape (batch, feature_dim)
        embeddings: Embeddings of shape (batch, num_points, embedding_dim)
        targets: Targets of shape (batch, num_points, target_dim)
        num_keep: Number of points to keep. If -1, keep all points.
    
    Returns:
        Downsampled features, embeddings, and targets
    """
    if num_keep == -1:
        features = features.unsqueeze(1).expand(1, embeddings.shape[1], -1)
        return features, embeddings, targets

    num_samples = embeddings.shape[1]
    
    # If num_keep is greater than available samples, keep all
    if num_keep >= num_samples:
        features = features.unsqueeze(1).expand(1, embeddings.shape[1], -1)
        return features, embeddings, targets
    
    # Generate random indices to keep
    indices = torch.multinomial(
        torch.ones(num_samples, device=features.device), num_keep, replacement=False
    )

    # Use the same indices to downsample all tensors
    downsampled_embeddings = embeddings[:, indices]
    downsampled_targets = targets[:, indices]
    # Expand features to match the downsampled shape
    downsampled_features = features.unsqueeze(1).expand(
        1, downsampled_embeddings.shape[1], -1
    )

    return downsampled_features, downsampled_embeddings, downsampled_targets
