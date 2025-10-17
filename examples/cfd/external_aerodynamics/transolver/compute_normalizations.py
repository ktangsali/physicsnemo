# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import torch
from omegaconf import OmegaConf

from datapipe import DomainParallelZarrDataset

"""
This file provides utilities to compute normalization statistics (mean, std, min, max)
for a given field in a dataset, typically used for preprocessing in CFD workflows.
"""


def compute_mean_std_min_max(
    dataset: DomainParallelZarrDataset, field_key: str, ignore_w_component: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean, standard deviation, minimum, and maximum for a specified field
    across all samples in a dataset.

    Uses a numerically stable online algorithm for mean and variance.

    Args:
        dataset (DomainParallelZarrDataset): The dataset to process.
        field_key (str): The key for the field to normalize.
        ignore_w_component (bool): If True, exclude w (U_z) component for volume fields.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, std, min, max tensors for the field.
    """
    N = 0  # Total number of elements processed
    mean = None
    M2 = None  # Sum of squares of differences from the current mean
    min_val = None
    max_val = None

    for i in range(len(dataset)):
        print(f"reading file: {i}")
        data = dataset[i][field_key]
        
        # Exclude w (U_z) component and implicit_distance if specified
        # volume_fields: [U_x, U_y, U_z, p, nut, implicit_distance]
        if ignore_w_component and field_key == "volume_fields":
            # Exclude U_z (index 2) and implicit_distance (index 5, last column)
            data = torch.cat([data[..., :2], data[..., 3:5]], dim=-1)  # [U_x, U_y, p, nut]
        
        if mean is None:
            # Initialize accumulators based on the shape of the data
            mean = torch.zeros(data.shape[-1], device=data.device)
            M2 = torch.zeros(data.shape[-1], device=data.device)
            min_val = torch.full((data.shape[-1],), float("inf"), device=data.device)
            max_val = torch.full((data.shape[-1],), float("-inf"), device=data.device)
        n = data.shape[1]
        N += n

        # Compute batch statistics
        batch_mean = data.mean(axis=(0, 1))
        batch_M2 = ((data - batch_mean) ** 2).sum(axis=(0, 1))
        batch_n = data.shape[1]

        # Update min/max
        batch_min = data.amin(dim=(0, 1))
        batch_max = data.amax(dim=(0, 1))
        min_val = torch.minimum(min_val, batch_min)
        max_val = torch.maximum(max_val, batch_max)

        # Update running mean and M2 (Welford's algorithm)
        delta = batch_mean - mean
        N += batch_n
        mean = mean + delta * (batch_n / N)
        M2 = M2 + batch_M2 + delta**2 * (batch_n * N) / N

    var = M2 / (N - 1)
    std = torch.sqrt(var)
    return mean, std, min_val, max_val


if __name__ == "__main__":
    """
    Script entry point for computing normalization statistics for a specified field
    in a dataset, using configuration from a YAML file.

    The computed statistics are printed and saved to a .npz file.
    
    Usage:
        python compute_normalizations.py                    # Uses mode from config
        python compute_normalizations.py --mode surface     # Force surface mode
        python compute_normalizations.py --mode volume      # Force volume mode
    """
    import sys
    
    config_path: str = "conf/train.yaml"
    cfg = OmegaConf.load(config_path)
    
    mode = None
    if len(sys.argv) > 2 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
    else:
        mode = cfg.data.mode
    
    if mode == "surface":
        field_key = "surface_fields"
    elif mode == "volume":
        field_key = "volume_fields"
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'surface' or 'volume'")
    
    # Get ignore_w_component from config (only applicable for volume mode)
    ignore_w_component = cfg.data.get("ignore_w_component", False) if mode == "volume" else False
    
    print(f"Computing normalization for {mode} mode, field: {field_key}")
    if ignore_w_component:
        print("Note: Excluding w (U_z) component and implicit_distance from volume normalization")
        print("      Normalizing only: [U_x, U_y, p, nut]")

    dataset = DomainParallelZarrDataset(
        data_path=cfg.data.train.data_path,
        device_mesh=None,
        placements=None,
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=[field_key],
        large_keys=[field_key],
    )

    mean, std, min_val, max_val = compute_mean_std_min_max(
        dataset, field_key, ignore_w_component
    )
    print(f"Mean for {field_key}: {mean}")
    print(f"Std for {field_key}: {std}")
    print(f"Min for {field_key}: {min_val}")
    print(f"Max for {field_key}: {max_val}")

    output_file = f"{field_key}_normalization.npz"
    np.savez(
        output_file,
        mean=mean.cpu().numpy(),
        std=std.cpu().numpy(),
        min=min_val.cpu().numpy(),
        max=max_val.cpu().numpy(),
    )
    print(f"\nSaved normalization statistics to {output_file}")
