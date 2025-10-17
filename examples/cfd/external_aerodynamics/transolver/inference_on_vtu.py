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

import numpy as np
import torch
import torchinfo
import hydra
from omegaconf import DictConfig

import pyvista as pv
from physicsnemo.models.transolver.transolver import Transolver
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.launch.logging import RankZeroLoggingWrapper, PythonLogger

from physicsnemo.distributed import DistributedManager

import vtk
from vtk.util import numpy_support
import os
import math
import time

from train import (
    update_model_params_for_fp8,
    cast_precisions,
    get_autocast_context,
    pad_input_for_fp8,
    unpad_output_for_fp8,
)


def extract_airfrans_params(vtu_path: str) -> tuple[float, float, float]:
    """
    Extract air density, stream velocity, and angle of attack from AIRFRANS filename pattern.
    
    Args:
        vtu_path (str): Path to the VTU file.
        
    Returns:
        tuple: (air_density, stream_velocity, angle_of_attack)
    """
    # Default values
    air_density = 1.0
    stream_velocity = 30.0
    angle_of_attack = 0.0
    
    # Extract parameters from filename pattern
    # Format: airFoil2D_SST_Uinf_aoa_digit1_digit2_..._internal.vtu
    # Example: airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32_internal.vtu
    filename = os.path.basename(vtu_path)
    
    # Remove .vtu extension and _internal suffix
    name = filename.replace('.vtu', '').replace('_internal', '')
    
    # Split by underscore
    parts = name.split("_")
    
    # parts[0] = 'airFoil2D', parts[1] = 'SST', parts[2] = Uinf, parts[3] = aoa
    if len(parts) >= 3:
        try:
            stream_velocity = float(parts[2])
        except ValueError:
            pass
    if len(parts) >= 4:
        try:
            angle_of_attack = float(parts[3])
        except ValueError:
            pass
    
    # print(angle_of_attack)
    return air_density, stream_velocity, angle_of_attack


def read_data_from_vtu(
    vtu_path: str,
    air_density: float = None,
    stream_velocity: float = None,
    angle_of_attack: float = None,
) -> tuple:
    """
    Reads mesh and volume data from a VTU file and prepares a batch dictionary for inference.

    Args:
        vtu_path (str): Path to the VTU file.
        air_density (float, optional): Air density value. If None, extracts from filename.
        stream_velocity (float, optional): Stream velocity value. If None, extracts from filename.
        angle_of_attack (float, optional): Angle of attack value. If None, extracts from filename.

    Returns:
        tuple: (mesh, batch, stl_data) where:
            - mesh is a pyvista mesh object
            - batch is a dictionary of torch tensors for model input
            - stl_data is dictionary containing STL geometry information for COM calculation
    """

    dm = DistributedManager()
    
    # Extract parameters from filename if not provided
    if air_density is None or stream_velocity is None or angle_of_attack is None:
        extracted_density, extracted_velocity, extracted_aoa = extract_airfrans_params(vtu_path)
        if air_density is None:
            air_density = extracted_density
        if stream_velocity is None:
            stream_velocity = extracted_velocity
        if angle_of_attack is None:
            angle_of_attack = extracted_aoa

    mesh = pv.read(vtu_path)
    
    # Clip volume to box [(-2, 4), (-1.5, 1.5), (0, 1)]
    # This removes cells outside the region of interest
    mesh = mesh.clip_box([-2, 4, -1.5, 1.5, 0, 1], invert=False, crinkle=True)

    batch = {}

    # Extract volume mesh centers (POINTS/VERTICES - matching curator processing)
    # The curator uses get_vertices() which returns polydata.GetPoints(), so we use mesh.points
    batch["volume_mesh_centers"] = np.asarray(mesh.points)

    # Extract volume fields from POINT data (matching curator processing)
    # The curator uses polydata.GetPointData(), so we use mesh.point_data
    # Expected fields: U (velocity vector), p (pressure), nut (turbulent viscosity)
    # Combine into single array: [U_x, U_y, U_z, p, nut, implicit_distance]
    U = np.asarray(mesh.point_data["U"])  # Shape: (n_points, 3)
    p = np.asarray(mesh.point_data["p"])  # Shape: (n_points,)
    nut = np.asarray(mesh.point_data["nut"])  # Shape: (n_points,)
    
    # Read implicit_distance (pre-computed in the VTU file)
    if "implicit_distance" in mesh.point_data:
        implicit_distance = np.asarray(mesh.point_data["implicit_distance"])[:, np.newaxis]
    elif "implicit_distance" in mesh.cell_data:
        implicit_distance = np.asarray(mesh.cell_data["implicit_distance"])[:, np.newaxis]
    else:
        # Fallback: compute from surface if not available
        vtu_dir = os.path.dirname(vtu_path)
        vtp_files = [f for f in os.listdir(vtu_dir) if f.endswith('.vtp')]
        
        if len(vtp_files) == 0:
            implicit_distance = np.zeros((batch["volume_mesh_centers"].shape[0], 1), dtype=np.float32)
        else:
            vtp_path = os.path.join(vtu_dir, vtp_files[0])
            surface = pv.read(vtp_path)
            volume_centers = batch["volume_mesh_centers"]
            volume_centers_polydata = pv.PolyData(volume_centers)
            volume_centers_polydata = volume_centers_polydata.compute_implicit_distance(surface, inplace=False)
            implicit_distance = np.asarray(volume_centers_polydata['implicit_distance'])[:, np.newaxis].astype(np.float32)
    
    # Stack into volume_fields: [U_x, U_y, U_z, p, nut, implicit_distance]
    batch["volume_fields"] = np.column_stack([U, p[:, np.newaxis], nut[:, np.newaxis], implicit_distance])

    # Create scalars (shape [] not [1]) - will be unsqueezed to [1] later
    batch["air_density"] = np.array(air_density, dtype="float32")
    batch["stream_velocity"] = np.array(stream_velocity, dtype="float32")
    batch["angle_of_attack"] = np.array(angle_of_attack, dtype="float32")

    # For geometry data needed for COM calculation, read STL or VTP file
    # This matches the curator workflow (external_aero_geometry_data_processors.py)
    stl_data = {}
    
    # Find geometry file (STL or VTP) in the same directory as the VTU file
    vtu_dir = os.path.dirname(vtu_path)
    stl_files = [f for f in os.listdir(vtu_dir) if f.endswith('.stl')]
    vtp_files = [f for f in os.listdir(vtu_dir) if f.endswith('_aerofoil.vtp')]
    
    if len(stl_files) > 0:
        # 3D geometry (STL with triangular faces)
        stl_path = os.path.join(vtu_dir, stl_files[0])
        stl_polydata = pv.read(stl_path)
        stl_areas_mesh = stl_polydata.compute_cell_sizes(length=False, area=True, volume=False)
        stl_data["stl_areas"] = np.array(stl_areas_mesh.cell_data["Area"])
        stl_data["stl_centers"] = np.asarray(stl_polydata.cell_centers().points)
    elif len(vtp_files) > 0:
        # 2D geometry (VTP with line segments)
        vtp_path = os.path.join(vtu_dir, vtp_files[0])
        vtp_polydata = pv.read(vtp_path)
        # Compute line lengths for 2D geometry
        lengths_mesh = vtp_polydata.compute_cell_sizes(length=True, area=False, volume=False)
        stl_data["stl_areas"] = np.array(lengths_mesh.cell_data["Length"])
        stl_data["stl_centers"] = np.asarray(vtp_polydata.cell_centers().points)
    else:
        # Fallback: extract surface from VTU
        surface = mesh.extract_surface()
        stl_data["stl_centers"] = np.asarray(surface.cell_centers().points)
        surface_areas = surface.compute_cell_sizes(length=False, area=True, volume=False)
        stl_data["stl_areas"] = np.array(surface_areas.cell_data["Area"])

    # Convert to torch tensors
    batch = {
        k: torch.from_numpy(v).to(device=dm.device, dtype=torch.float32)
        for k, v in batch.items()
    }
    stl_data = {
        k: torch.from_numpy(v).to(device=dm.device, dtype=torch.float32)
        for k, v in stl_data.items()
    }

    # Add batch dimension
    batch = {k: torch.unsqueeze(v, dim=0) for k, v in batch.items()}
    stl_data = {k: torch.unsqueeze(v, dim=0) for k, v in stl_data.items()}

    return mesh, batch, stl_data


def preprocess_volume_data_inference(
    batch: dict,
    stl_data: dict,
    ignore_w_component: bool = False,
) -> tuple:
    """
    Preprocesses the batch data to generate node features and embeddings for the model (volume version).

    Args:
        batch (dict): Batch dictionary containing mesh and physical properties.
        stl_data (dict): Dictionary containing STL geometry for COM calculation.
        ignore_w_component (bool): Whether to ignore the w (U_z) component.

    Returns:
        tuple: (node_features, embeddings) as torch tensors.
    """

    mesh_centers = batch["volume_mesh_centers"]
    volume_fields = batch["volume_fields"]
    
    # Extract implicit_distance (last column) as SDF
    sdf = volume_fields[..., -1:]
    
    node_features = torch.stack(
        [batch["air_density"], batch["stream_velocity"], batch["angle_of_attack"]], dim=-1
    ).to(torch.float32)

    # Calculate center of mass using STL geometry
    sizes = stl_data["stl_areas"]
    centers = stl_data["stl_centers"]

    total_weighted_position = torch.einsum("ki,kij->kj", sizes, centers)
    total_size = torch.sum(sizes)
    center_of_mass = total_weighted_position[None, ...] / total_size

    # Subtract the COM from the centers:
    mesh_centers = mesh_centers - center_of_mass

    # Create Fourier features for positional encoding (matching training preprocessing)
    # fourier_sin_features = [
    #     torch.sin(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(2)
    # ]
    # fourier_cos_features = [
    #     torch.cos(mesh_centers * (2 ** i) * torch.pi)
    #     for i in range(2)
    # ]

    # Create embeddings: [mesh_centers, sdf, fourier_sin, fourier_cos]
    embeddings = torch.cat(
        [
            mesh_centers,
            sdf,
            # *fourier_sin_features,
            # *fourier_cos_features
        ],
        dim=-1,
    )
    
    # Expand features to match embedding shape (must unsqueeze first!)
    node_features = node_features.unsqueeze(1).expand(1, embeddings.shape[1], -1)

    return node_features, embeddings


def model_inference(model, features, embeddings, precision, output_pad_size):
    """
    Run model inference with proper precision handling.
    
    Args:
        model: The neural network model
        features: Input features tensor
        embeddings: Input embeddings tensor
        precision: Precision mode (float16, bfloat16, float8, or float32)
        output_pad_size: Output padding size for FP8
        
    Returns:
        Model outputs
    """
    # Cast precisions:
    features, embeddings = cast_precisions(features, embeddings, precision)
    with get_autocast_context(precision):
        # For fp8, we may have to pad the inputs:
        if precision == "float8":
            features = pad_input_for_fp8(features, embeddings)

        outputs = model(features, embeddings)

        outputs = unpad_output_for_fp8(outputs, output_pad_size)

    return outputs


def process_vtu_file(
    vtu_file: str,
    model: torch.nn.Module,
    norm_factors: dict,
    output_folder: str,
    precision: str,
    output_pad_size: int,
    ignore_w_component: bool = False,
    batch_size: int = 300_000,
) -> None:
    """
    Processes a single VTU file: runs inference, computes errors, and writes predictions to a new VTU file.

    Args:
        vtu_file (str): Path to the VTU file.
        model (torch.nn.Module): The trained model for inference.
        norm_factors (dict): Normalization factors for output.
        output_folder (str): Directory to save output files.
        precision (str): Precision mode for inference.
        output_pad_size (int): Output padding size for FP8.
        ignore_w_component (bool): Whether to ignore w component in predictions.
        batch_size (int, optional): Batch size for inference (number of points to process at once). Defaults to 300_000.

    Returns:
        None
    """

    # First, load the data and mesh from the file:
    try:
        mesh, batch, stl_data = read_data_from_vtu(vtu_file)
    except FileNotFoundError as e:
        return
    except Exception as e:
        return

    # Run preprocessing to prepare the data for the model:
    fx, embedding = preprocess_volume_data_inference(batch, stl_data, ignore_w_component)

    with torch.no_grad():
        if batch_size > fx.shape[1]:
            outputs = model_inference(model, fx, embedding, precision, output_pad_size)

            prediction = outputs * norm_factors["std"] + norm_factors["mean"]

        else:
            # Split the indices by a batch size.  We shuffle the cells into
            # the batches (don't forget to unshuffle later!)
            indices = torch.randperm(fx.shape[1], device=fx.device)

            index_blocks = torch.split(indices, batch_size)

            predictions = []
            for i, index_block in enumerate(index_blocks):
                local_fx = fx[:, index_block]
                local_embedding = embedding[:, index_block]

                # Just in the fp8 case, we have to pad the batch shape, too:
                sample_shape = local_fx.shape[1]
                if precision == "float8" and sample_shape % 8 != 0:
                    # NOTE: this padding is along axis 1, not -1!
                    padding = 8 - (sample_shape % 8)

                    # Create zero tensors to pad
                    fx_pad = torch.zeros(
                        *local_fx.shape[:1],
                        padding,
                        *local_fx.shape[2:],
                        dtype=local_fx.dtype,
                        device=local_fx.device,
                    )
                    emb_pad = torch.zeros(
                        *local_embedding.shape[:1],
                        padding,
                        *local_embedding.shape[2:],
                        dtype=local_embedding.dtype,
                        device=local_embedding.device,
                    )

                    # Concatenate along dim=1
                    local_fx = torch.cat([local_fx, fx_pad], dim=1)
                    local_embedding = torch.cat([local_embedding, emb_pad], dim=1)

                outputs = model_inference(
                    model, local_fx, local_embedding, precision, output_pad_size
                )

                # And, if we padded, we have to unpad the output:
                if precision == "float8" and sample_shape % 8 != 0:
                    outputs = outputs[:, :-padding, :]

                predictions.append(outputs * norm_factors["std"] + norm_factors["mean"])

            prediction = torch.cat(predictions, dim=1)

            # Now, we have to *unshuffle* the prediction to the original index
            inverse_indices = torch.empty_like(indices)
            inverse_indices[indices] = torch.arange(
                indices.size(0), device=indices.device
            )
            # Suppose prediction is of shape [batch, N, ...]
            prediction = prediction[:, inverse_indices]

        # Extract predicted fields and compute L2 errors PER VARIABLE (matching training)
        # prediction shape: [batch, n_points, n_fields]
        # Ground truth from volume_fields: [U_x, U_y, U_z, p, nut, implicit_distance]
        target_fields = batch["volume_fields"]
        
        # Extract only the fields we're predicting (exclude implicit_distance which is last column)
        # volume_fields: [U_x, U_y, U_z, p, nut, implicit_distance]
        target_fields = target_fields[..., :-1]  # Remove implicit_distance -> [U_x, U_y, U_z, p, nut]
        
        # Filter target_fields to match prediction if ignore_w_component=True
        # This matches training preprocessing (preprocess.py lines 112-118)
        if ignore_w_component:
            # Remove U_z component (index 2): [U_x, U_y, U_z, p, nut] -> [U_x, U_y, p, nut]
            target_fields = torch.cat([target_fields[..., :2], target_fields[..., 3:]], dim=-1)
        
        # Compute L2 exactly as in training (metrics.py lines 112-138)
        # L2 is computed per variable, summing only over points (dim=1)
        l2_num = (prediction - target_fields) ** 2
        l2_num = torch.sum(l2_num, dim=1)  # Sum over points only
        l2_num = torch.sqrt(l2_num)
        
        l2_denom = target_fields ** 2
        l2_denom = torch.sum(l2_denom, dim=1)  # Sum over points only
        l2_denom = torch.sqrt(l2_denom)
        
        l2 = l2_num / (l2_denom + 1e-10)  # Shape: [batch, num_variables]
        
        # Extract predictions for saving to file
        if ignore_w_component:
            # prediction has 4 components: [U_x, U_y, p, nut]
            pred_U_x = prediction[:, :, 0]
            pred_U_y = prediction[:, :, 1]
            pred_p = prediction[:, :, 2]
            pred_nut = prediction[:, :, 3]
            
            # Create full velocity vector with zero w component for saving
            pred_U = torch.stack([pred_U_x, pred_U_y, torch.zeros_like(pred_U_x)], dim=-1)
            
            # L2 errors per variable
            l2_u = l2[0, 0].item()
            l2_v = l2[0, 1].item()
            l2_p = l2[0, 2].item()
            l2_nut = l2[0, 3].item()
            
            print(f"L2 U_x: {l2_u:.6f}")
            print(f"L2 U_y: {l2_v:.6f}")
            print(f"L2 p: {l2_p:.6f}")
            print(f"L2 nut: {l2_nut:.6f}")
        else:
            # prediction has 5 components: [U_x, U_y, U_z, p, nut]
            pred_U = prediction[:, :, :3]
            pred_p = prediction[:, :, 3]
            pred_nut = prediction[:, :, 4]
            
            # L2 errors per variable
            l2_u = l2[0, 0].item()
            l2_v = l2[0, 1].item()
            l2_w = l2[0, 2].item()
            l2_p = l2[0, 3].item()
            l2_nut = l2[0, 4].item()
            
            print(f"L2 U_x: {l2_u:.6f}")
            print(f"L2 U_y: {l2_v:.6f}")
            print(f"L2 U_z: {l2_w:.6f}")
            print(f"L2 p: {l2_p:.6f}")
            print(f"L2 nut: {l2_nut:.6f}")

    # Write the output to a new .vtu file.  Clone the old information:
    output_mesh = mesh.copy()
    
    # Convert tensors to numpy arrays and squeeze batch dimension
    pred_U_np = pred_U[0].cpu().numpy()
    pred_p_np = pred_p[0].cpu().numpy()
    pred_nut_np = pred_nut[0].cpu().numpy()
    # Extract implicit_distance from volume_fields (last column)
    sdf_np = batch["volume_fields"][0, :, -1].cpu().numpy()
    
    # Add arrays to the mesh as POINT data (matching what we loaded)
    output_mesh.point_data["PredictedU"] = pred_U_np
    output_mesh.point_data["Predictedp"] = pred_p_np
    output_mesh.point_data["Predictednut"] = pred_nut_np
    output_mesh.point_data["sdf"] = sdf_np
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Construct output file path
    base_name = os.path.basename(vtu_file)
    output_path = os.path.join(output_folder, f"pred_{base_name}")
    
    # Write to file
    output_mesh.save(output_path)
    # print(f"Saved prediction VTU to: {output_path}")


def inference_on_vtu(cfg: DictConfig) -> None:
    """
    Main inference loop for processing multiple VTU files using the provided configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        None
    """

    DistributedManager.initialize()

    dist_manager = DistributedManager()

    run_id = cfg.run_id

    logger = RankZeroLoggingWrapper(PythonLogger(name="inference"), dist_manager)

    cfg, output_pad_size = update_model_params_for_fp8(cfg, logger)

    # Set up model
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"\n{torchinfo.summary(model, verbose=0)}")
    model.eval()
    model.to(dist_manager.device)

    ckpt_args = {
        "path": f"{cfg.output_dir}/{cfg.run_id}/checkpoints",
        "models": model,
    }

    # Load the normalization factors:
    norm_file = "volume_fields_normalization.npz"
    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }

    # Restore the model:
    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)
    print(f"Loaded model from epoch: {loaded_epoch}")

    # Configure input/output paths
    # Modify these paths according to your data location
    vtu_input_path = cfg.get("inference_input_path", "/workspace/airfoil-experiments/Dataset_orig/")
    vtu_output_path = cfg.get("inference_output_path", f"/workspace/airfoil-experiments/physicsnemo/examples/cfd/external_aerodynamics/transolver/test/{run_id}/")

    # Get list of VTU files to process
    # Example: process files in a directory or a specific list
    if os.path.isdir(vtu_input_path):
        # Recursively search for all .vtu files in the directory and subdirectories
        all_files = []
        for root, dirs, files in os.walk(vtu_input_path):
            for f in files:
                if f.endswith(".vtu"):
                    all_files.append(os.path.join(root, f))
    elif os.path.isfile(vtu_input_path) and vtu_input_path.endswith(".vtu"):
        all_files = [vtu_input_path]
    else:
        # Assume it's a pattern or list specified in config
        all_files = []
        logger.warning(f"No VTU files found at: {vtu_input_path}")

    # Remove files that already have predictions
    filtered_files = []
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        output_path = os.path.join(vtu_output_path, f"pred_{base_name}")
        if not os.path.exists(output_path):
            filtered_files.append(file_path)
    all_files = filtered_files

    logger.info(f"Processing {len(all_files)} VTU files")

    # Distribute files across GPUs
    this_device_files = all_files[dist_manager.rank :: dist_manager.world_size]

    print(
        f"Rank {dist_manager.rank} of {dist_manager.world_size} is processing {len(this_device_files)} files"
    )
    
    ignore_w_component = cfg.data.get("ignore_w_component", False)
    
    for vtu_file in this_device_files:
        start = time.time()

        # Process files:
        process_vtu_file(
            vtu_file,
            model,
            norm_factors,
            vtu_output_path,
            precision=cfg.training.precision,
            output_pad_size=output_pad_size,
            ignore_w_component=ignore_w_component,
            batch_size=cfg.data.resolution,
        )
        end = time.time()
        print(f"Processed {vtu_file} in {end - start:.2f}s")


@hydra.main(version_base=None, config_path="conf", config_name="train")
def launch(cfg: DictConfig) -> None:
    """Launch inference with hydra configuration

    Args:
        cfg: Hydra configuration object
    """
    inference_on_vtu(cfg)


if __name__ == "__main__":
    launch()

