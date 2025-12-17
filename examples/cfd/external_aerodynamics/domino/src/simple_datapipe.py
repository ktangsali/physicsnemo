# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified DoMINO DataPipe - clean and easy to follow.
Removes: bounding box filtering, domain parallelism, complex sharding.
"""

from pathlib import Path
from typing import Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from physicsnemo.datapipes.cae.cae_dataset import CAEDataset
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.domino.utils import (
    calculate_center_of_mass,
    create_grid,
    normalize,
    shuffle_array,
    standardize,
    unnormalize,
    unstandardize,
)
from physicsnemo.utils.neighbors import knn
from physicsnemo.utils.sdf import signed_distance_field


class SimpleDoMINODataPipe(Dataset):
    """
    Simplified DoMINO DataPipe.
    
    Main flow:
    1. Load data from CAEDataset
    2. Process surface: filter invalid, sample, kNN for neighbors, normalize
    3. Process volume: sample, compute SDF, normalize
    4. Return dictionary with all keys
    """
    
    def __init__(
        self,
        data_path: str,
        phase: Literal["train", "val", "test"],
        model_type: Literal["surface", "volume", "combined"],
        # Grid settings
        grid_resolution: tuple = (128, 64, 64),
        # Bounding boxes [min, max]
        bounding_box_volume: tuple = None,
        bounding_box_surface: tuple = None,
        # Sampling
        sampling: bool = True,
        volume_points_sample: int = 8192,
        surface_points_sample: int = 8192,
        geom_points_sample: int = 300000,
        num_surface_neighbors: int = 7,
        surface_sampling_algorithm: str = "area_weighted",
        # Normalization
        normalize_coordinates: bool = True,
        scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None,
        volume_factors: Optional[torch.Tensor] = None,
        surface_factors: Optional[torch.Tensor] = None,
        # Device
        gpu_preprocessing: bool = True,
        gpu_output: bool = True,
    ):
        self.data_path = Path(data_path)
        self.phase = phase
        self.model_type = model_type
        
        # Sampling settings
        self.sampling = sampling
        self.volume_points_sample = volume_points_sample
        self.surface_points_sample = surface_points_sample
        self.geom_points_sample = geom_points_sample
        self.num_surface_neighbors = num_surface_neighbors
        self.surface_sampling_algorithm = surface_sampling_algorithm
        
        # Normalization
        self.normalize_coordinates = normalize_coordinates
        self.scaling_type = scaling_type
        
        # Setup distributed manager
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        dist = DistributedManager()
        
        # Set devices
        self.preproc_device = dist.device if gpu_preprocessing else torch.device("cpu")
        self.output_device = dist.device if gpu_output else torch.device("cpu")
        
        # Setup bounding boxes and grids
        if bounding_box_volume is None:
            raise ValueError("Volume bounding box required")
        if bounding_box_surface is None:
            raise ValueError("Surface bounding box required")
            
        v_min, v_max = bounding_box_volume
        s_min, s_max = bounding_box_surface
        
        self.v_min = torch.tensor(v_min, device=self.preproc_device, dtype=torch.float32)
        self.v_max = torch.tensor(v_max, device=self.preproc_device, dtype=torch.float32)
        self.s_min = torch.tensor(s_min, device=self.preproc_device, dtype=torch.float32)
        self.s_max = torch.tensor(s_max, device=self.preproc_device, dtype=torch.float32)
        
        self.volume_grid = create_grid(self.v_max, self.v_min, grid_resolution)
        self.surface_grid = create_grid(self.s_max, self.s_min, grid_resolution)
        
        # Setup scaling factors
        if volume_factors is not None:
            if not isinstance(volume_factors, torch.Tensor):
                volume_factors = torch.tensor(volume_factors)
            self.volume_factors = volume_factors.to(self.preproc_device, dtype=torch.float32)
        else:
            self.volume_factors = None
            
        if surface_factors is not None:
            if not isinstance(surface_factors, torch.Tensor):
                surface_factors = torch.tensor(surface_factors)
            self.surface_factors = surface_factors.to(self.preproc_device, dtype=torch.float32)
        else:
            self.surface_factors = None
        
        self.dataset = None
        self.mesh_connectivity = None  # Store mesh connectivity for physics loss
    
    def _scale_fields(self, fields: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        """Apply scaling to fields."""
        if self.scaling_type == "mean_std_scaling":
            return standardize(fields, factors[0], factors[1])
        elif self.scaling_type == "min_max_scaling":
            return normalize(fields, factors[0], factors[1])
        return fields
    
    def process_surface(
        self,
        surface_coordinates: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_sizes: torch.Tensor,
        surface_fields: torch.Tensor | None,
        center_of_mass: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process surface data: filter, sample, kNN, normalize."""
        
        # Remove invalid sizes
        idx = surface_sizes > 0
        surface_coordinates = surface_coordinates[idx]
        surface_normals = surface_normals[idx]
        surface_sizes = surface_sizes[idx]
        if surface_fields is not None:
            surface_fields = surface_fields[idx]
        
        # Keep full coordinates for kNN
        full_surface_coordinates = surface_coordinates
        full_surface_normals = surface_normals
        full_surface_sizes = surface_sizes
        
        # Downsample
        if self.sampling:
            weights = surface_sizes if self.surface_sampling_algorithm == "area_weighted" else None
            surface_coordinates, idx_surface = shuffle_array(
                surface_coordinates, self.surface_points_sample, weights=weights
            )
            if surface_fields is not None:
                surface_fields = surface_fields[idx_surface]
            surface_normals = surface_normals[idx_surface]
            surface_sizes = surface_sizes[idx_surface]
        
        # kNN for neighbors
        if self.num_surface_neighbors > 1:
            neighbor_indices, _ = knn(
                points=full_surface_coordinates,
                queries=surface_coordinates,
                k=self.num_surface_neighbors,
            )
            surface_neighbors = full_surface_coordinates[neighbor_indices][:, 1:]
            surface_neighbors_normals = full_surface_normals[neighbor_indices][:, 1:]
            surface_neighbors_sizes = full_surface_sizes[neighbor_indices][:, 1:]
        else:
            surface_neighbors = surface_coordinates
            surface_neighbors_normals = surface_normals
            surface_neighbors_sizes = surface_sizes
        
        # Normalize coordinates
        if self.normalize_coordinates:
            surface_coordinates = normalize(surface_coordinates, self.s_max, self.s_min)
            surface_neighbors = normalize(surface_neighbors, self.s_max, self.s_min)
            center_of_mass = normalize(center_of_mass, self.s_max, self.s_min)
        
        pos_normals_com_surface = surface_coordinates - center_of_mass
        
        # Scale fields
        if self.scaling_type is not None and surface_fields is not None and self.surface_factors is not None:
            surface_fields = self._scale_fields(surface_fields, self.surface_factors)
        
        result = {
            "pos_surface_center_of_mass": pos_normals_com_surface,
            "surface_mesh_centers": surface_coordinates,
            "surface_mesh_neighbors": surface_neighbors,
            "surface_normals": surface_normals,
            "surface_neighbors_normals": surface_neighbors_normals,
            "surface_areas": surface_sizes,
            "surface_neighbors_areas": surface_neighbors_sizes,
        }
        
        if surface_fields is not None:
            result["surface_fields"] = surface_fields
        
        return result
    
    def process_volume(
        self,
        volume_coordinates: torch.Tensor,
        volume_fields: torch.Tensor | None,
        stl_vertices: torch.Tensor,
        stl_indices: torch.Tensor,
        center_of_mass: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process volume data: sample, compute SDF, normalize."""
        
        # Downsample and track cell indices for physics loss computation
        if self.sampling:
            volume_coordinates, idx_volume = shuffle_array(
                volume_coordinates, self.volume_points_sample
            )
            if volume_fields is not None:
                volume_fields = volume_fields[idx_volume]
            # Store cell indices for later mesh connectivity extraction
            cell_indices = idx_volume
        else:
            # If no sampling, all cells are used
            cell_indices = torch.arange(len(volume_coordinates), device=volume_coordinates.device)
        
        # Prepare for SDF
        if self.normalize_coordinates:
            volume_coordinates = normalize(volume_coordinates, self.v_max, self.v_min)
            grid = normalize(self.volume_grid, self.v_max, self.v_min)
            normed_vertices = normalize(stl_vertices, self.v_max, self.v_min)
            center_of_mass = normalize(center_of_mass, self.v_max, self.v_min)
        else:
            grid = self.volume_grid
            normed_vertices = stl_vertices
        
        # Scale fields
        if self.scaling_type is not None and volume_fields is not None and self.volume_factors is not None:
            volume_fields = self._scale_fields(volume_fields, self.volume_factors)
        
        # Compute SDF on grid
        sdf_grid, _ = signed_distance_field(
            normed_vertices, stl_indices, grid, use_sign_winding_number=True
        )
        
        # Compute SDF at volume points
        sdf_nodes, sdf_node_closest_point = signed_distance_field(
            normed_vertices, stl_indices, volume_coordinates, use_sign_winding_number=True
        )
        sdf_nodes = sdf_nodes.reshape((-1, 1))
        
        # Volume encodings
        pos_volume_closest = volume_coordinates - sdf_node_closest_point
        pos_volume_com = volume_coordinates - center_of_mass
        
        result = {
            "volume_mesh_centers": volume_coordinates,
            "volume_cell_indices": cell_indices,  # Track which cells were sampled
            "sdf_nodes": sdf_nodes,
            "grid": grid,
            "sdf_grid": sdf_grid,
            "pos_volume_closest": pos_volume_closest,
            "pos_volume_center_of_mass": pos_volume_com,
        }
        
        if volume_fields is not None:
            result["volume_fields"] = volume_fields
        
        return result
    
    @torch.no_grad()
    def process_data(self, data_dict: dict) -> dict:
        """Main preprocessing pipeline."""
        
        # Initialize output
        result = {
            "global_params_values": data_dict["global_params_values"],
            "global_params_reference": data_dict["global_params_reference"],
        }
        
        # Get STL data
        stl_coords = data_dict["stl_coordinates"]
        stl_faces = data_dict["stl_faces"].to(torch.int32)
        stl_centers = data_dict["stl_centers"]
        stl_areas = data_dict["stl_areas"]
        
        # Compute center of mass
        center_of_mass = calculate_center_of_mass(stl_centers, stl_areas)
        
        # Compute length scale same way as curator (for non-dimensionalization)
        # This is the maximum dimension of the STL bounding box
        if isinstance(stl_coords, torch.Tensor):
            stl_coords_np = stl_coords.cpu().numpy()
        else:
            stl_coords_np = stl_coords
        length_scale = float(np.amax(np.amax(stl_coords_np, axis=0) - np.amin(stl_coords_np, axis=0)))
        result["length_scale"] = torch.tensor(length_scale, dtype=torch.float32, device=self.output_device)
        
        # Process surface grid SDF
        if self.normalize_coordinates:
            normed_vertices = normalize(stl_coords, self.s_max, self.s_min)
            surf_grid = normalize(self.surface_grid, self.s_max, self.s_min)
        else:
            normed_vertices = stl_coords
            surf_grid = self.surface_grid
        
        sdf_surf_grid, _ = signed_distance_field(
            normed_vertices, stl_faces, surf_grid, use_sign_winding_number=True
        )
        
        result["sdf_surf_grid"] = sdf_surf_grid
        result["surf_grid"] = surf_grid
        
        # Store bounding box min/max if normalizing
        if self.normalize_coordinates:
            result["surface_min_max"] = torch.stack([self.s_min, self.s_max])
        
        # Downsample geometry
        if self.sampling:
            geom_coords, _ = shuffle_array(stl_coords, self.geom_points_sample)
        else:
            geom_coords = stl_coords
        result["geometry_coordinates"] = geom_coords
        
        # Process surface
        if self.model_type in ["surface", "combined"]:
            surface_fields = data_dict.get("surface_fields", None)
            surface_dict = self.process_surface(
                surface_coordinates=data_dict["surface_mesh_centers"],
                surface_normals=data_dict["surface_normals"],
                surface_sizes=data_dict["surface_areas"],
                surface_fields=surface_fields,
                center_of_mass=center_of_mass,
            )
            result.update(surface_dict)
        
        # Process volume
        if self.model_type in ["volume", "combined"]:
            # Store volume bounding box min/max if normalizing
            if self.normalize_coordinates:
                result["volume_min_max"] = torch.stack([self.v_min, self.v_max])
            
            volume_fields = data_dict.get("volume_fields", None)
            volume_dict = self.process_volume(
                volume_coordinates=data_dict["volume_mesh_centers"],
                volume_fields=volume_fields,
                stl_vertices=stl_coords,
                stl_indices=stl_faces,
                center_of_mass=center_of_mass,
            )
            result.update(volume_dict)
        
        return result
    
    def set_dataset(self, dataset: Dataset):
        """
        Set the underlying CAE dataset.
        
        This automatically loads mesh connectivity using the dataset's file reader.
        All samples share the same mesh topology, so connectivity is loaded once
        from the first file.
        """
        self.dataset = dataset
        
        # Automatically load mesh connectivity using the dataset's infrastructure
        self._load_mesh_connectivity_from_dataset()
    
    def _load_mesh_connectivity_from_dataset(self):
        """
        Load mesh connectivity using the CAEDataset's file reader.
        
        This leverages the existing CAEDataset infrastructure instead of
        bypassing it. Since all samples share the same mesh topology, we only
        need to load these arrays once from the first file.
        """
        import numpy as np
        
        if len(self.dataset) == 0:
            print("Warning: Dataset is empty, mesh connectivity not loaded")
            return
        
        try:
            # Define the connectivity keys we need for physics losses
            connectivity_keys = [
                'volume_points',
                'volume_fields',  # Cell-centered velocity/pressure/nut
                'volume_cell_centers',
                'volume_cell_volumes',
                'volume_cell_point_ids_flat',
                'volume_cell_point_ids_offsets',
                'volume_neighbors_flat',
                'volume_neighbors_offsets',
                'volume_face_point_ids_flat',
                'volume_face_offsets',
            ]
            
            # Use the dataset's file reader to load connectivity from first file
            filename = self.dataset._filenames[0]
            
            # Temporarily save and modify the keys to read
            original_keys = self.dataset._keys_to_read
            original_reader_keys = self.dataset.file_reader.keys_to_read
            original_is_volumetric = self.dataset.file_reader.is_volumetric
            
            self.dataset._keys_to_read = connectivity_keys
            self.dataset.file_reader.keys_to_read = connectivity_keys
            # Set is_volumetric=False to prevent slicing of connectivity arrays
            self.dataset.file_reader.is_volumetric = False
            
            # Load using the dataset's reader
            raw_data = self.dataset.file_reader.read_file(filename)
            
            # Restore original keys
            self.dataset._keys_to_read = original_keys
            self.dataset.file_reader.keys_to_read = original_reader_keys
            self.dataset.file_reader.is_volumetric = original_is_volumetric
            
            # Convert to numpy and store
            # Cell-centered FVM: velocity, pressure, nut are at cell centers
            mesh_data = {
                'points': raw_data['volume_points'].cpu().numpy().astype(np.float64),
                'velocity_data': raw_data['volume_fields'][:, :3].cpu().numpy().astype(np.float64),
                'pressure_data': raw_data['volume_fields'][:, 3].cpu().numpy().astype(np.float64),
                'nut_data': raw_data['volume_fields'][:, 4].cpu().numpy().astype(np.float64),
                'cell_centers': raw_data['volume_cell_centers'].cpu().numpy().astype(np.float64),
                'cell_volumes': raw_data['volume_cell_volumes'].cpu().numpy().astype(np.float64),
                'cell_point_ids_flat': raw_data['volume_cell_point_ids_flat'].cpu().numpy().astype(np.int32),
                'cell_point_ids_offsets': raw_data['volume_cell_point_ids_offsets'].cpu().numpy().astype(np.int32),
                'neighbors_flat': raw_data['volume_neighbors_flat'].cpu().numpy().astype(np.int32),
                'neighbors_offsets': raw_data['volume_neighbors_offsets'].cpu().numpy().astype(np.int32),
                'face_point_ids_flat': raw_data['volume_face_point_ids_flat'].cpu().numpy().astype(np.int32),
                'face_offsets': raw_data['volume_face_offsets'].cpu().numpy().astype(np.int32),
            }
            
            self.mesh_connectivity = mesh_data
            
            n_cells = len(mesh_data['cell_volumes'])
            n_points = len(mesh_data['points'])
            print(f"✓ Mesh connectivity auto-loaded via CAEDataset: {n_cells:,} cells, {n_points:,} points")
            
        except Exception as e:
            print(f"Warning: Could not load mesh connectivity: {e}")
            self.mesh_connectivity = None
    
    def get_batched_mesh_data(self, batch_dict: dict, cell_indices: torch.Tensor = None, 
                              field_data: dict = None):
        """
        Extract batched mesh connectivity data for a subset of volume cells.
        
        """
        if self.mesh_connectivity is None:
            raise ValueError(
                "Mesh connectivity not loaded. This should have been auto-loaded when "
                "set_dataset() was called. Check that the dataset's data_dir contains zarr files."
            )
        
        import numpy as np
        
        if cell_indices is None:
            if 'volume_cell_indices' not in batch_dict:
                raise ValueError("cell_indices not provided and 'volume_cell_indices' not in batch_dict")
            cell_indices = batch_dict['volume_cell_indices']
        
        # Convert to numpy if torch tensor
        if isinstance(cell_indices, torch.Tensor):
            cell_indices = cell_indices.cpu().numpy()
        
        # Remove batch dimension if present
        if cell_indices.ndim > 1:
            cell_indices = cell_indices.squeeze(0)
        
        # Collect all cells needed (sampled + neighbors) for cell-centered FVM
        # Field data for sampled cells and their neighbors for flux computation
        cells_needed_set = set(cell_indices)
        for cell_idx in cell_indices:
            nb_start = self.mesh_connectivity['neighbors_offsets'][cell_idx]
            nb_end = self.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
            neighbor_cells = self.mesh_connectivity['neighbors_flat'][nb_start:nb_end]
            for nb_cell in neighbor_cells:
                if nb_cell >= 0:  # Valid neighbor (not boundary)
                    cells_needed_set.add(nb_cell)
        
        global_cell_ids = np.array(sorted(cells_needed_set), dtype=np.int32)
        global_to_local_cell = {global_id: local_id for local_id, global_id in enumerate(global_cell_ids)}
        
        # Collect all unique point IDs needed for all cells (sampled + neighbors)
        point_ids_set = set()
        
        # Add points from all cells (sampled + neighbors)
        for cell_idx in global_cell_ids:
            start = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx]
            end = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx + 1]
            point_ids_set.update(self.mesh_connectivity['cell_point_ids_flat'][start:end])
        
        # Add points from all faces (for all cells)
        for cell_idx in global_cell_ids:
            nb_start = self.mesh_connectivity['neighbors_offsets'][cell_idx]
            nb_end = self.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
            n_neighbors = nb_end - nb_start
            
            for nb_idx in range(n_neighbors):
                face_start = self.mesh_connectivity['face_offsets'][nb_start + nb_idx]
                face_end = self.mesh_connectivity['face_offsets'][nb_start + nb_idx + 1]
                face_points = self.mesh_connectivity['face_point_ids_flat'][face_start:face_end]
                point_ids_set.update(face_points)
        
        # Create global-to-local point ID mapping
        global_point_ids = np.array(sorted(point_ids_set), dtype=np.int32)
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_point_ids)}
        
        # Extract local point coordinates (for face geometry)
        local_points = self.mesh_connectivity['points'][global_point_ids]
        
        # Extract cell-centered field data
        if field_data is not None:
            # Assume field_data is for all cells, extract only what we need
            cell_velocity = field_data['velocity'][global_cell_ids]
            cell_pressure = field_data['pressure'][global_cell_ids]
            cell_nut = field_data['nut'][global_cell_ids]
        else:
            # Use ground truth from mesh connectivity (cell-centered)
            cell_velocity = self.mesh_connectivity['velocity_data'][global_cell_ids]
            cell_pressure = self.mesh_connectivity['pressure_data'][global_cell_ids]
            cell_nut = self.mesh_connectivity['nut_data'][global_cell_ids]
        
        batched_data = {
            'points': local_points,  # For face geometry only
            'velocity_data': cell_velocity,  # Cell-centered [n_local_cells, 3]
            'pressure_data': cell_pressure,  # Cell-centered [n_local_cells]
            'nut_data': cell_nut,  # Cell-centered [n_local_cells]
        }
        
        # Remap cell indices to local space
        local_cell_indices = np.array([global_to_local_cell[gid] for gid in cell_indices], dtype=np.int32)
        
        # Extract cell-specific arrays for local cells
        batched_data['cell_volumes'] = self.mesh_connectivity['cell_volumes'][global_cell_ids]
        batched_data['cell_centers'] = self.mesh_connectivity['cell_centers'][global_cell_ids]
        
        # Extract and remap cell point IDs for all cells in batch
        cell_point_ids_list = []
        for cell_idx in global_cell_ids:
            start = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx]
            end = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx + 1]
            global_ids = self.mesh_connectivity['cell_point_ids_flat'][start:end]
            local_ids = np.array([global_to_local[gid] for gid in global_ids], dtype=np.int32)
            cell_point_ids_list.append(local_ids)
        
        batched_data['cell_point_ids_flat'] = np.concatenate(cell_point_ids_list) if cell_point_ids_list else np.array([], dtype=np.int32)
        batched_data['cell_point_ids_offsets'] = np.concatenate([
            [0], np.cumsum([len(x) for x in cell_point_ids_list])
        ]).astype(np.int32)
        
        # Extract neighbors and faces with remapped cell and point IDs
        neighbors_list = []
        face_point_ids_list = []
        face_offsets_list = [[0]]
        
        for cell_idx in global_cell_ids:
            nb_start = self.mesh_connectivity['neighbors_offsets'][cell_idx]
            nb_end = self.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
            n_neighbors = nb_end - nb_start
            
            # Remap neighbor cell IDs to local space
            global_neighbors = self.mesh_connectivity['neighbors_flat'][nb_start:nb_end]
            local_neighbors = np.array([
                global_to_local_cell[nb] if nb >= 0 and nb in global_to_local_cell else -1
                for nb in global_neighbors
            ], dtype=np.int32)
            neighbors_list.append(local_neighbors)
            
            # Extract and remap face point IDs
            for nb_idx in range(n_neighbors):
                face_start = self.mesh_connectivity['face_offsets'][nb_start + nb_idx]
                face_end = self.mesh_connectivity['face_offsets'][nb_start + nb_idx + 1]
                global_face_points = self.mesh_connectivity['face_point_ids_flat'][face_start:face_end]
                local_face_points = np.array([global_to_local[gid] for gid in global_face_points], dtype=np.int32)
                face_point_ids_list.append(local_face_points)
                face_offsets_list.append([len(local_face_points)])
        
        batched_data['neighbors_flat'] = np.concatenate(neighbors_list) if neighbors_list else np.array([], dtype=np.int32)
        batched_data['neighbors_offsets'] = np.concatenate([
            [0], np.cumsum([len(x) for x in neighbors_list])
        ]).astype(np.int32)
        
        batched_data['face_point_ids_flat'] = np.concatenate(face_point_ids_list) if face_point_ids_list else np.array([], dtype=np.int32)
        batched_data['face_offsets'] = np.concatenate([[0], np.cumsum([x[0] for x in face_offsets_list[1:]])]).astype(np.int32)
        
        # Return batched data with local cell indices for computing
        batched_data['local_cell_indices'] = local_cell_indices
        
        return batched_data
    
    def get_neighbor_cell_centers(self, batch_dict: dict, cell_indices: torch.Tensor = None, 
                                   max_neighbors_cap: int = 12):
        """
        Get neighbor cell center coordinates for computing physics loss.
        
        For each sampled cell, returns the coordinates of its neighbor cell centers.
        This is used to evaluate the neural network at neighbor positions for FVM.
        
        Args:
            batch_dict: Batch dictionary from __getitem__
            cell_indices: Optional tensor of cell indices. If None, uses batch_dict['volume_cell_indices']
            max_neighbors_cap: Maximum number of neighbors to include per cell (default: 12).
                              This caps memory usage for cells with many neighbors.
        
        Returns:
            neighbor_centers: [n_cells, max_neighbors_cap, 3] array of neighbor cell centers
                             Invalid neighbors (boundaries) have coordinates [0, 0, 0]
            neighbor_mask: [n_cells, max_neighbors_cap] boolean mask (True = valid neighbor)
        """
        if self.mesh_connectivity is None:
            raise ValueError("Mesh connectivity not loaded")
        
        import numpy as np
        
        if cell_indices is None:
            if 'volume_cell_indices' not in batch_dict:
                raise ValueError("cell_indices not provided and 'volume_cell_indices' not in batch_dict")
            cell_indices = batch_dict['volume_cell_indices']
        
        # Convert to numpy if torch tensor
        if isinstance(cell_indices, torch.Tensor):
            cell_indices = cell_indices.cpu().numpy()
        
        # Remove batch dimension if present
        if cell_indices.ndim > 1:
            cell_indices = cell_indices.squeeze(0)
        
        # Extract neighbors for each cell (capped at max_neighbors_cap)
        neighbor_lists = []
        
        for cell_idx in cell_indices:
            nb_start = self.mesh_connectivity['neighbors_offsets'][cell_idx]
            nb_end = self.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
            neighbor_ids = self.mesh_connectivity['neighbors_flat'][nb_start:nb_end]
            
            # Cap the number of neighbors to reduce memory usage
            if len(neighbor_ids) > max_neighbors_cap:
                neighbor_ids = neighbor_ids[:max_neighbors_cap]  # Take first N neighbors
            
            neighbor_lists.append(neighbor_ids)
        
        # Create padded arrays with fixed max_neighbors_cap
        n_cells = len(cell_indices)
        neighbor_centers = np.zeros((n_cells, max_neighbors_cap, 3), dtype=np.float32)
        neighbor_mask = np.zeros((n_cells, max_neighbors_cap), dtype=bool)
        
        for i, neighbor_ids in enumerate(neighbor_lists):
            for j, nb_id in enumerate(neighbor_ids):
                if nb_id >= 0:  # Valid neighbor (not boundary)
                    neighbor_centers[i, j] = self.mesh_connectivity['cell_centers'][nb_id]
                    neighbor_mask[i, j] = True
        
        return neighbor_centers, neighbor_mask
    
    def get_points_for_batch(self, batch_dict: dict, cell_indices: torch.Tensor = None):
        """
        Get the point IDs and coordinates needed for computing residuals on a batch.
        
        This is useful for running the neural network on only the required points
        instead of all points in the mesh.
        
        Args:
            batch_dict: Batch dictionary from __getitem__
            cell_indices: Optional tensor of cell indices. If None, uses batch_dict['volume_cell_indices']
        
        Returns:
            tuple: (global_point_ids, point_coordinates)
                - global_point_ids: numpy array of global point IDs [n_local_points]
                - point_coordinates: numpy array of point coordinates [n_local_points, 3]
        """
        if self.mesh_connectivity is None:
            raise ValueError("Mesh connectivity not loaded")
        
        import numpy as np
        
        if cell_indices is None:
            if 'volume_cell_indices' not in batch_dict:
                raise ValueError("cell_indices not provided and 'volume_cell_indices' not in batch_dict")
            cell_indices = batch_dict['volume_cell_indices']
        
        if isinstance(cell_indices, torch.Tensor):
            cell_indices = cell_indices.cpu().numpy()
        
        if cell_indices.ndim > 1:
            cell_indices = cell_indices.squeeze(0)
        
        # Collect all unique point IDs needed
        point_ids_set = set()
        
        for cell_idx in cell_indices:
            start = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx]
            end = self.mesh_connectivity['cell_point_ids_offsets'][cell_idx + 1]
            point_ids_set.update(self.mesh_connectivity['cell_point_ids_flat'][start:end])
            
            # Add neighbor cell points
            nb_start = self.mesh_connectivity['neighbors_offsets'][cell_idx]
            nb_end = self.mesh_connectivity['neighbors_offsets'][cell_idx + 1]
            neighbor_cells = self.mesh_connectivity['neighbors_flat'][nb_start:nb_end]
            
            for nb_cell in neighbor_cells:
                if nb_cell >= 0:
                    start = self.mesh_connectivity['cell_point_ids_offsets'][nb_cell]
                    end = self.mesh_connectivity['cell_point_ids_offsets'][nb_cell + 1]
                    point_ids_set.update(self.mesh_connectivity['cell_point_ids_flat'][start:end])
            
            # Add face points
            n_neighbors = nb_end - nb_start
            for nb_idx in range(n_neighbors):
                face_start = self.mesh_connectivity['face_offsets'][nb_start + nb_idx]
                face_end = self.mesh_connectivity['face_offsets'][nb_start + nb_idx + 1]
                face_points = self.mesh_connectivity['face_point_ids_flat'][face_start:face_end]
                point_ids_set.update(face_points)
        
        global_point_ids = np.array(sorted(point_ids_set), dtype=np.int32)
        point_coordinates = self.mesh_connectivity['points'][global_point_ids]
        
        return global_point_ids, point_coordinates
    
    def __len__(self):
        return len(self.dataset) if self.dataset else 0
    
    def __getitem__(self, idx):
        """Fetch and process a single item."""
        if self.dataset is None:
            raise ValueError("Dataset not set. Call set_dataset() first.")
        
        data_dict = self.dataset[idx]
        processed = self.process_data(data_dict)
        
        processed = {k: v.to(self.output_device) for k, v in processed.items()}
        
        # Add batch dimension
        processed = {k: v.unsqueeze(0) for k, v in processed.items()}
        
        return processed
    
    def unscale_model_outputs(
        self,
        volume_fields: torch.Tensor | None = None,
        surface_fields: torch.Tensor | None = None,
    ):
        """Unscale model outputs back to original scale."""
        if volume_fields is not None and self.volume_factors is not None:
            if self.scaling_type == "mean_std_scaling":
                volume_fields = unstandardize(
                    volume_fields, self.volume_factors[0], self.volume_factors[1]
                )
            elif self.scaling_type == "min_max_scaling":
                volume_fields = unnormalize(
                    volume_fields, self.volume_factors[0], self.volume_factors[1]
                )
        
        if surface_fields is not None and self.surface_factors is not None:
            if self.scaling_type == "mean_std_scaling":
                surface_fields = unstandardize(
                    surface_fields, self.surface_factors[0], self.surface_factors[1]
                )
            elif self.scaling_type == "min_max_scaling":
                surface_fields = unnormalize(
                    surface_fields, self.surface_factors[0], self.surface_factors[1]
                )
        
        return volume_fields, surface_fields

