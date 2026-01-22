# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Compute FVM residuals on VTU meshes using face-based Warp kernels with autodiff support.

Usage:
    python compute_residuals_vtu.py path/to/mesh.vtu --nu 1.5e-5 --save output.vtu
"""

import numpy as np
import time
import warp as wp
import torch
from numba import njit, prange
from tqdm import tqdm

wp.init()


# ============================================================================
# Numba-accelerated face geometry computation
# ============================================================================

@njit(fastmath=True)
def compute_face_geometry_numba(point_ids, points, cell_center):
    """Compute face area, normal, and center using numba."""
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)
    
    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec[0] += edge1[1] * edge2[2] - edge1[2] * edge2[1]
        area_vec[1] += edge1[2] * edge2[0] - edge1[0] * edge2[2]
        area_vec[2] += edge1[0] * edge2[1] - edge1[1] * edge2[0]
    
    area_mag = np.sqrt(area_vec[0]**2 + area_vec[1]**2 + area_vec[2]**2)
    if area_mag < 1e-30:
        return 0.0, np.zeros(3), np.zeros(3)
    
    area = 0.5 * area_mag
    normal = area_vec / area_mag
    
    # Face center
    face_center = np.zeros(3)
    for i in range(n_points):
        face_center += points[point_ids[i]]
    face_center /= n_points
    
    # Orient normal outward from owner cell
    face_to_cell = cell_center - face_center
    if normal[0]*face_to_cell[0] + normal[1]*face_to_cell[1] + normal[2]*face_to_cell[2] > 0:
        normal = -normal
    
    return area, normal, face_center


# ============================================================================
# VTU Mesh Loading (adapted from compure_physics_loss_standalone.py)
# ============================================================================

def make_vtk_progress_callback(pbar):
    """Create a VTK progress callback that updates a tqdm progress bar."""
    def callback(caller, event):
        if event == "ProgressEvent":
            progress = caller.GetProgress()
            pbar.n = int(progress * 100)
            pbar.refresh()
    return callback


def load_vtu_mesh(filename, velocity_field="UMean", pressure_field="pMean", nut_field="nutMean"):
    """Load VTU mesh and extract cell-centered data."""
    import vtk
    from vtk.util import numpy_support
    
    print(f"Loading mesh: {filename}")
    t0 = time.time()
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    ugrid = reader.GetOutput()
    
    print(f"  {ugrid.GetNumberOfCells():,} cells, {ugrid.GetNumberOfPoints():,} points")
    
    # Convert point data to cell data if needed
    if velocity_field not in [ugrid.GetCellData().GetArrayName(i) for i in range(ugrid.GetCellData().GetNumberOfArrays())]:
        with tqdm(total=100, desc="  Point to Cell", unit="%") as pbar:
            p2c = vtk.vtkPointDataToCellData()
            p2c.SetInputData(ugrid)
            p2c.PassPointDataOn()
            p2c.AddObserver("ProgressEvent", make_vtk_progress_callback(pbar))
            p2c.Update()
            pbar.n = 100
            pbar.refresh()
        ugrid = p2c.GetOutput()

    # Compute cell volumes
    if not ugrid.GetCellData().HasArray("Volume"):
        with tqdm(total=100, desc="  Cell Volumes", unit="%") as pbar:
            cell_size_filter = vtk.vtkCellSizeFilter()
            cell_size_filter.SetInputData(ugrid)
            cell_size_filter.SetComputeLength(False)
            cell_size_filter.SetComputeArea(False)
            cell_size_filter.SetComputeVolume(True)
            cell_size_filter.SetComputeVertexCount(False)
            cell_size_filter.AddObserver("ProgressEvent", make_vtk_progress_callback(pbar))
            cell_size_filter.Update()
            pbar.n = 100
            pbar.refresh()
        ugrid = cell_size_filter.GetOutput()
    
    # Extract arrays
    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float32)
    velocity_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(velocity_field)).astype(np.float32)
    pressure_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(pressure_field)).astype(np.float32)
    nut_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(nut_field)).astype(np.float32)
    cell_volumes = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray("Volume")).astype(np.float32)

    # Compute cell centers
    with tqdm(total=100, desc="  Cell Centers", unit="%") as pbar:
        cell_centers_filter = vtk.vtkCellCenters()
        cell_centers_filter.SetInputData(ugrid)
        cell_centers_filter.AddObserver("ProgressEvent", make_vtk_progress_callback(pbar))
        cell_centers_filter.Update()
        pbar.n = 100
        pbar.refresh()
    cell_centers = numpy_support.vtk_to_numpy(cell_centers_filter.GetOutput().GetPoints().GetData()).astype(np.float32)

    print(f"  Loaded in {time.time() - t0:.2f}s")
    
    return {
        'ugrid': ugrid,
        'points': points,
        'velocity_data': velocity_data,
        'pressure_data': pressure_data,
        'nut_data': nut_data,
        'cell_volumes': cell_volumes,
        'cell_centers': cell_centers,
        'n_cells': ugrid.GetNumberOfCells(),
    }


def build_face_connectivity(ugrid):
    """Build face connectivity from VTK unstructured grid."""
    from vtk.util import numpy_support
    import vtk
    
    print("Building face connectivity...")
    t0 = time.time()
    
    n_cells = ugrid.GetNumberOfCells()
    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)
    
    # Compute cell centers
    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ugrid)
    cell_centers_filter.Update()
    cell_centers = numpy_support.vtk_to_numpy(cell_centers_filter.GetOutput().GetPoints().GetData()).astype(np.float64)
    
    # Step 1: Extract faces and build neighbor map
    print("  Step 1/2: Extracting faces from cells...")
    face_to_cells = {}
    
    for cell_idx in tqdm(range(n_cells), desc="  Cells", unit="cells"):
        cell = ugrid.GetCell(cell_idx)
        n_faces = cell.GetNumberOfFaces()
        
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids = face.GetPointIds()
            n_pts = face_point_ids.GetNumberOfIds()
            face_pts = tuple(sorted([face_point_ids.GetId(i) for i in range(n_pts)]))
            
            if face_pts not in face_to_cells:
                face_to_cells[face_pts] = []
            face_to_cells[face_pts].append((cell_idx, np.array([face_point_ids.GetId(i) for i in range(n_pts)], dtype=np.int64)))
    
    # Step 2: Build face geometry arrays using numba
    print(f"  Step 2/2: Computing geometry for {len(face_to_cells):,} unique faces...")
    
    face_owner = []
    face_neighbor = []
    face_area = []
    face_normal = []
    face_center_list = []
    
    for face_pts_sorted, cell_list in tqdm(face_to_cells.items(), desc="  Faces", unit="faces"):
        if len(cell_list) == 2:
            # Interior face
            (cell1, pts1), (cell2, pts2) = cell_list
            owner, neighbor = (cell1, cell2) if cell1 < cell2 else (cell2, cell1)
            face_pts = pts1
        elif len(cell_list) == 1:
            # Boundary face
            owner = cell_list[0][0]
            neighbor = -1
            face_pts = cell_list[0][1]
        else:
            continue
        
        # Compute face geometry using numba
        area, normal, fc = compute_face_geometry_numba(face_pts, points, cell_centers[owner])
        
        if area < 1e-30:
            continue
        
        face_owner.append(owner)
        face_neighbor.append(neighbor)
        face_area.append(area)
        face_normal.append(normal)
        face_center_list.append(fc)
    
    print(f"  {len(face_owner):,} faces built in {time.time() - t0:.2f}s")
    
    return {
        'n_faces': len(face_owner),
        'face_owner': np.array(face_owner, dtype=np.int32),
        'face_neighbor': np.array(face_neighbor, dtype=np.int32),
        'face_area': np.array(face_area, dtype=np.float32),
        'face_normal': np.array(face_normal, dtype=np.float32),
        'face_centers': np.array(face_center_list, dtype=np.float32),
    }



# ============================================================================
# Batched kernels (for loss computation with subset of cells)
# ============================================================================

@wp.kernel
def compute_face_flux_kernel_batched(
    n_faces: int,
    velocity: wp.array(dtype=wp.vec3),
    face_owner: wp.array(dtype=int),
    face_neighbor: wp.array(dtype=int),
    face_area: wp.array(dtype=float),
    face_normal: wp.array(dtype=wp.vec3),
    cell_centers: wp.array(dtype=wp.vec3),
    face_centers: wp.array(dtype=wp.vec3),
    n_cells: int,
    cell_to_batch_idx: wp.array(dtype=int),
    continuity: wp.array(dtype=float)
):
    """Compute continuity flux, accumulating only to batch cells."""
    face_id = wp.tid()
    if face_id >= n_faces:
        return
    
    owner = face_owner[face_id]
    neighbor = face_neighbor[face_id]
    area = face_area[face_id]
    normal = face_normal[face_id]
    fc = face_centers[face_id]
    
    v_owner = velocity[owner]
    oc = cell_centers[owner]
    
    if neighbor >= 0 and neighbor < n_cells:
        v_neighbor = velocity[neighbor]
        nc = cell_centers[neighbor]
        d1 = wp.length(fc - oc)
        d2 = wp.length(fc - nc)
        total = d1 + d2
        if total > 1e-30:
            w1, w2 = d2 / total, d1 / total
            v_face = w1 * v_owner + w2 * v_neighbor
        else:
            v_face = 0.5 * (v_owner + v_neighbor)
    else:
        v_face = v_owner
    
    flux = area * wp.dot(normal, v_face)
    
    # Only accumulate to batch cells
    owner_batch_idx = cell_to_batch_idx[owner]
    if owner_batch_idx >= 0:
        wp.atomic_add(continuity, owner_batch_idx, flux)
    
    if neighbor >= 0 and neighbor < n_cells:
        neighbor_batch_idx = cell_to_batch_idx[neighbor]
        if neighbor_batch_idx >= 0:
            wp.atomic_add(continuity, neighbor_batch_idx, -flux)


@wp.kernel
def compute_velocity_gradients_kernel_batched(
    n_faces: int,
    velocity: wp.array(dtype=wp.vec3),
    face_owner: wp.array(dtype=int),
    face_neighbor: wp.array(dtype=int),
    face_area: wp.array(dtype=float),
    face_normal: wp.array(dtype=wp.vec3),
    face_centers: wp.array(dtype=wp.vec3),
    cell_centers: wp.array(dtype=wp.vec3),
    cell_volumes: wp.array(dtype=float),
    n_cells: int,
    cell_to_batch_idx: wp.array(dtype=int),
    grad_u: wp.array(dtype=wp.vec3),
    grad_v: wp.array(dtype=wp.vec3),
    grad_w: wp.array(dtype=wp.vec3),
):
    """Compute velocity gradients, accumulating only to batch cells."""
    face_id = wp.tid()
    if face_id >= n_faces:
        return
    
    owner = face_owner[face_id]
    neighbor = face_neighbor[face_id]
    area = face_area[face_id]
    normal = face_normal[face_id]
    fc = face_centers[face_id]
    
    v_owner = velocity[owner]
    oc = cell_centers[owner]
    
    if neighbor >= 0 and neighbor < n_cells:
        v_neighbor = velocity[neighbor]
        nc = cell_centers[neighbor]
        d1 = wp.length(fc - oc)
        d2 = wp.length(fc - nc)
        total = d1 + d2
        if total > 1e-30:
            w1, w2 = d2 / total, d1 / total
            v_face = w1 * v_owner + w2 * v_neighbor
        else:
            v_face = 0.5 * (v_owner + v_neighbor)
    else:
        v_face = v_owner
    
    contrib_u = area * v_face[0] * normal
    contrib_v = area * v_face[1] * normal
    contrib_w = area * v_face[2] * normal
    
    # Only accumulate to batch cells
    owner_batch_idx = cell_to_batch_idx[owner]
    if owner_batch_idx >= 0:
        vol_owner = cell_volumes[owner]
        wp.atomic_add(grad_u, owner_batch_idx, contrib_u / vol_owner)
        wp.atomic_add(grad_v, owner_batch_idx, contrib_v / vol_owner)
        wp.atomic_add(grad_w, owner_batch_idx, contrib_w / vol_owner)
    
    if neighbor >= 0 and neighbor < n_cells:
        neighbor_batch_idx = cell_to_batch_idx[neighbor]
        if neighbor_batch_idx >= 0:
            vol_neighbor = cell_volumes[neighbor]
            wp.atomic_add(grad_u, neighbor_batch_idx, -contrib_u / vol_neighbor)
            wp.atomic_add(grad_v, neighbor_batch_idx, -contrib_v / vol_neighbor)
            wp.atomic_add(grad_w, neighbor_batch_idx, -contrib_w / vol_neighbor)


@wp.kernel
def compute_momentum_flux_kernel_batched(
    n_faces: int,
    velocity: wp.array(dtype=wp.vec3),
    pressure: wp.array(dtype=float),
    nut: wp.array(dtype=float),
    grad_u: wp.array(dtype=wp.vec3),
    grad_v: wp.array(dtype=wp.vec3),
    grad_w: wp.array(dtype=wp.vec3),
    nu: float,
    include_convection: int,  # 1 = Navier-Stokes (with convection), 0 = Stokes (no convection)
    use_laplacian_diffusion: int,  # 1 = Laplacian form (for Stokes), 0 = stress tensor form (for NS)
    face_owner: wp.array(dtype=int),
    face_neighbor: wp.array(dtype=int),
    face_area: wp.array(dtype=float),
    face_normal: wp.array(dtype=wp.vec3),
    face_centers: wp.array(dtype=wp.vec3),
    cell_centers: wp.array(dtype=wp.vec3),
    n_cells: int,
    cell_to_batch_idx: wp.array(dtype=int),
    batch_grad_u: wp.array(dtype=wp.vec3),
    batch_grad_v: wp.array(dtype=wp.vec3),
    batch_grad_w: wp.array(dtype=wp.vec3),
    momentum_x: wp.array(dtype=float),
    momentum_y: wp.array(dtype=float),
    momentum_z: wp.array(dtype=float),
):
    """Compute momentum fluxes, accumulating only to batch cells.
    
    Args:
        include_convection: 1 for Navier-Stokes (includes (u·∇)u), 0 for Stokes (no convection)
        use_laplacian_diffusion: 1 for simple Laplacian ν∇²u (Stokes), 0 for stress tensor ∇·τ (NS)
            - Laplacian: visc = ν * (∇φ · n) - simpler, exact match with PINN Stokes
            - Stress tensor: visc = (τ · n) where τ = ν(∇u + (∇u)^T) - general NS form
    """
    face_id = wp.tid()
    if face_id >= n_faces:
        return
    
    owner = face_owner[face_id]
    neighbor = face_neighbor[face_id]
    area = face_area[face_id]
    normal = face_normal[face_id]
    fc = face_centers[face_id]
    
    v_owner = velocity[owner]
    p_owner = pressure[owner]
    nut_owner = nut[owner]
    oc = cell_centers[owner]
    
    if neighbor >= 0 and neighbor < n_cells:
        v_neighbor = velocity[neighbor]
        p_neighbor = pressure[neighbor]
        nc = cell_centers[neighbor]
        d1 = wp.length(fc - oc)
        d2 = wp.length(fc - nc)
        total = d1 + d2
        if total > 1e-30:
            w1, w2 = d2 / total, d1 / total
            v_face = w1 * v_owner + w2 * v_neighbor
            p_face = w1 * p_owner + w2 * p_neighbor
        else:
            v_face = 0.5 * (v_owner + v_neighbor)
            p_face = 0.5 * (p_owner + p_neighbor)
    else:
        v_face = v_owner
        p_face = p_owner
    
    # Check if owner is in batch - use its gradient for viscous term
    owner_batch_idx = cell_to_batch_idx[owner]
    
    # Convective term: (u·n)u - only for Navier-Stokes, zero for Stokes
    conv_x = 0.0
    conv_y = 0.0
    conv_z = 0.0
    if include_convection == 1:
        v_dot_n = wp.dot(v_face, normal)
        conv_x = v_dot_n * v_face[0]
        conv_y = v_dot_n * v_face[1]
        conv_z = v_dot_n * v_face[2]
    
    pres_x = p_face * normal[0]
    pres_y = p_face * normal[1]
    pres_z = p_face * normal[2]
    
    # Compute owner's contribution
    if owner_batch_idx >= 0:
        nu_eff = nu + nut_owner
        gu_owner = batch_grad_u[owner_batch_idx]
        gv_owner = batch_grad_v[owner_batch_idx]
        gw_owner = batch_grad_w[owner_batch_idx]
        
        # Compute viscous flux based on diffusion model
        visc_x = 0.0
        visc_y = 0.0
        visc_z = 0.0
        
        if use_laplacian_diffusion == 1:
            # Laplacian form: ν∇²u -> flux = ν * (∇φ · n)
            # This matches the PINN Stokes formulation exactly
            visc_x = nu_eff * wp.dot(gu_owner, normal)
            visc_y = nu_eff * wp.dot(gv_owner, normal)
            visc_z = nu_eff * wp.dot(gw_owner, normal)
        else:
            # Stress tensor form: ∇·τ where τ = ν(∇u + (∇u)^T)
            # More general, used for Navier-Stokes
            tau_xx = nu_eff * 2.0 * gu_owner[0]
            tau_yy = nu_eff * 2.0 * gv_owner[1]
            tau_zz = nu_eff * 2.0 * gw_owner[2]
            tau_xy = nu_eff * (gu_owner[1] + gv_owner[0])
            tau_xz = nu_eff * (gu_owner[2] + gw_owner[0])
            tau_yz = nu_eff * (gv_owner[2] + gw_owner[1])
            
            visc_x = tau_xx * normal[0] + tau_xy * normal[1] + tau_xz * normal[2]
            visc_y = tau_xy * normal[0] + tau_yy * normal[1] + tau_yz * normal[2]
            visc_z = tau_xz * normal[0] + tau_yz * normal[1] + tau_zz * normal[2]
        
        flux_x = area * (conv_x + pres_x - visc_x)
        flux_y = area * (conv_y + pres_y - visc_y)
        flux_z = area * (conv_z + pres_z - visc_z)
        
        wp.atomic_add(momentum_x, owner_batch_idx, flux_x)
        wp.atomic_add(momentum_y, owner_batch_idx, flux_y)
        wp.atomic_add(momentum_z, owner_batch_idx, flux_z)
    
    # Compute neighbor's contribution
    if neighbor >= 0 and neighbor < n_cells:
        neighbor_batch_idx = cell_to_batch_idx[neighbor]
        if neighbor_batch_idx >= 0:
            nut_neighbor = nut[neighbor]
            nu_eff_n = nu + nut_neighbor
            gu_neighbor = batch_grad_u[neighbor_batch_idx]
            gv_neighbor = batch_grad_v[neighbor_batch_idx]
            gw_neighbor = batch_grad_w[neighbor_batch_idx]
            
            # Compute viscous flux based on diffusion model
            visc_x_n = 0.0
            visc_y_n = 0.0
            visc_z_n = 0.0
            
            if use_laplacian_diffusion == 1:
                # Laplacian form: ν∇²u -> flux = ν * (∇φ · n)
                visc_x_n = nu_eff_n * wp.dot(gu_neighbor, normal)
                visc_y_n = nu_eff_n * wp.dot(gv_neighbor, normal)
                visc_z_n = nu_eff_n * wp.dot(gw_neighbor, normal)
            else:
                # Stress tensor form
                tau_xx_n = nu_eff_n * 2.0 * gu_neighbor[0]
                tau_yy_n = nu_eff_n * 2.0 * gv_neighbor[1]
                tau_zz_n = nu_eff_n * 2.0 * gw_neighbor[2]
                tau_xy_n = nu_eff_n * (gu_neighbor[1] + gv_neighbor[0])
                tau_xz_n = nu_eff_n * (gu_neighbor[2] + gw_neighbor[0])
                tau_yz_n = nu_eff_n * (gv_neighbor[2] + gw_neighbor[1])
                
                visc_x_n = tau_xx_n * normal[0] + tau_xy_n * normal[1] + tau_xz_n * normal[2]
                visc_y_n = tau_xy_n * normal[0] + tau_yy_n * normal[1] + tau_yz_n * normal[2]
                visc_z_n = tau_xz_n * normal[0] + tau_yz_n * normal[1] + tau_zz_n * normal[2]
            
            # Neighbor gets negative flux (opposite direction)
            flux_x_n = area * (conv_x + pres_x - visc_x_n)
            flux_y_n = area * (conv_y + pres_y - visc_y_n)
            flux_z_n = area * (conv_z + pres_z - visc_z_n)
            
            wp.atomic_add(momentum_x, neighbor_batch_idx, -flux_x_n)
            wp.atomic_add(momentum_y, neighbor_batch_idx, -flux_y_n)
            wp.atomic_add(momentum_z, neighbor_batch_idx, -flux_z_n)


def build_batch_face_data(face_data, batch_cell_indices, n_cells):
    """
    Identify faces that touch at least one batch cell.
    
    Args:
        face_data: Full face connectivity from build_face_connectivity()
        batch_cell_indices: Array of cell indices in the batch
        n_cells: Total number of cells in mesh
    
    Returns:
        dict with:
            - batch_face_indices: Indices of faces touching batch cells
            - cell_to_batch_idx: Mapping from global cell ID to batch index (-1 if not in batch)
    """
    batch_set = set(batch_cell_indices)
    
    # Create cell-to-batch-index mapping
    cell_to_batch_idx = np.full(n_cells, -1, dtype=np.int32)
    for batch_idx, cell_id in enumerate(batch_cell_indices):
        cell_to_batch_idx[cell_id] = batch_idx
    
    # Find faces touching batch cells
    face_owner = face_data['face_owner']
    face_neighbor = face_data['face_neighbor']
    
    batch_face_indices = []
    for face_id in range(len(face_owner)):
        owner = face_owner[face_id]
        neighbor = face_neighbor[face_id]
        if owner in batch_set or (neighbor >= 0 and neighbor in batch_set):
            batch_face_indices.append(face_id)
    
    return {
        'batch_face_indices': np.array(batch_face_indices, dtype=np.int32),
        'cell_to_batch_idx': cell_to_batch_idx,
    }


def compute_residuals_warp_batch(mesh_data, face_data, batch_cell_indices, nu, device='cuda:0', stokes_flow=False):
    """
    Compute residuals for a batch (subset) of cells using face-based Warp kernels.
    
    Args:
        mesh_data: Full mesh data from load_vtu_mesh()
        face_data: Full face connectivity from build_face_connectivity()
        batch_cell_indices: Array of cell indices to compute [n_batch]
        nu: Kinematic viscosity
        device: Warp device
        stokes_flow: If True, use Stokes equations (no convection). If False, use Navier-Stokes.
    
    Returns:
        dict with continuity, momentum_x, momentum_y, momentum_z arrays of shape [n_batch]
    """
    batch_cell_indices = np.asarray(batch_cell_indices, dtype=np.int32)
    n_batch = len(batch_cell_indices)
    n_cells = mesh_data['n_cells']
    
    # Build batch face data
    batch_info = build_batch_face_data(face_data, batch_cell_indices, n_cells)
    batch_face_indices = batch_info['batch_face_indices']
    cell_to_batch_idx = batch_info['cell_to_batch_idx']
    n_batch_faces = len(batch_face_indices)
    
    # Extract face data for batch faces
    batch_face_owner = face_data['face_owner'][batch_face_indices]
    batch_face_neighbor = face_data['face_neighbor'][batch_face_indices]
    batch_face_area = face_data['face_area'][batch_face_indices]
    batch_face_normal = face_data['face_normal'][batch_face_indices]
    batch_face_centers = face_data['face_centers'][batch_face_indices]
    
    # Create Warp arrays (full mesh data needed for neighbor lookups)
    velocity_wp = wp.array(mesh_data['velocity_data'], dtype=wp.vec3, device=device)
    pressure_wp = wp.array(mesh_data['pressure_data'], dtype=float, device=device)
    nut_wp = wp.array(mesh_data['nut_data'], dtype=float, device=device)
    cell_centers_wp = wp.array(mesh_data['cell_centers'], dtype=wp.vec3, device=device)
    cell_volumes_wp = wp.array(mesh_data['cell_volumes'], dtype=float, device=device)
    
    # Batch face data
    face_owner_wp = wp.array(batch_face_owner, dtype=int, device=device)
    face_neighbor_wp = wp.array(batch_face_neighbor, dtype=int, device=device)
    face_area_wp = wp.array(batch_face_area, dtype=float, device=device)
    face_normal_wp = wp.array(batch_face_normal, dtype=wp.vec3, device=device)
    face_centers_wp = wp.array(batch_face_centers, dtype=wp.vec3, device=device)
    
    # Cell-to-batch mapping
    cell_to_batch_idx_wp = wp.array(cell_to_batch_idx, dtype=int, device=device)
    
    # Continuity (batch-sized output)
    continuity_wp = wp.zeros(n_batch, dtype=float, device=device)
    wp.launch(
        kernel=compute_face_flux_kernel_batched,
        dim=n_batch_faces,
        inputs=[n_batch_faces, velocity_wp, face_owner_wp, face_neighbor_wp,
                face_area_wp, face_normal_wp, cell_centers_wp, face_centers_wp,
                n_cells, cell_to_batch_idx_wp, continuity_wp],
        device=device
    )
    
    # Velocity gradients (batch-sized output)
    grad_u_wp = wp.zeros(n_batch, dtype=wp.vec3, device=device)
    grad_v_wp = wp.zeros(n_batch, dtype=wp.vec3, device=device)
    grad_w_wp = wp.zeros(n_batch, dtype=wp.vec3, device=device)
    wp.launch(
        kernel=compute_velocity_gradients_kernel_batched,
        dim=n_batch_faces,
        inputs=[n_batch_faces, velocity_wp, face_owner_wp, face_neighbor_wp,
                face_area_wp, face_normal_wp, face_centers_wp, cell_centers_wp,
                cell_volumes_wp, n_cells, cell_to_batch_idx_wp,
                grad_u_wp, grad_v_wp, grad_w_wp],
        device=device
    )
    
    # Momentum (batch-sized output)
    momentum_x_wp = wp.zeros(n_batch, dtype=float, device=device)
    momentum_y_wp = wp.zeros(n_batch, dtype=float, device=device)
    momentum_z_wp = wp.zeros(n_batch, dtype=float, device=device)
    include_convection = 0 if stokes_flow else 1  # 0 = Stokes, 1 = Navier-Stokes
    use_laplacian_diffusion = 1 if stokes_flow else 0  # 1 = Laplacian for Stokes, 0 = stress tensor for NS
    wp.launch(
        kernel=compute_momentum_flux_kernel_batched,
        dim=n_batch_faces,
        inputs=[n_batch_faces, velocity_wp, pressure_wp, nut_wp,
                grad_u_wp, grad_v_wp, grad_w_wp, nu, include_convection,
                use_laplacian_diffusion,
                face_owner_wp, face_neighbor_wp, face_area_wp, face_normal_wp,
                face_centers_wp, cell_centers_wp, n_cells, cell_to_batch_idx_wp,
                grad_u_wp, grad_v_wp, grad_w_wp,
                momentum_x_wp, momentum_y_wp, momentum_z_wp],
        device=device
    )
    
    wp.synchronize()
    
    return {
        'continuity': continuity_wp.numpy(),
        'momentum_x': momentum_x_wp.numpy(),
        'momentum_y': momentum_y_wp.numpy(),
        'momentum_z': momentum_z_wp.numpy(),
    }


# ============================================================================
# PyTorch Autograd Function with Warp Autodiff Support
# ============================================================================

class FVMResidualsAutogradFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function that wraps face-based Warp FVM kernels
    with true automatic differentiation support.
    
    Uses Warp's adjoint=True flag to run backward kernels for gradient computation.
    """
    
    @staticmethod
    def forward(
        ctx,
        velocity_data: torch.Tensor,
        pressure_data: torch.Tensor,
        nut_data: torch.Tensor,
        cell_centers: torch.Tensor,
        cell_volumes: torch.Tensor,
        face_owner: torch.Tensor,
        face_neighbor: torch.Tensor,
        face_area: torch.Tensor,
        face_normal: torch.Tensor,
        face_centers: torch.Tensor,
        cell_to_batch_idx: torch.Tensor,
        batch_cell_indices: torch.Tensor,
        n_cells: int,
        n_batch: int,
        n_batch_faces: int,
        nu: float,
        device_str: str,
        stokes_flow: bool = False
    ):
        """
        Forward pass: compute FVM residuals using face-based Warp kernels.
        
        Args:
            stokes_flow: If True, use Stokes equations (no convection). If False, use Navier-Stokes.
        """
        wp_device = device_str
        
        # Convert PyTorch tensors to Warp arrays with requires_grad=True
        ctx.velocity_wp = wp.from_torch(velocity_data.contiguous(), dtype=wp.vec3, requires_grad=True)
        ctx.pressure_wp = wp.from_torch(pressure_data.contiguous(), dtype=wp.float32, requires_grad=True)
        ctx.nut_wp = wp.from_torch(nut_data.contiguous(), dtype=wp.float32, requires_grad=True)
        
        # Non-differentiable mesh data
        ctx.cell_centers_wp = wp.from_torch(cell_centers.contiguous(), dtype=wp.vec3)
        ctx.cell_volumes_wp = wp.from_torch(cell_volumes.contiguous(), dtype=wp.float32)
        ctx.face_owner_wp = wp.from_torch(face_owner.contiguous().int(), dtype=wp.int32)
        ctx.face_neighbor_wp = wp.from_torch(face_neighbor.contiguous().int(), dtype=wp.int32)
        ctx.face_area_wp = wp.from_torch(face_area.contiguous(), dtype=wp.float32)
        ctx.face_normal_wp = wp.from_torch(face_normal.contiguous(), dtype=wp.vec3)
        ctx.face_centers_wp = wp.from_torch(face_centers.contiguous(), dtype=wp.vec3)
        ctx.cell_to_batch_idx_wp = wp.from_torch(cell_to_batch_idx.contiguous().int(), dtype=wp.int32)
        
        # Store context
        ctx.n_cells = n_cells
        ctx.n_batch = n_batch
        ctx.n_batch_faces = n_batch_faces
        ctx.nu = nu
        ctx.wp_device = wp_device
        ctx.include_convection = 0 if stokes_flow else 1  # 0 = Stokes, 1 = Navier-Stokes
        ctx.use_laplacian_diffusion = 1 if stokes_flow else 0  # 1 = Laplacian for Stokes, 0 = stress tensor for NS
        
        # Allocate output arrays with requires_grad=True
        ctx.continuity_wp = wp.zeros(n_batch, dtype=wp.float32, device=wp_device, requires_grad=True)
        ctx.grad_u_wp = wp.zeros(n_batch, dtype=wp.vec3, device=wp_device, requires_grad=True)
        ctx.grad_v_wp = wp.zeros(n_batch, dtype=wp.vec3, device=wp_device, requires_grad=True)
        ctx.grad_w_wp = wp.zeros(n_batch, dtype=wp.vec3, device=wp_device, requires_grad=True)
        ctx.momentum_x_wp = wp.zeros(n_batch, dtype=wp.float32, device=wp_device, requires_grad=True)
        ctx.momentum_y_wp = wp.zeros(n_batch, dtype=wp.float32, device=wp_device, requires_grad=True)
        ctx.momentum_z_wp = wp.zeros(n_batch, dtype=wp.float32, device=wp_device, requires_grad=True)
        
        # Launch continuity kernel
        wp.launch(
            kernel=compute_face_flux_kernel_batched,
            dim=n_batch_faces,
            inputs=[
                n_batch_faces, ctx.velocity_wp, ctx.face_owner_wp, ctx.face_neighbor_wp,
                ctx.face_area_wp, ctx.face_normal_wp, ctx.cell_centers_wp, ctx.face_centers_wp,
                n_cells, ctx.cell_to_batch_idx_wp, ctx.continuity_wp
            ],
            device=wp_device
        )
        
        # Launch velocity gradient kernel
        wp.launch(
            kernel=compute_velocity_gradients_kernel_batched,
            dim=n_batch_faces,
            inputs=[
                n_batch_faces, ctx.velocity_wp, ctx.face_owner_wp, ctx.face_neighbor_wp,
                ctx.face_area_wp, ctx.face_normal_wp, ctx.face_centers_wp, ctx.cell_centers_wp,
                ctx.cell_volumes_wp, n_cells, ctx.cell_to_batch_idx_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp
            ],
            device=wp_device
        )
        
        # Launch momentum kernel
        wp.launch(
            kernel=compute_momentum_flux_kernel_batched,
            dim=n_batch_faces,
            inputs=[
                n_batch_faces, ctx.velocity_wp, ctx.pressure_wp, ctx.nut_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp, nu, ctx.include_convection,
                ctx.use_laplacian_diffusion,
                ctx.face_owner_wp, ctx.face_neighbor_wp, ctx.face_area_wp, ctx.face_normal_wp,
                ctx.face_centers_wp, ctx.cell_centers_wp, n_cells, ctx.cell_to_batch_idx_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp,
                ctx.momentum_x_wp, ctx.momentum_y_wp, ctx.momentum_z_wp
            ],
            device=wp_device
        )
        
        wp.synchronize()
        
        # Convert outputs to PyTorch tensors
        continuity = wp.to_torch(ctx.continuity_wp)
        momentum_x = wp.to_torch(ctx.momentum_x_wp)
        momentum_y = wp.to_torch(ctx.momentum_y_wp)
        momentum_z = wp.to_torch(ctx.momentum_z_wp)
        
        return continuity, momentum_x, momentum_y, momentum_z
    
    @staticmethod
    def backward(ctx, grad_continuity, grad_momentum_x, grad_momentum_y, grad_momentum_z):
        """
        Backward pass: compute gradients using Warp's adjoint kernels.
        """
        # Map incoming PyTorch gradients to Warp arrays
        ctx.continuity_wp.grad = wp.from_torch(grad_continuity.contiguous(), dtype=wp.float32)
        ctx.momentum_x_wp.grad = wp.from_torch(grad_momentum_x.contiguous(), dtype=wp.float32)
        ctx.momentum_y_wp.grad = wp.from_torch(grad_momentum_y.contiguous(), dtype=wp.float32)
        ctx.momentum_z_wp.grad = wp.from_torch(grad_momentum_z.contiguous(), dtype=wp.float32)
        
        # Zero the input gradients
        ctx.velocity_wp.grad = wp.zeros_like(ctx.velocity_wp)
        ctx.pressure_wp.grad = wp.zeros_like(ctx.pressure_wp)
        ctx.nut_wp.grad = wp.zeros_like(ctx.nut_wp)
        ctx.grad_u_wp.grad = wp.zeros(ctx.n_batch, dtype=wp.vec3, device=ctx.wp_device)
        ctx.grad_v_wp.grad = wp.zeros(ctx.n_batch, dtype=wp.vec3, device=ctx.wp_device)
        ctx.grad_w_wp.grad = wp.zeros(ctx.n_batch, dtype=wp.vec3, device=ctx.wp_device)
        
        # Backward through momentum kernel (reverse order!)
        wp.launch(
            kernel=compute_momentum_flux_kernel_batched,
            dim=ctx.n_batch_faces,
            inputs=[
                ctx.n_batch_faces, ctx.velocity_wp, ctx.pressure_wp, ctx.nut_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp, ctx.nu, ctx.include_convection,
                ctx.use_laplacian_diffusion,
                ctx.face_owner_wp, ctx.face_neighbor_wp, ctx.face_area_wp, ctx.face_normal_wp,
                ctx.face_centers_wp, ctx.cell_centers_wp, ctx.n_cells, ctx.cell_to_batch_idx_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp,
                ctx.momentum_x_wp, ctx.momentum_y_wp, ctx.momentum_z_wp
            ],
            adj_inputs=[
                None, ctx.velocity_wp.grad, ctx.pressure_wp.grad, ctx.nut_wp.grad,
                ctx.grad_u_wp.grad, ctx.grad_v_wp.grad, ctx.grad_w_wp.grad, None, None,
                None,  # use_laplacian_diffusion (not differentiable)
                None, None, None, None,
                None, None, None, None,
                None, None, None,
                ctx.momentum_x_wp.grad, ctx.momentum_y_wp.grad, ctx.momentum_z_wp.grad
            ],
            device=ctx.wp_device,
            adjoint=True
        )
        
        # Backward through velocity gradient kernel
        wp.launch(
            kernel=compute_velocity_gradients_kernel_batched,
            dim=ctx.n_batch_faces,
            inputs=[
                ctx.n_batch_faces, ctx.velocity_wp, ctx.face_owner_wp, ctx.face_neighbor_wp,
                ctx.face_area_wp, ctx.face_normal_wp, ctx.face_centers_wp, ctx.cell_centers_wp,
                ctx.cell_volumes_wp, ctx.n_cells, ctx.cell_to_batch_idx_wp,
                ctx.grad_u_wp, ctx.grad_v_wp, ctx.grad_w_wp
            ],
            adj_inputs=[
                None, ctx.velocity_wp.grad, None, None,
                None, None, None, None,
                None, None, None,
                ctx.grad_u_wp.grad, ctx.grad_v_wp.grad, ctx.grad_w_wp.grad
            ],
            device=ctx.wp_device,
            adjoint=True
        )
        
        # Backward through continuity kernel
        wp.launch(
            kernel=compute_face_flux_kernel_batched,
            dim=ctx.n_batch_faces,
            inputs=[
                ctx.n_batch_faces, ctx.velocity_wp, ctx.face_owner_wp, ctx.face_neighbor_wp,
                ctx.face_area_wp, ctx.face_normal_wp, ctx.cell_centers_wp, ctx.face_centers_wp,
                ctx.n_cells, ctx.cell_to_batch_idx_wp, ctx.continuity_wp
            ],
            adj_inputs=[
                None, ctx.velocity_wp.grad, None, None,
                None, None, None, None,
                None, None, ctx.continuity_wp.grad
            ],
            device=ctx.wp_device,
            adjoint=True
        )
        
        wp.synchronize()
        
        # Return gradients for inputs (None for non-differentiable inputs)
        grad_velocity = wp.to_torch(ctx.velocity_wp.grad)
        grad_pressure = wp.to_torch(ctx.pressure_wp.grad)
        grad_nut = wp.to_torch(ctx.nut_wp.grad)
        
        return (
            grad_velocity, grad_pressure, grad_nut,
            None, None,  # cell_centers, cell_volumes
            None, None, None, None, None,  # face data
            None, None,  # cell_to_batch_idx, batch_cell_indices
            None, None, None, None, None, None  # n_cells, n_batch, n_batch_faces, nu, device_str, stokes_flow
        )


def compute_residuals_warp_batch_autograd(
    velocity_data: torch.Tensor,
    pressure_data: torch.Tensor, 
    nut_data: torch.Tensor,
    mesh_data: dict,
    face_data: dict,
    batch_cell_indices: np.ndarray,
    nu: float,
    device: str = 'cuda:0',
    stokes_flow: bool = False
) -> tuple:
    """
    Compute FVM residuals with PyTorch autograd support using Warp autodiff.
    
    This is the recommended function for training - it properly integrates
    with PyTorch's autograd system for gradient computation.
    
    Args:
        velocity_data: [n_cells, 3] velocity field (torch tensor, requires_grad=True)
        pressure_data: [n_cells] pressure field (torch tensor, requires_grad=True)
        nut_data: [n_cells] turbulent viscosity (torch tensor, requires_grad=True)
        mesh_data: Dict with cell_centers, cell_volumes, n_cells
        face_data: Face connectivity from build_face_connectivity()
        batch_cell_indices: [n_batch] indices of cells to compute
        nu: Kinematic viscosity
        device: Device string
        stokes_flow: If True, use Stokes equations (no convection). If False, use Navier-Stokes.
    
    Returns:
        tuple of (continuity, momentum_x, momentum_y, momentum_z) torch tensors with gradients
    """
    batch_cell_indices = np.asarray(batch_cell_indices, dtype=np.int32)
    n_batch = len(batch_cell_indices)
    n_cells = mesh_data['n_cells']
    
    # Build batch face data
    batch_info = build_batch_face_data(face_data, batch_cell_indices, n_cells)
    batch_face_indices = batch_info['batch_face_indices']
    cell_to_batch_idx = batch_info['cell_to_batch_idx']
    n_batch_faces = len(batch_face_indices)
    
    # Extract face data for batch faces
    batch_face_owner = face_data['face_owner'][batch_face_indices]
    batch_face_neighbor = face_data['face_neighbor'][batch_face_indices]
    batch_face_area = face_data['face_area'][batch_face_indices]
    batch_face_normal = face_data['face_normal'][batch_face_indices]
    batch_face_centers = face_data['face_centers'][batch_face_indices]
    
    # Convert mesh data to torch tensors
    torch_device = torch.device(device.replace('cuda:', 'cuda:') if 'cuda' in device else device)
    
    cell_centers_t = torch.from_numpy(mesh_data['cell_centers'].astype(np.float32)).to(torch_device)
    cell_volumes_t = torch.from_numpy(mesh_data['cell_volumes'].astype(np.float32)).to(torch_device)
    face_owner_t = torch.from_numpy(batch_face_owner.astype(np.int32)).to(torch_device)
    face_neighbor_t = torch.from_numpy(batch_face_neighbor.astype(np.int32)).to(torch_device)
    face_area_t = torch.from_numpy(batch_face_area.astype(np.float32)).to(torch_device)
    face_normal_t = torch.from_numpy(batch_face_normal.astype(np.float32)).to(torch_device)
    face_centers_t = torch.from_numpy(batch_face_centers.astype(np.float32)).to(torch_device)
    cell_to_batch_idx_t = torch.from_numpy(cell_to_batch_idx.astype(np.int32)).to(torch_device)
    batch_indices_t = torch.from_numpy(batch_cell_indices.astype(np.int32)).to(torch_device)
    
    # Call the autograd function
    return FVMResidualsAutogradFunction.apply(
        velocity_data,
        pressure_data,
        nut_data,
        cell_centers_t,
        cell_volumes_t,
        face_owner_t,
        face_neighbor_t,
        face_area_t,
        face_normal_t,
        face_centers_t,
        cell_to_batch_idx_t,
        batch_indices_t,
        n_cells,
        n_batch,
        n_batch_faces,
        nu,
        device,
        stokes_flow
    )


def compute_residuals_warp(mesh_data, face_data, nu, batch_size=2048, device='cuda:0', verbose=True, stokes_flow=False):
    """
    Compute residuals for full mesh using batched implementation.
    
    This processes the entire mesh in batches using the same batched kernels
    used for training (compute_residuals_warp_batch).
    
    Args:
        mesh_data: Full mesh data from load_vtu_mesh()
        face_data: Full face connectivity from build_face_connectivity()
        nu: Kinematic viscosity
        batch_size: Number of cells per batch
        device: Warp device
        verbose: Print progress
        stokes_flow: If True, use Stokes equations (no convection). If False, use Navier-Stokes.
    
    Returns:
        dict with continuity, momentum_x, momentum_y, momentum_z arrays of shape [n_cells]
    """
    n_cells = mesh_data['n_cells']
    
    # Allocate full-mesh output
    continuity_full = np.zeros(n_cells, dtype=np.float32)
    momentum_x_full = np.zeros(n_cells, dtype=np.float32)
    momentum_y_full = np.zeros(n_cells, dtype=np.float32)
    momentum_z_full = np.zeros(n_cells, dtype=np.float32)
    
    n_batches = (n_cells + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Computing residuals in {n_batches} batches of {batch_size} cells...")
        iterator = tqdm(range(0, n_cells, batch_size), desc="  Batches", unit="batch")
    else:
        iterator = range(0, n_cells, batch_size)
    
    for start_idx in iterator:
        end_idx = min(start_idx + batch_size, n_cells)
        batch_indices = np.arange(start_idx, end_idx, dtype=np.int32)
        
        # Compute residuals for this batch
        residuals = compute_residuals_warp_batch(mesh_data, face_data, batch_indices, nu, device, stokes_flow)
        
        # Store results at correct positions
        continuity_full[start_idx:end_idx] = residuals['continuity']
        momentum_x_full[start_idx:end_idx] = residuals['momentum_x']
        momentum_y_full[start_idx:end_idx] = residuals['momentum_y']
        momentum_z_full[start_idx:end_idx] = residuals['momentum_z']
    
    if verbose:
        print(f"  Done!")
    
    return {
        'continuity': continuity_full,
        'momentum_x': momentum_x_full,
        'momentum_y': momentum_y_full,
        'momentum_z': momentum_z_full,
    }


def save_results_to_vtu(ugrid, residuals, output_path):
    """Save residuals to VTU file."""
    from vtk.util import numpy_support
    import vtk
    
    print(f"Saving results to {output_path}...")
    
    for name, data in residuals.items():
        arr = numpy_support.numpy_to_vtk(data)
        arr.SetName(f"FVM_{name}")
        ugrid.GetCellData().AddArray(arr)
    
    # Momentum magnitude
    mom_mag = np.sqrt(residuals['momentum_x']**2 + residuals['momentum_y']**2 + residuals['momentum_z']**2)
    arr = numpy_support.numpy_to_vtk(mom_mag)
    arr.SetName("FVM_momentum_magnitude")
    ugrid.GetCellData().AddArray(arr)
    
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(ugrid)
    writer.Write()
    
    print(f"  Saved!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute FVM residuals on VTU mesh using Warp")
    parser.add_argument("mesh_file", help="Path to VTU mesh file")
    parser.add_argument("--velocity", default="UMean", help="Velocity field name")
    parser.add_argument("--pressure", default="pMean", help="Pressure field name")
    parser.add_argument("--nut", default="nutMean", help="Turbulent viscosity field name")
    parser.add_argument("--nu", type=float, default=1.5e-5, help="Kinematic viscosity")
    parser.add_argument("--device", default="cuda:0", help="Warp device")
    parser.add_argument("--save", help="Output VTU file path")
    parser.add_argument("--batch-size", type=int, default=2048000, help="Batch size for computation")
    args = parser.parse_args()
    
    print("=" * 80)
    print("FVM Residuals Computation (Face-based Warp)")
    print("=" * 80)
    
    # Load mesh
    mesh_data = load_vtu_mesh(args.mesh_file, args.velocity, args.pressure, args.nut)
    
    # Build face connectivity
    face_data = build_face_connectivity(mesh_data['ugrid'])
    
    # Compute residuals (batched)
    residuals = compute_residuals_warp(
        mesh_data, face_data, args.nu, args.batch_size, args.device
    )
    
    # Print statistics
    print("\nResidual Statistics:")
    for name, data in residuals.items():
        print(f"  {name:12s}: min={data.min():.2e}, max={data.max():.2e}, mean={data.mean():.2e}")
    
    # Save if requested
    if args.save:
        save_results_to_vtu(mesh_data['ugrid'], residuals, args.save)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    
    return mesh_data, face_data, residuals


if __name__ == "__main__":
    main()
