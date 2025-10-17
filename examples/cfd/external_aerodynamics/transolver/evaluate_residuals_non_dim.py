"""
Evaluate residuals and L2 errors on non-dimensionalized variables.

This script computes CFD residuals (continuity and momentum equations) and L2 errors
using non-dimensionalized flow variables. 

Non-dimensionalization scales:
- Length scale: L_ref = 1.0
- Velocity scale: U_ref = Uinf (extracted from filename)
- Density: rho = 1.0
- Kinematic viscosity: nu = 1.5498148291427463e-05

Variables are non-dimensionalized as:
- U* = U/U_ref
- p* = p/(rho*U_ref^2)
- nut* = nut/(U_ref*L_ref)
- nu* = nu/(U_ref*L_ref)

All residuals and L2 errors are computed on these non-dimensionalized variables,
making them dimensionless and comparable across different flow conditions.
"""

import pyvista as pv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

# Import residual computation functions from physics module
from physics import _prepare_mesh_data, _compute_residuals_fvm

def create_boundary_mask(mesh, box_position, box_length):
    bounds = [
        box_position[0], box_position[0] + box_length[0],
        box_position[1], box_position[1] + box_length[1],
        box_position[2], box_position[2] + box_length[2]
    ]
    
    inside_cell_ids = mesh.find_cells_within_bounds(bounds)
    
    mask = np.zeros(mesh.n_cells, dtype=bool)
    mask[inside_cell_ids] = True
    
    return mask

def extract_uinf_and_aoa_from_filename(filename):
    """Extract Uinf and AoA from filename like: pred_airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32_internal.vtu"""
    # Remove prefix and suffix, then split
    parts = filename.replace('pred_', '').replace('_internal', '').replace('.vtu', '').split('_')
    # Format after cleanup: airFoil2D_SST_Uinf_aoa_digit1_digit2_...
    # parts[0] = 'airFoil2D', parts[1] = 'SST', parts[2] = Uinf, parts[3] = AoA
    uinf = float(parts[2])
    aoa = float(parts[3])
    return uinf, aoa

def process_file(args):
    vtu_file, nu, output_dir, box_position, box_length, L_ref, rho, is_2d = args
    
    # Extract reference velocity and angle of attack from filename
    U_ref, aoa = extract_uinf_and_aoa_from_filename(vtu_file.name)
    
    mesh = pv.read(vtu_file)
    
    # Read dimensional data
    velocity_pred_dim = mesh.point_data["PredictedU"]
    pressure_pred_dim = mesh.point_data["Predictedp"]
    nut_pred_dim = mesh.point_data["Predictednut"]
    
    velocity_true_dim = mesh.point_data["U"]
    pressure_true_dim = mesh.point_data["p"]
    nut_true_dim = mesh.point_data["nut"]
    
    # Non-dimensionalize variables
    # U* = U/U_ref
    velocity_pred = velocity_pred_dim / U_ref
    velocity_true = velocity_true_dim / U_ref
    
    # p* = p/(rho*U_ref^2) = p/U_ref^2 (since rho=1.0)
    pressure_pred = pressure_pred_dim / (rho * U_ref**2)
    pressure_true = pressure_true_dim / (rho * U_ref**2)
    
    # nut* = nut/(U_ref*L_ref) = nut/U_ref (since L_ref=1.0)
    nut_pred = nut_pred_dim / (U_ref * L_ref)
    nut_true = nut_true_dim / (U_ref * L_ref)
    
    # nu* = nu/(U_ref*L_ref) = nu/U_ref (since L_ref=1.0)
    nu_star = nu / (U_ref * L_ref)
    
    # Temporarily store non-dimensionalized data in mesh for processing
    mesh.point_data["U_temp"] = velocity_pred.astype(np.float64)
    mesh.point_data["p_temp"] = pressure_pred.astype(np.float64)
    mesh.point_data["nut_temp"] = nut_pred.astype(np.float64)
    
    # Convert to VTK format for physics module
    import vtk
    from vtk.util import numpy_support
    
    # Create VTK unstructured grid from PyVista mesh
    ugrid_pred = mesh.cast_to_unstructured_grid()
    
    # Prepare mesh data and compute residuals using physics module
    mesh_data_pred = _prepare_mesh_data(ugrid_pred, "U_temp", "p_temp", "nut_temp", is_2d=is_2d)
    continuity_pred, momentum_x_pred, momentum_y_pred, momentum_z_pred = _compute_residuals_fvm(
        mesh_data_pred, nu_star, device="cpu", is_2d=is_2d
    )
    
    # Repeat for true values
    mesh.point_data["U_temp"] = velocity_true.astype(np.float64)
    mesh.point_data["p_temp"] = pressure_true.astype(np.float64)
    mesh.point_data["nut_temp"] = nut_true.astype(np.float64)
    
    ugrid_true = mesh.cast_to_unstructured_grid()
    mesh_data_true = _prepare_mesh_data(ugrid_true, "U_temp", "p_temp", "nut_temp", is_2d=is_2d)
    continuity_true, momentum_x_true, momentum_y_true, momentum_z_true = _compute_residuals_fvm(
        mesh_data_true, nu_star, device="cpu", is_2d=is_2d
    )
    
    # Extract cell volumes/areas from mesh_data
    cell_volumes = mesh_data_pred['cell_volumes']
    
    mask = create_boundary_mask(mesh, box_position, box_length)
    
    mesh.cell_data["Continuity_Pred"] = continuity_pred
    mesh.cell_data["Momentum_X_Pred"] = momentum_x_pred
    mesh.cell_data["Momentum_Y_Pred"] = momentum_y_pred
    mesh.cell_data["Momentum_Z_Pred"] = momentum_z_pred
    
    mesh.cell_data["Continuity_True"] = continuity_true
    mesh.cell_data["Momentum_X_True"] = momentum_x_true
    mesh.cell_data["Momentum_Y_True"] = momentum_y_true
    mesh.cell_data["Momentum_Z_True"] = momentum_z_true
    
    mesh.cell_data["BoundaryMask"] = mask.astype(np.float32)
    
    output_file = output_dir / vtu_file.name
    mesh.save(str(output_file))
    
    # Store non-dimensionalized data in mesh for cell conversion
    mesh.point_data["PredictedU_nondim"] = velocity_pred
    mesh.point_data["U_nondim"] = velocity_true
    mesh.point_data["Predictedp_nondim"] = pressure_pred
    mesh.point_data["p_nondim"] = pressure_true
    mesh.point_data["Predictednut_nondim"] = nut_pred
    mesh.point_data["nut_nondim"] = nut_true
    
    mesh_cell = mesh.point_data_to_cell_data(pass_point_data=True)
    
    # Use non-dimensionalized variables for L2 error computation
    velocity_pred_cell = mesh_cell.cell_data["PredictedU_nondim"]
    velocity_true_cell = mesh_cell.cell_data["U_nondim"]
    pressure_pred_cell = mesh_cell.cell_data["Predictedp_nondim"]
    pressure_true_cell = mesh_cell.cell_data["p_nondim"]
    nut_pred_cell = mesh_cell.cell_data["Predictednut_nondim"]
    nut_true_cell = mesh_cell.cell_data["nut_nondim"]
    
    masked_volumes = cell_volumes[mask]
    total_volume = masked_volumes.sum()
    
    total_continuity_pred = np.abs(continuity_pred[mask]).sum() / total_volume
    total_momentum_x_pred = np.abs(momentum_x_pred[mask]).sum() / total_volume
    total_momentum_y_pred = np.abs(momentum_y_pred[mask]).sum() / total_volume
    total_momentum_z_pred = np.abs(momentum_z_pred[mask]).sum() / total_volume
    
    total_continuity_true = np.abs(continuity_true[mask]).sum() / total_volume
    total_momentum_x_true = np.abs(momentum_x_true[mask]).sum() / total_volume
    total_momentum_y_true = np.abs(momentum_y_true[mask]).sum() / total_volume
    total_momentum_z_true = np.abs(momentum_z_true[mask]).sum() / total_volume
    
    # For 2D data, only compute L2 on in-plane components (U_x, U_y) to match continuity residual
    if is_2d:
        velocity_pred_cell_2d = velocity_pred_cell[mask, :2]  # Only x and y components
        velocity_true_cell_2d = velocity_true_cell[mask, :2]
    else:
        velocity_pred_cell_2d = velocity_pred_cell[mask]
        velocity_true_cell_2d = velocity_true_cell[mask]
    
    l2_U = np.sqrt(np.mean((velocity_pred_cell_2d - velocity_true_cell_2d)**2))
    l2_p = np.sqrt(np.mean((pressure_pred_cell[mask] - pressure_true_cell[mask])**2))
    l2_nut = np.sqrt(np.mean((nut_pred_cell[mask] - nut_true_cell[mask])**2))
    
    l2_num_U = np.sqrt(np.sum((velocity_pred_cell_2d - velocity_true_cell_2d)**2))
    l2_denom_U = np.sqrt(np.sum(velocity_true_cell_2d**2))
    relative_l2_U = l2_num_U / (l2_denom_U + 1e-10)
    
    l2_num_p = np.sqrt(np.sum((pressure_pred_cell[mask] - pressure_true_cell[mask])**2))
    l2_denom_p = np.sqrt(np.sum(pressure_true_cell[mask]**2))
    relative_l2_p = l2_num_p / (l2_denom_p + 1e-10)
    
    l2_num_nut = np.sqrt(np.sum((nut_pred_cell[mask] - nut_true_cell[mask])**2))
    l2_denom_nut = np.sqrt(np.sum(nut_true_cell[mask]**2))
    relative_l2_nut = l2_num_nut / (l2_denom_nut + 1e-10)
    
    # All metrics are computed on non-dimensionalized variables
    # Residuals are dimensionless (computed on non-dimensional variables)
    # L2 errors are dimensionless (errors between non-dimensional predictions and ground truth)
    return {
        'filename': vtu_file.name,
        'U_ref': U_ref,  # Reference velocity used for non-dimensionalization
        'aoa': aoa,  # Angle of attack (degrees)
        'total_continuity_pred': total_continuity_pred,
        'total_momentum_x_pred': total_momentum_x_pred,
        'total_momentum_y_pred': total_momentum_y_pred,
        'total_momentum_z_pred': total_momentum_z_pred,
        'total_continuity_true': total_continuity_true,
        'total_momentum_x_true': total_momentum_x_true,
        'total_momentum_y_true': total_momentum_y_true,
        'total_momentum_z_true': total_momentum_z_true,
        'l2_U': l2_U,
        'l2_p': l2_p,
        'l2_nut': l2_nut,
        'relative_l2_U': relative_l2_U,
        'relative_l2_p': relative_l2_p,
        'relative_l2_nut': relative_l2_nut
    }

def main():
    nu = 1.5498148291427463e-05  # Kinematic viscosity
    L_ref = 1.0  # Reference length scale
    rho = 1.0    # Reference density
    is_2d = True  # Set to True for 2D meshes (slices), False for full 3D meshes
    
    input_dir = Path("test/airfrans_dataset_orig/bfloat16_full_2")
    output_dir = Path("test/airfrans_dataset_orig/bfloat16_full_2_with_res_non_dim")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    box_position = np.array([-1.8, -1.2, 0.0])
    box_length = np.array([5.4, 2.4, 1.0])
    
    vtu_files = list(input_dir.glob("*.vtu"))
    
    args_list = [(vtu_file, nu, output_dir, box_position, box_length, L_ref, rho, is_2d) for vtu_file in vtu_files]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    df = pd.DataFrame(results)
    df.to_csv("residuals_and_errors_non_dim_2d.csv", index=False)

if __name__ == "__main__":
    main()

