"""
Evaluate uncertainty quantification metrics for deep ensemble predictions.

This script computes ensemble statistics (mean, std) across 8 ensemble members,
then evaluates residuals and L2 errors using the ensemble mean predictions.
All computations use non-dimensionalized variables.
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
    filename, ensemble_dirs, nu, box_position, box_length, L_ref, rho, is_2d = args
    
    U_ref, aoa = extract_uinf_and_aoa_from_filename(filename)
    
    # Load ensemble members
    ensemble_velocities = []
    ensemble_pressures = []
    ensemble_nuts = []
    
    mesh = None
    for ensemble_dir in ensemble_dirs:
        vtu_file = ensemble_dir / filename
        mesh = pv.read(vtu_file)
        
        velocity_dim = mesh.point_data["PredictedU"]
        pressure_dim = mesh.point_data["Predictedp"]
        nut_dim = mesh.point_data["Predictednut"]
        
        # Non-dimensionalize
        velocity = velocity_dim / U_ref
        pressure = pressure_dim / (rho * U_ref**2)
        nut = nut_dim / (U_ref * L_ref)
        
        ensemble_velocities.append(velocity)
        ensemble_pressures.append(pressure)
        ensemble_nuts.append(nut)
    
    # Compute ensemble statistics
    ensemble_velocities = np.array(ensemble_velocities)
    ensemble_pressures = np.array(ensemble_pressures)
    ensemble_nuts = np.array(ensemble_nuts)
    
    velocity_mean = ensemble_velocities.mean(axis=0)
    pressure_mean = ensemble_pressures.mean(axis=0)
    nut_mean = ensemble_nuts.mean(axis=0)
    
    velocity_std = ensemble_velocities.std(axis=0)
    pressure_std = ensemble_pressures.std(axis=0)
    nut_std = ensemble_nuts.std(axis=0)
    
    # Load ground truth
    velocity_true_dim = mesh.point_data["U"]
    pressure_true_dim = mesh.point_data["p"]
    nut_true_dim = mesh.point_data["nut"]
    
    velocity_true = velocity_true_dim / U_ref
    pressure_true = pressure_true_dim / (rho * U_ref**2)
    nut_true = nut_true_dim / (U_ref * L_ref)
    
    nu_star = nu / (U_ref * L_ref)
    
    # Temporarily store non-dimensionalized data in mesh for processing
    mesh.point_data["U_temp"] = velocity_mean.astype(np.float64)
    mesh.point_data["p_temp"] = pressure_mean.astype(np.float64)
    mesh.point_data["nut_temp"] = nut_mean.astype(np.float64)
    
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
    
    # Convert to cell data for L2 computation
    mesh.point_data["velocity_mean"] = velocity_mean
    mesh.point_data["velocity_true"] = velocity_true
    mesh.point_data["pressure_mean"] = pressure_mean
    mesh.point_data["pressure_true"] = pressure_true
    mesh.point_data["nut_mean"] = nut_mean
    mesh.point_data["nut_true"] = nut_true
    mesh.point_data["velocity_std"] = velocity_std
    mesh.point_data["pressure_std"] = pressure_std
    mesh.point_data["nut_std"] = nut_std
    
    mesh_cell = mesh.point_data_to_cell_data(pass_point_data=False)
    
    velocity_mean_cell = mesh_cell.cell_data["velocity_mean"]
    velocity_true_cell = mesh_cell.cell_data["velocity_true"]
    pressure_mean_cell = mesh_cell.cell_data["pressure_mean"]
    pressure_true_cell = mesh_cell.cell_data["pressure_true"]
    nut_mean_cell = mesh_cell.cell_data["nut_mean"]
    nut_true_cell = mesh_cell.cell_data["nut_true"]
    
    velocity_std_cell = mesh_cell.cell_data["velocity_std"]
    pressure_std_cell = mesh_cell.cell_data["pressure_std"]
    nut_std_cell = mesh_cell.cell_data["nut_std"]
    
    masked_volumes = cell_volumes[mask]
    total_volume = masked_volumes.sum()
    
    # Residual metrics
    total_continuity_pred = np.abs(continuity_pred[mask]).sum() / total_volume
    total_momentum_x_pred = np.abs(momentum_x_pred[mask]).sum() / total_volume
    total_momentum_y_pred = np.abs(momentum_y_pred[mask]).sum() / total_volume
    total_momentum_z_pred = np.abs(momentum_z_pred[mask]).sum() / total_volume
    
    total_continuity_true = np.abs(continuity_true[mask]).sum() / total_volume
    total_momentum_x_true = np.abs(momentum_x_true[mask]).sum() / total_volume
    total_momentum_y_true = np.abs(momentum_y_true[mask]).sum() / total_volume
    total_momentum_z_true = np.abs(momentum_z_true[mask]).sum() / total_volume
    
    # L2 error metrics
    # For 2D data, only compute L2 on in-plane components (U_x, U_y) to match continuity residual
    if is_2d:
        velocity_mean_cell_2d = velocity_mean_cell[mask, :2]  # Only x and y components
        velocity_true_cell_2d = velocity_true_cell[mask, :2]
    else:
        velocity_mean_cell_2d = velocity_mean_cell[mask]
        velocity_true_cell_2d = velocity_true_cell[mask]
    
    l2_U = np.sqrt(np.mean((velocity_mean_cell_2d - velocity_true_cell_2d)**2))
    l2_p = np.sqrt(np.mean((pressure_mean_cell[mask] - pressure_true_cell[mask])**2))
    l2_nut = np.sqrt(np.mean((nut_mean_cell[mask] - nut_true_cell[mask])**2))
    
    l2_num_U = np.sqrt(np.sum((velocity_mean_cell_2d - velocity_true_cell_2d)**2))
    l2_denom_U = np.sqrt(np.sum(velocity_true_cell_2d**2))
    relative_l2_U = l2_num_U / (l2_denom_U + 1e-10)
    
    l2_num_p = np.sqrt(np.sum((pressure_mean_cell[mask] - pressure_true_cell[mask])**2))
    l2_denom_p = np.sqrt(np.sum(pressure_true_cell[mask]**2))
    relative_l2_p = l2_num_p / (l2_denom_p + 1e-10)
    
    l2_num_nut = np.sqrt(np.sum((nut_mean_cell[mask] - nut_true_cell[mask])**2))
    l2_denom_nut = np.sqrt(np.sum(nut_true_cell[mask]**2))
    relative_l2_nut = l2_num_nut / (l2_denom_nut + 1e-10)
    
    # Uncertainty metrics (total std per variable)
    total_std_U = np.mean(np.linalg.norm(velocity_std_cell[mask], axis=1))
    total_std_p = np.mean(pressure_std_cell[mask])
    total_std_nut = np.mean(nut_std_cell[mask])
    
    return {
        'filename': filename,
        'U_ref': U_ref,
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
        'relative_l2_nut': relative_l2_nut,
        'total_std_U': total_std_U,
        'total_std_p': total_std_p,
        'total_std_nut': total_std_nut
    }

def main():
    nu = 1.5498148291427463e-05
    L_ref = 1.0
    rho = 1.0
    is_2d = False  # Set to True for 2D meshes (slices), False for full 3D meshes
    
    base_dir = Path("test/airfrans")
    ensemble_dirs = [base_dir / f"float32_full_uq_{i}" for i in range(1, 9)]
    
    box_position = np.array([-1.8, -1.2, 0.0])
    box_length = np.array([5.4, 2.4, 1.0])
    
    # Get list of VTU files from first ensemble member
    vtu_files = list(ensemble_dirs[0].glob("*.vtu"))
    filenames = [vtu_file.name for vtu_file in vtu_files]
    
    args_list = [(filename, ensemble_dirs, nu, box_position, box_length, L_ref, rho, is_2d) for filename in filenames]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    df = pd.DataFrame(results)
    df.to_csv("uq_residuals_and_errors.csv", index=False)

if __name__ == "__main__":
    main()

