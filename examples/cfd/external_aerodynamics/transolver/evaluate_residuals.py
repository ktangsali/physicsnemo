import pyvista as pv
import numpy as np
from tqdm import tqdm
from numba import njit
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

@njit
def compute_face_area_numba(point_ids, points):
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)
    
    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec += np.cross(edge1, edge2)
    
    return 0.5 * np.sqrt(np.sum(area_vec * area_vec))

@njit
def compute_face_normal_numba(point_ids, points, cell_center):
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)
    
    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec += np.cross(edge1, edge2)
    
    normal = area_vec / (2.0 * np.sqrt(np.sum(area_vec * area_vec)))
    
    face_center = np.zeros(3)
    for i in range(n_points):
        face_center += points[point_ids[i]]
    face_center /= n_points
    
    face_to_cell = cell_center - face_center
    
    if np.sum(normal * face_to_cell) > 0:
        normal = -normal
    
    return normal

@njit
def compute_face_velocity_numba(point_ids, velocity_data):
    vel = np.zeros(3)
    n = len(point_ids)
    for i in range(n):
        vel += velocity_data[point_ids[i]]
    return vel / n

@njit
def compute_face_scalar_numba(point_ids, scalar_data):
    val = 0.0
    n = len(point_ids)
    for i in range(n):
        val += scalar_data[point_ids[i]]
    return val / n

def compute_cell_gradient_gauss_green(cell_id, points, scalar_data, cell_volumes, cell_centers, neighbors, face_point_ids_map):
    gradient = np.zeros(3)
    cell1_center = cell_centers[cell_id]
    cell_volume = cell_volumes[cell_id]

    for neighbor_idx in neighbors[cell_id]:
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        scalar_face = compute_face_scalar_numba(face_point_ids, scalar_data)
        gradient += area * scalar_face * normal
    
    return gradient / cell_volume

@njit
def compute_viscous_stress_tensor(velocity_grad, nu_eff):
    tau = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            tau[i, j] = nu_eff * (velocity_grad[i, j] + velocity_grad[j, i])
    return tau

def compute_cell_momentum(cell_id, points, velocity_data, pressure_data, nut_data, nu, cell_volumes, cell_centers, cell_point_ids, neighbors, face_point_ids_map):
    momentum_residual = np.zeros(3)
    cell1_center = cell_centers[cell_id]
    
    u_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 0], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    v_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 1], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    w_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 2], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    
    velocity_grad = np.array([u_grad, v_grad, w_grad])
    
    cell_pts = cell_point_ids[cell_id]
    cell_velocity = np.mean([velocity_data[pt_id] for pt_id in cell_pts], axis=0)
    cell_pressure = np.mean([pressure_data[pt_id] for pt_id in cell_pts])
    cell_nut = np.mean([nut_data[pt_id] for pt_id in cell_pts])
    nu_eff = nu + cell_nut
    
    tau = compute_viscous_stress_tensor(velocity_grad, nu_eff)
    
    for neighbor_idx in neighbors[cell_id]:
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        vel_face = compute_face_velocity_numba(face_point_ids, velocity_data)
        pressure_face = compute_face_scalar_numba(face_point_ids, pressure_data)
        
        convective_flux = area * np.outer(vel_face, vel_face) @ normal
        pressure_flux = area * pressure_face * normal
        viscous_flux = area * tau @ normal
        
        momentum_residual += convective_flux + pressure_flux - viscous_flux
    
    return momentum_residual
    
def compute_cell_continuity(cell_id, points, velocity_data, cell_centers, neighbors, face_point_ids_map):
    flux = 0.0
    cell1_center = cell_centers[cell_id]

    for neighbor_idx in neighbors[cell_id]:
        if (cell_id, neighbor_idx) not in face_point_ids_map:
            continue
        
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        vel_face_center = compute_face_velocity_numba(face_point_ids, velocity_data)
        flux += area * np.sum(normal * vel_face_center)
    
    return flux

def compute_residuals_for_field(mesh, velocity_data, pressure_data, nut_data, nu, cell_volumes, points, cell_centers, cell_point_ids, neighbors, face_point_ids_map):
    n_cells = mesh.n_cells
    continuity = np.zeros(n_cells)
    momentum_x = np.zeros(n_cells)
    momentum_y = np.zeros(n_cells)
    momentum_z = np.zeros(n_cells)
    
    for idx in range(n_cells):
        continuity_cell = compute_cell_continuity(idx, points, velocity_data, cell_centers, neighbors, face_point_ids_map)
        momentum_cell = compute_cell_momentum(idx, points, velocity_data, pressure_data, nut_data, nu, cell_volumes, cell_centers, cell_point_ids, neighbors, face_point_ids_map)
        
        continuity[idx] = continuity_cell
        momentum_x[idx] = momentum_cell[0]
        momentum_y[idx] = momentum_cell[1]
        momentum_z[idx] = momentum_cell[2]
    
    return continuity, momentum_x, momentum_y, momentum_z

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

def process_file(args):
    vtu_file, nu, output_dir, box_position, box_length = args
    
    mesh = pv.read(vtu_file)
    mesh_with_sizes = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    cell_volumes = mesh_with_sizes.cell_data['Volume']
    
    points = mesh.points
    
    cell_centers = []
    cell_point_ids = []
    neighbors = []
    face_point_ids_map = {}

    for cell_id in range(mesh.n_cells):
        cell = mesh.get_cell(cell_id)
        cell_centers.append(cell.center)
        cell_point_ids.append(list(cell.point_ids))
        
        cell_neighbors = list(mesh.cell_neighbors(cell_id, 'faces'))
        neighbors.append(cell_neighbors)
        
        for neighbor_id in cell_neighbors:
            if (cell_id, neighbor_id) not in face_point_ids_map:
                neighbor_cell = mesh.get_cell(neighbor_id)
                
                for i in range(cell.n_faces):
                    face1 = cell.get_face(i)
                    face1_point_ids = set(face1.point_ids)
                    for j in range(neighbor_cell.n_faces):
                        face2_point_ids = set(neighbor_cell.get_face(j).point_ids)
                        if face1_point_ids == face2_point_ids:
                            face_arr = np.array(face1.point_ids, dtype=np.int64)
                            face_point_ids_map[(cell_id, neighbor_id)] = face_arr
                            face_point_ids_map[(neighbor_id, cell_id)] = face_arr
                            break
                    if (cell_id, neighbor_id) in face_point_ids_map:
                        break

    cell_centers = np.array(cell_centers)
    
    velocity_pred = mesh.point_data["PredictedU"]
    pressure_pred = mesh.point_data["Predictedp"]
    nut_pred = mesh.point_data["Predictednut"]
    
    velocity_true = mesh.point_data["U"]
    pressure_true = mesh.point_data["p"]
    nut_true = mesh.point_data["nut"]
    
    continuity_pred, momentum_x_pred, momentum_y_pred, momentum_z_pred = compute_residuals_for_field(
        mesh, velocity_pred, pressure_pred, nut_pred, nu, cell_volumes, points, 
        cell_centers, cell_point_ids, neighbors, face_point_ids_map
    )
    
    continuity_true, momentum_x_true, momentum_y_true, momentum_z_true = compute_residuals_for_field(
        mesh, velocity_true, pressure_true, nut_true, nu, cell_volumes, points, 
        cell_centers, cell_point_ids, neighbors, face_point_ids_map
    )
    
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
    
    mesh_cell = mesh.point_data_to_cell_data(pass_point_data=True)
    velocity_pred_cell = mesh_cell.cell_data["PredictedU"]
    velocity_true_cell = mesh_cell.cell_data["U"]
    pressure_pred_cell = mesh_cell.cell_data["Predictedp"]
    pressure_true_cell = mesh_cell.cell_data["p"]
    nut_pred_cell = mesh_cell.cell_data["Predictednut"]
    nut_true_cell = mesh_cell.cell_data["nut"]
    
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
    
    l2_U = np.sqrt(np.mean((velocity_pred_cell[mask] - velocity_true_cell[mask])**2))
    l2_p = np.sqrt(np.mean((pressure_pred_cell[mask] - pressure_true_cell[mask])**2))
    l2_nut = np.sqrt(np.mean((nut_pred_cell[mask] - nut_true_cell[mask])**2))
    
    l2_num_U = np.sqrt(np.sum((velocity_pred_cell[mask] - velocity_true_cell[mask])**2))
    l2_denom_U = np.sqrt(np.sum(velocity_true_cell[mask]**2))
    relative_l2_U = l2_num_U / (l2_denom_U + 1e-10)
    
    l2_num_p = np.sqrt(np.sum((pressure_pred_cell[mask] - pressure_true_cell[mask])**2))
    l2_denom_p = np.sqrt(np.sum(pressure_true_cell[mask]**2))
    relative_l2_p = l2_num_p / (l2_denom_p + 1e-10)
    
    l2_num_nut = np.sqrt(np.sum((nut_pred_cell[mask] - nut_true_cell[mask])**2))
    l2_denom_nut = np.sqrt(np.sum(nut_true_cell[mask]**2))
    relative_l2_nut = l2_num_nut / (l2_denom_nut + 1e-10)
    
    return {
        'filename': vtu_file.name,
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
    nu = 1.5498148291427463e-05
    input_dir = Path("test/airfrans/float32_full_expt_2")
    output_dir = Path("test/airfrans/float32_full_expt_2_with_res")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    box_position = np.array([-1.8, -1.2, 0.0])
    box_length = np.array([5.4, 2.4, 1.0])
    
    vtu_files = list(input_dir.glob("*.vtu"))
    
    args_list = [(vtu_file, nu, output_dir, box_position, box_length) for vtu_file in vtu_files]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    df = pd.DataFrame(results)
    df.to_csv("residuals_and_errors.csv", index=False)

if __name__ == "__main__":
    main()

