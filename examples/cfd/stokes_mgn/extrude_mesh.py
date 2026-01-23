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

"""
Extrude 2D mesh to 3D and extract no-slip wall surfaces.

Marker convention (from utils.py):
- marker=1: inflow
- marker=2: outflow  
- marker=3: wall (no-slip)
- marker=4: polygon (no-slip)
"""

import argparse
import numpy as np
import vtk
from vtk.util import numpy_support

try:
    import pyvista as pv
except ImportError:
    raise ImportError("This script requires pyvista. Install with: pip install pyvista")


def extrude_2d_to_3d_volume(polydata_2d, z_min=0.0, z_max=1.0, n_layers=10):
    """Extrude 2D PolyData mesh to 3D volumetric cells (wedges/hexahedra)."""
    points_2d = np.array(polydata_2d.points)
    n_points_2d = len(points_2d)
    
    z_levels = np.linspace(z_min, z_max, n_layers + 1)

    all_points = []
    for z in z_levels:
        if points_2d.shape[1] == 2:
            pts = np.column_stack([points_2d, np.full(n_points_2d, z)])
        else:
            pts = points_2d.copy()
            pts[:, 2] = z
        all_points.append(pts)
    all_points = np.vstack(all_points)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(all_points.astype(np.float64)))

    cells = vtk.vtkCellArray()
    cell_types = []

    VTK_WEDGE = 13
    VTK_HEXAHEDRON = 12

    polys = polydata_2d.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()
    
    cell_connectivity = []
    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        n_pts = id_list.GetNumberOfIds()
        cell_connectivity.append([id_list.GetId(i) for i in range(n_pts)])

    for layer in range(n_layers):
        offset_bottom = layer * n_points_2d
        offset_top = (layer + 1) * n_points_2d
        
        for cell_pts in cell_connectivity:
            n_pts = len(cell_pts)
            
            if n_pts == 3:
                p0, p1, p2 = cell_pts
                cells.InsertNextCell(6)
                cells.InsertCellPoint(p0 + offset_bottom)
                cells.InsertCellPoint(p1 + offset_bottom)
                cells.InsertCellPoint(p2 + offset_bottom)
                cells.InsertCellPoint(p0 + offset_top)
                cells.InsertCellPoint(p1 + offset_top)
                cells.InsertCellPoint(p2 + offset_top)
                cell_types.append(VTK_WEDGE)
            elif n_pts == 4:
                p0, p1, p2, p3 = cell_pts
                cells.InsertNextCell(8)
                cells.InsertCellPoint(p0 + offset_bottom)
                cells.InsertCellPoint(p1 + offset_bottom)
                cells.InsertCellPoint(p2 + offset_bottom)
                cells.InsertCellPoint(p3 + offset_bottom)
                cells.InsertCellPoint(p0 + offset_top)
                cells.InsertCellPoint(p1 + offset_top)
                cells.InsertCellPoint(p2 + offset_top)
                cells.InsertCellPoint(p3 + offset_top)
                cell_types.append(VTK_HEXAHEDRON)
            else:
                raise ValueError(f"Unsupported 2D cell with {n_pts} points")

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtk_points)
    ugrid.SetCells(cell_types, cells)

    for i in range(polydata_2d.GetPointData().GetNumberOfArrays()):
        arr = polydata_2d.GetPointData().GetArray(i)
        arr_name = arr.GetName()
        arr_np = numpy_support.vtk_to_numpy(arr)
        arr_combined = np.tile(arr_np, n_layers + 1)
        new_arr = numpy_support.numpy_to_vtk(arr_combined)
        new_arr.SetName(arr_name)
        ugrid.GetPointData().AddArray(new_arr)

    return pv.wrap(ugrid)


def extrude_lines_to_surface(lines_polydata, z_min, z_max, n_layers=10, 
                              domain_centroid=None, obstacle_centroid=None, 
                              edge_is_obstacle=None):
    """
    Extrude 1D lines to a 3D quad surface with proper normal orientation.
    
    Normals point INTO the flow domain:
    - Outer walls: toward domain centroid
    - Internal obstacle: away from obstacle centroid
    """
    points_2d = np.array(lines_polydata.points)
    n_points_2d = len(points_2d)
    
    z_levels = np.linspace(z_min, z_max, n_layers + 1)
    
    all_points = []
    for z in z_levels:
        pts = points_2d.copy()
        pts[:, 2] = z
        all_points.append(pts)
    all_points = np.vstack(all_points)
    
    lines_array = lines_polydata.lines
    
    edge_list = []
    i = 0
    while i < len(lines_array):
        n_pts = lines_array[i]
        if n_pts == 2:
            p0 = lines_array[i + 1]
            p1 = lines_array[i + 2]
            edge_list.append((p0, p1))
        i += n_pts + 1
    
    if domain_centroid is None:
        domain_centroid = np.mean(points_2d[:, :2], axis=0)
    
    cells = vtk.vtkCellArray()
    n_quads = 0
    
    for layer in range(n_layers):
        offset_bottom = layer * n_points_2d
        offset_top = (layer + 1) * n_points_2d
        
        for edge_idx, (p0, p1) in enumerate(edge_list):
            mid_x = (points_2d[p0, 0] + points_2d[p1, 0]) / 2.0
            mid_y = (points_2d[p0, 1] + points_2d[p1, 1]) / 2.0
            
            edge_dx = points_2d[p1, 0] - points_2d[p0, 0]
            edge_dy = points_2d[p1, 1] - points_2d[p0, 1]
            
            # Outward normal (perpendicular to edge)
            normal_x = -edge_dy
            normal_y = edge_dx
            
            is_obstacle = edge_is_obstacle[edge_idx] if edge_is_obstacle else False
            
            if is_obstacle and obstacle_centroid is not None:
                # Obstacle edges: normal points away from obstacle center
                to_obs_x = obstacle_centroid[0] - mid_x
                to_obs_y = obstacle_centroid[1] - mid_y
                dot = normal_x * to_obs_x + normal_y * to_obs_y
                should_flip = (dot >= 0)
            else:
                # Outer walls: normal points toward domain centroid
                to_centroid_x = domain_centroid[0] - mid_x
                to_centroid_y = domain_centroid[1] - mid_y
                dot = normal_x * to_centroid_x + normal_y * to_centroid_y
                should_flip = (dot < 0)
            
            cells.InsertNextCell(4)
            if not should_flip:
                cells.InsertCellPoint(p0 + offset_bottom)
                cells.InsertCellPoint(p1 + offset_bottom)
                cells.InsertCellPoint(p1 + offset_top)
                cells.InsertCellPoint(p0 + offset_top)
            else:
                cells.InsertCellPoint(p1 + offset_bottom)
                cells.InsertCellPoint(p0 + offset_bottom)
                cells.InsertCellPoint(p0 + offset_top)
                cells.InsertCellPoint(p1 + offset_top)
            n_quads += 1
    
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(all_points.astype(np.float64)))
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(cells)
    
    return pv.wrap(polydata)


def extract_noslip_edges_2d(polydata_2d):
    """
    Extract no-slip boundary edges (marker=3 or marker=4) as 1D lines.
    
    Returns:
        noslip_edges: list of (p0, p1) tuples
        noslip_lines: PyVista PolyData with line cells
        edge_is_obstacle: list of bools (True if edge is on obstacle)
        obstacle_centroid: [x, y] centroid of obstacle points (or None)
    """
    marker = np.array(polydata_2d.point_data["marker"])
    points = np.array(polydata_2d.points)
    n_points = len(points)

    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(n_points)])
    else:
        points = points.copy()
        points[:, 2] = 0.0

    wall_indices = set(np.where(marker == 3)[0])
    polygon_indices = set(np.where(marker == 4)[0])
    noslip_indices = wall_indices | polygon_indices

    obstacle_centroid = None
    if len(polygon_indices) > 0:
        polygon_pts = points[list(polygon_indices), :2]
        obstacle_centroid = np.mean(polygon_pts, axis=0)

    polys = polydata_2d.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    noslip_edges = []
    edge_is_obstacle = []
    edge_set = set()

    for _ in range(polys.GetNumberOfCells()):
        polys.GetNextCell(id_list)
        n_pts = id_list.GetNumberOfIds()

        for j in range(n_pts):
            p0 = id_list.GetId(j)
            p1 = id_list.GetId((j + 1) % n_pts)

            if p0 in noslip_indices and p1 in noslip_indices:
                edge = (min(p0, p1), max(p0, p1))
                if edge not in edge_set:
                    edge_set.add(edge)
                    noslip_edges.append((p0, p1))
                    is_obs = (p0 in polygon_indices) and (p1 in polygon_indices)
                    edge_is_obstacle.append(is_obs)

    if len(noslip_edges) == 0:
        return [], None, [], None

    used_points = set()
    for p0, p1 in noslip_edges:
        used_points.add(p0)
        used_points.add(p1)
    
    old_to_new = {old: new for new, old in enumerate(sorted(used_points))}
    new_points = points[sorted(used_points)]
    
    lines = vtk.vtkCellArray()
    for p0, p1 in noslip_edges:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(old_to_new[p0])
        lines.InsertCellPoint(old_to_new[p1])

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(new_points.astype(np.float64)))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)

    return noslip_edges, pv.wrap(polydata), edge_is_obstacle, obstacle_centroid


def main():
    parser = argparse.ArgumentParser(
        description="Extrude 2D mesh to 3D and compute implicit distance from no-slip walls."
    )
    parser.add_argument("input_vtp", type=str, help="Path to input 2D VTP mesh file")
    parser.add_argument("--extrusion-length", type=float, default=1.0,
                        help="Total extrusion length in z direction (default: 1.0)")
    parser.add_argument("--output-volume", type=str, default=None,
                        help="Output path for extruded volume VTU")
    parser.add_argument("--output-noslip", type=str, default=None,
                        help="Output path for extruded no-slip walls VTP")

    args = parser.parse_args()

    input_base = args.input_vtp.replace(".vtp", "")
    output_volume = args.output_volume or f"{input_base}_extruded.vtu"
    output_noslip = args.output_noslip or f"{input_base}_noslip_walls.vtp"

    print(f"Input: {args.input_vtp}")
    print(f"Extrusion length: {args.extrusion_length}")

    # Load 2D mesh
    polydata_2d = pv.read(args.input_vtp)
    print(f"2D mesh: {polydata_2d.n_points} points, {polydata_2d.n_cells} cells")

    if "marker" not in polydata_2d.point_data:
        raise ValueError("Input mesh must have 'marker' point data")

    # Ensure z=0
    points = np.array(polydata_2d.points)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    else:
        points[:, 2] = 0.0
    polydata_2d.points = points

    # Extract no-slip boundary edges
    noslip_edges, noslip_lines, edge_is_obstacle, obstacle_centroid = extract_noslip_edges_2d(polydata_2d)

    if noslip_lines is None or len(noslip_edges) == 0:
        print("No no-slip edges found! Extruding without distance field...")
        ugrid_3d = polydata_2d.extrude([0, 0, args.extrusion_length], capping=True)
        ugrid_3d.save(output_volume)
        print(f"Saved to: {output_volume}")
        return

    print(f"No-slip edges: {len(noslip_edges)}")

    # Extrude no-slip lines to 3D surface for implicit distance
    half_ext = args.extrusion_length / 2.0
    n_z_layers = 1
    
    domain_centroid = np.mean(points[:, :2], axis=0)
    
    noslip_surface = extrude_lines_to_surface(
        noslip_lines, z_min=-half_ext, z_max=half_ext, n_layers=n_z_layers,
        domain_centroid=domain_centroid, obstacle_centroid=obstacle_centroid,
        edge_is_obstacle=edge_is_obstacle
    )
    noslip_surface_tri = noslip_surface.triangulate()

    # Compute implicit distance on 2D mesh
    polydata_2d_with_dist = polydata_2d.compute_implicit_distance(noslip_surface_tri, inplace=False)
    dist = polydata_2d_with_dist.point_data["implicit_distance"]
    print(f"Implicit distance range: [{dist.min():.6f}, {dist.max():.6f}]")

    # Extrude to 3D volumetric mesh
    ugrid_3d = extrude_2d_to_3d_volume(polydata_2d_with_dist, z_min=-half_ext, z_max=half_ext, n_layers=n_z_layers)
    print(f"3D volume: {ugrid_3d.n_cells} cells, {ugrid_3d.n_points} points")

    # Create final no-slip surface
    noslip_final = extrude_lines_to_surface(
        noslip_lines, z_min=-half_ext, z_max=half_ext, n_layers=n_z_layers,
        domain_centroid=domain_centroid, obstacle_centroid=obstacle_centroid,
        edge_is_obstacle=edge_is_obstacle
    )
    noslip_final_tri = noslip_final.triangulate()

    # Save outputs
    noslip_final_tri.save(output_noslip)
    ugrid_3d.save(output_volume)
    print(f"Saved volume: {output_volume}")
    print(f"Saved no-slip walls: {output_noslip}")


if __name__ == "__main__":
    main()
