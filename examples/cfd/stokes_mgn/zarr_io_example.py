"""Minimal zarr I/O example for FVM data."""
import numpy as np
import zarr


def load_fvm_data(zarr_path):
    """
    Load FVM connectivity and field data from zarr.

    Args:
        zarr_path: Path to zarr store

    Returns:
        dict with all arrays needed for FVM computation
    """
    store = zarr.open(zarr_path, mode='r')

    # Face connectivity (n_faces,)
    face_owner = np.array(store['face_owner'][:], dtype=np.int32)
    face_neighbor = np.array(store['face_neighbor'][:], dtype=np.int32)

    # Face geometry (n_faces,) and (n_faces, 3)
    face_area = np.array(store['face_area'][:], dtype=np.float32)
    face_normal = np.array(store['face_normal'][:], dtype=np.float32)
    face_centers = np.array(store['face_centers'][:], dtype=np.float32)

    # Cell geometry (n_cells,) and (n_cells, 3)
    cell_centers = np.array(store['cell_centers'][:], dtype=np.float32)
    cell_volumes = np.array(store['cell_volumes'][:], dtype=np.float32)

    # Field data (n_cells, 3) - columns are (u, v, p)
    volume_fields = np.array(store['volume_fields'][:], dtype=np.float32)

    n_cells = len(cell_volumes)
    n_faces = len(face_owner)

    return {
        # Connectivity
        'face_owner': face_owner,
        'face_neighbor': face_neighbor,
        # Face geometry
        'face_area': face_area,
        'face_normal': face_normal,
        'face_centers': face_centers,
        # Cell geometry
        'cell_centers': cell_centers,
        'cell_volumes': cell_volumes,
        # Fields
        'u': volume_fields[:, 0],
        'v': volume_fields[:, 1],
        'p': volume_fields[:, 2],
        # Counts
        'n_cells': n_cells,
        'n_faces': n_faces,
        # Stl data
        'stl_coordinates': np.array(store['stl_coordinates'][:], dtype=np.float32),
        'stl_centers': np.array(store['stl_centers'][:], dtype=np.float32)
    }


def compute_sdf_2d(cell_centers, stl_coords):
    """
    Compute signed distance field from cell centers to STL boundary.
    
    Uses 2D projection (x, y) and approximates sign by checking if point
    is inside the convex hull of the STL boundary.
    
    Args:
        cell_centers: [n_cells, 3] cell center coordinates
        stl_coords: [n_stl, 3] STL vertex coordinates
    
    Returns:
        [n_cells] signed distance values (negative inside, positive outside)
    """
    from scipy.spatial import cKDTree, Delaunay
    
    # Project to 2D (x, y)
    cells_2d = cell_centers[:, :2]
    stl_2d = stl_coords[:, :2]
    
    # Get unique STL boundary points
    stl_unique = np.unique(stl_2d, axis=0)
    
    # Build KD-tree for fast distance queries
    tree = cKDTree(stl_unique)
    distances, _ = tree.query(cells_2d)
    
    # Determine sign using Delaunay triangulation of STL points
    # Points inside the convex hull of STL get negative distance
    try:
        hull = Delaunay(stl_unique)
        inside = hull.find_simplex(cells_2d) >= 0
        sdf = np.where(inside, -distances, distances)
    except:
        # If Delaunay fails, just return unsigned distance
        sdf = distances
    
    return sdf


def plot_sdf(data, output_path='sdf.png'):
    """Plot SDF as a 2D scatter plot and save to PNG."""
    import matplotlib.pyplot as plt
    
    cell_centers = data['cell_centers']
    stl_coords = data['stl_coordinates']
    
    # Compute SDF
    sdf = compute_sdf_2d(cell_centers, stl_coords)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot SDF
    ax = axes[0]
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1], 
                    c=sdf, cmap='RdBu', s=1, alpha=0.8)
    ax.scatter(stl_coords[:, 0], stl_coords[:, 1], c='black', s=1, alpha=0.5, label='STL boundary')
    plt.colorbar(sc, ax=ax, label='Signed Distance')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Signed Distance Field')
    ax.set_aspect('equal')
    ax.legend()
    
    # Plot velocity magnitude for reference
    ax = axes[1]
    vel_mag = np.sqrt(data['u']**2 + data['v']**2)
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1], 
                    c=vel_mag, cmap='viridis', s=1, alpha=0.8)
    ax.scatter(stl_coords[:, 0], stl_coords[:, 1], c='red', s=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Velocity Magnitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Magnitude')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved to {output_path}')
    plt.show()
    
    return sdf


if __name__ == "__main__":
    data = load_fvm_data('physics-curated/test/res_107.zarr')

    print(f"Cells: {data['n_cells']}")
    print(f"Faces: {data['n_faces']}")
    print(f"\nArrays:")
    print(f"  face_owner:   {data['face_owner'].shape}, {data['face_owner'].dtype}")
    print(f"  face_neighbor:{data['face_neighbor'].shape}, {data['face_neighbor'].dtype}")
    print(f"  face_area:    {data['face_area'].shape}, {data['face_area'].dtype}")
    print(f"  face_normal:  {data['face_normal'].shape}, {data['face_normal'].dtype}")
    print(f"  cell_centers: {data['cell_centers'].shape}, {data['cell_centers'].dtype}")
    print(f"  cell_volumes: {data['cell_volumes'].shape}, {data['cell_volumes'].dtype}")
    print(f"\nField ranges:")    
    print(f"  u: [{data['u'].min():.4f}, {data['u'].max():.4f}]")
    print(f"  v: [{data['v'].min():.4f}, {data['v'].max():.4f}]")
    print(f"  p: [{data['p'].min():.4f}, {data['p'].max():.4f}]")

    print(f"\nCoord ranges:")
    print(f"  x: [{data['cell_centers'][:,0].min():.4f}, {data['cell_centers'][:,0].max():.4f}]")
    print(f"  y: [{data['cell_centers'][:,1].min():.4f}, {data['cell_centers'][:,1].max():.4f}]")
    print(f"  z: [{data['cell_centers'][:,2].min():.4f}, {data['cell_centers'][:,2].max():.4f}]")

    print(f"\nSTL Coord ranges:")
    print(f"  x: [{data['stl_centers'][:,0].min():.4f}, {data['stl_centers'][:,0].max():.4f}]")
    print(f"  y: [{data['stl_centers'][:,1].min():.4f}, {data['stl_centers'][:,1].max():.4f}]")
    print(f"  z: [{data['stl_centers'][:,2].min():.4f}, {data['stl_centers'][:,2].max():.4f}]")
    
    # Compute and plot SDF
    print("\nComputing SDF...")
    sdf = plot_sdf(data, 'sdf.png')
    print(f"SDF range: [{sdf.min():.4f}, {sdf.max():.4f}]")