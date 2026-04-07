"""
Convert ShiftSUV VTP files to Zarr format for the Geo Transolver recipe.

Dataset-specific details:
  - Pressure field: "pressure_average" (subtract 101325 Pa gauge offset)
  - WSS field: "wall_shear_stress_average" (3-component vector)
  - Geometry translated by -0.25 in the z-direction
  - Fixed stream velocity: 30.0 m/s

Two-pass process:
  1. Inspect all VTPs: compute stats, detect zero-length normals and outliers.
  2. Convert only VTPs that pass filters.

Usage:
    python convert_shiftsuv_vtp_to_zarr.py --input_dir test_shiftsuv_estate_raw --output_dir test_shiftsuv_estate_zarr
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
import pyvista as pv
import zarr

AIR_DENSITY = 1.225
STREAM_VELOCITY = 30.0

PRESSURE_NAME = "pressure_average"
WSS_NAME = "wall_shear_stress_average"
PRESSURE_OFFSET = 101325.0
Z_TRANSLATE = -0.25

OUTLIER_STD = 5.0


def _build_wss_vector(mesh):
    cd = mesh.cell_data
    if WSS_NAME not in cd:
        raise KeyError(
            f"Missing '{WSS_NAME}' in cell_data. Available: {list(cd.keys())}"
        )
    wss = np.asarray(cd[WSS_NAME], dtype=np.float32)
    if wss.ndim == 1:
        wss = wss.reshape(-1, 3)
    if wss.shape[1] != 3:
        raise ValueError(
            f"Expected 3-component WSS vector, got shape {wss.shape}"
        )
    return wss


def _compute_field_stats(pressure, wss):
    p = np.asarray(pressure, dtype=np.float64).ravel()
    wss_mag = np.linalg.norm(wss, axis=1).astype(np.float64)
    valid_p = np.isfinite(p)
    valid_w = np.isfinite(wss_mag)
    n_cells = len(p)
    has_nan_inf = not (np.all(valid_p) and np.all(valid_w))
    if valid_p.sum() == 0 or valid_w.sum() == 0:
        return {
            "n_cells": n_cells, "has_nan_inf": has_nan_inf,
            "pressure_min": np.nan, "pressure_max": np.nan,
            "pressure_mean": np.nan, "pressure_std": np.nan,
            "wss_magnitude_min": np.nan, "wss_magnitude_max": np.nan,
            "wss_magnitude_mean": np.nan, "wss_magnitude_std": np.nan,
        }
    return {
        "n_cells": n_cells, "has_nan_inf": has_nan_inf,
        "pressure_min": float(np.min(p[valid_p])),
        "pressure_max": float(np.max(p[valid_p])),
        "pressure_mean": float(np.mean(p[valid_p])),
        "pressure_std": float(np.std(p[valid_p])),
        "wss_magnitude_min": float(np.min(wss_mag[valid_w])),
        "wss_magnitude_max": float(np.max(wss_mag[valid_w])),
        "wss_magnitude_mean": float(np.mean(wss_mag[valid_w])),
        "wss_magnitude_std": float(np.std(wss_mag[valid_w])),
    }


def _get_normals_raw(mesh):
    normals = mesh.compute_normals(
        cell_normals=True, point_normals=False, consistent_normals=True,
    )
    return -np.asarray(normals.cell_data["Normals"], dtype=np.float32)


def _has_zero_length_normals(normals, tol=1e-6):
    norms = np.linalg.norm(normals, axis=1)
    return (norms < tol).any()


def _load_mesh(vtp_path):
    mesh = pv.read(vtp_path)
    if hasattr(mesh, "cast_to_unstructured_grid"):
        mesh = mesh.cast_to_unstructured_grid().extract_surface()
    elif isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.extract_surface()
    mesh.points[:, 2] += Z_TRANSLATE
    return mesh


def inspect_vtp(vtp_path: str):
    mesh = _load_mesh(vtp_path)

    normals = _get_normals_raw(mesh)
    has_zero_normals = _has_zero_length_normals(normals)

    if PRESSURE_NAME not in mesh.cell_data:
        raise KeyError(
            f"Missing '{PRESSURE_NAME}' in cell_data. "
            f"Available: {list(mesh.cell_data.keys())}"
        )
    pressure = np.asarray(mesh.cell_data[PRESSURE_NAME], dtype=np.float32)
    pressure = pressure - PRESSURE_OFFSET
    if pressure.ndim == 1:
        pressure = pressure.reshape(-1, 1)
    wss = _build_wss_vector(mesh)
    stats = _compute_field_stats(pressure, wss)
    stats["has_zero_normals"] = has_zero_normals
    return stats


def _stl_faces_from_mesh(mesh):
    raw = np.asarray(mesh.faces, dtype=np.int32)
    tri_list = []
    i = 0
    while i < len(raw):
        n = raw[i]
        i += 1
        if n < 3:
            i += n
            continue
        idx = raw[i : i + n]
        i += n
        if n == 3:
            tri_list.append(idx)
        else:
            for j in range(1, n - 1):
                tri_list.append(np.array([idx[0], idx[j], idx[j + 1]]))
    return np.concatenate(tri_list, axis=0).astype(np.int32)


def vtp_to_zarr_one(vtp_path: str, zarr_path: str,
                     air_density: float, stream_velocity: float):
    mesh = _load_mesh(vtp_path)
    n_cells = mesh.n_cells

    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    assert centers.shape == (n_cells, 3)

    normals = _get_normals_raw(mesh)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    normals = (normals / norms).astype(np.float32)
    zero_norm = np.linalg.norm(normals, axis=1) < 1e-6
    if zero_norm.any():
        normals[zero_norm] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cell_sizes = mesh.compute_cell_sizes(
        length=False, area=True, volume=False, vertex_count=False
    )
    areas = np.asarray(cell_sizes.cell_data["Area"], dtype=np.float32)
    assert areas.shape == (n_cells,)

    if PRESSURE_NAME not in mesh.cell_data:
        raise KeyError(
            f"Missing '{PRESSURE_NAME}' in cell_data. "
            f"Available: {list(mesh.cell_data.keys())}"
        )
    pressure = np.asarray(mesh.cell_data[PRESSURE_NAME], dtype=np.float32)
    pressure = pressure - PRESSURE_OFFSET
    if pressure.ndim == 1:
        pressure = pressure.reshape(-1, 1)
    wss = _build_wss_vector(mesh)
    surface_fields = np.concatenate([pressure, wss], axis=1).astype(np.float32)
    assert surface_fields.shape == (n_cells, 4)

    stats = _compute_field_stats(pressure, wss)

    points = np.asarray(mesh.points, dtype=np.float32)
    stl_faces = _stl_faces_from_mesh(mesh)

    root = zarr.open_group(zarr_path, mode="w")

    def store(name, data, dtype):
        root.create_dataset(name, shape=data.shape, dtype=dtype, data=data)

    store("surface_mesh_centers", centers, np.float32)
    store("surface_normals", normals, np.float32)
    store("surface_fields", surface_fields, np.float32)
    store("surface_areas", areas, np.float32)
    store("stl_centers", centers, np.float32)
    store("stl_areas", areas, np.float32)
    store("stl_faces", stl_faces, np.int32)
    store("stl_coordinates", points, np.float32)
    root.create_dataset("air_density", shape=(), dtype=np.float32)
    root["air_density"][()] = np.float32(air_density)
    root.create_dataset("stream_velocity", shape=(), dtype=np.float32)
    root["stream_velocity"][()] = np.float32(stream_velocity)

    return stats


def _worker_inspect(args):
    (vtp_path,) = args
    try:
        stats = inspect_vtp(vtp_path)
        return (vtp_path, None, stats)
    except Exception as e:
        return (vtp_path, e, None)


def _worker_convert(args):
    vtp_path, zarr_path, air_density, stream_velocity = args
    try:
        stats = vtp_to_zarr_one(
            vtp_path, zarr_path,
            air_density=air_density,
            stream_velocity=stream_velocity,
        )
        return (vtp_path, None, stats)
    except Exception as e:
        return (vtp_path, e, None)


def _detect_outliers(stats_list, outlier_std: float):
    if not stats_list:
        return set(), {}
    pressure_max = np.array([s["pressure_max"] for _, s in stats_list], dtype=np.float64)
    wss_max = np.array([s["wss_magnitude_max"] for _, s in stats_list], dtype=np.float64)
    p_fin = np.isfinite(pressure_max)
    w_fin = np.isfinite(wss_max)
    p_med = np.median(pressure_max[p_fin]) if p_fin.any() else np.nan
    w_med = np.median(wss_max[w_fin]) if w_fin.any() else np.nan
    p_mad = np.median(np.abs(pressure_max[p_fin] - p_med)) if p_fin.any() else 0.0
    w_mad = np.median(np.abs(wss_max[w_fin] - w_med)) if w_fin.any() else 0.0
    p_std = np.std(pressure_max[p_fin]) if p_fin.sum() > 1 else 0.0
    w_std = np.std(wss_max[w_fin]) if w_fin.sum() > 1 else 0.0
    p_thresh = p_med + outlier_std * (1.4826 * p_mad if p_mad > 1e-30 else p_std) if np.isfinite(p_med) else np.inf
    w_thresh = w_med + outlier_std * (1.4826 * w_mad if w_mad > 1e-30 else w_std) if np.isfinite(w_med) else np.inf
    exclude_paths = set()
    for (p, s) in stats_list:
        if s.get("has_zero_normals", False):
            exclude_paths.add(p)
        elif s.get("has_nan_inf", False):
            exclude_paths.add(p)
        elif s["pressure_max"] > p_thresh or s["wss_magnitude_max"] > w_thresh:
            exclude_paths.add(p)
    global_stats = {
        "pressure_max_median": float(p_med), "pressure_max_threshold": float(p_thresh),
        "wss_magnitude_max_median": float(w_med), "wss_magnitude_max_threshold": float(w_thresh),
    }
    return exclude_paths, global_stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShiftSUV VTP files to zarr")
    parser.add_argument("--input_dir", required=True,
                        help="Directory with run_XXXXX.vtp files")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write .zarr groups")
    parser.add_argument("--stats_csv", default=None,
                        help="Path for stats CSV (default: vtk_stats_<input_dir_name>.csv)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = cpu_count - 1)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    stats_csv = args.stats_csv or f"vtk_stats_{os.path.basename(input_dir.rstrip('/'))}.csv"
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    vtp_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".vtp")]
    )
    if not vtp_files:
        raise SystemExit(f"No .vtp files found in {input_dir}")

    print(f"Found {len(vtp_files)} VTP files")
    print(f"Stream velocity: {STREAM_VELOCITY} m/s (fixed)")
    print(f"Pressure offset: -{PRESSURE_OFFSET} Pa")
    print(f"Z-translation: {Z_TRANSLATE}")

    os.makedirs(output_dir, exist_ok=True)

    # Pass 1: inspect
    print(f"\nPass 1: Inspecting {len(vtp_files)} VTP files (workers={workers})")
    inspect_tasks = [(os.path.join(input_dir, f),) for f in vtp_files]
    with Pool(workers) as pool:
        inspect_results = pool.map(_worker_inspect, inspect_tasks)

    failed = [(path, err) for path, err, _ in inspect_results if err is not None]
    if failed:
        for path, err in failed:
            print(f"FAILED {path}: {err}")
        raise SystemExit(len(failed))

    success_stats = [(p, s) for p, e, s in inspect_results if e is None and s is not None]
    exclude_paths, global_stats = _detect_outliers(success_stats, OUTLIER_STD)

    if exclude_paths:
        print(f"\nExcluding {len(exclude_paths)} files:")
        for p in sorted(exclude_paths):
            print(f"  {os.path.basename(p)}")

    # Pass 2: convert
    to_convert = []
    for f in vtp_files:
        vtp_path = os.path.join(input_dir, f)
        if vtp_path in exclude_paths:
            continue
        zarr_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.zarr")
        to_convert.append((vtp_path, zarr_path, AIR_DENSITY, STREAM_VELOCITY))

    if not to_convert:
        raise SystemExit("No VTP files passed filters; nothing to convert.")

    print(f"\nPass 2: Converting {len(to_convert)} VTP files to zarr (workers={workers})")
    with Pool(workers) as pool:
        convert_results = pool.map(_worker_convert, to_convert)

    convert_failed = [(path, err) for path, err, _ in convert_results if err is not None]
    if convert_failed:
        for path, err in convert_failed:
            print(f"FAILED {path}: {err}")
        raise SystemExit(len(convert_failed))

    # Write stats CSV
    all_results = inspect_results
    success = [(p, s) for p, err, s in all_results if err is None and s is not None]
    if success and stats_csv:
        with open(stats_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "file", "n_cells", "has_nan_inf", "has_zero_normals", "excluded",
                "pressure_min", "pressure_max", "pressure_mean", "pressure_std",
                "wss_magnitude_min", "wss_magnitude_max", "wss_magnitude_mean", "wss_magnitude_std",
            ])
            for (p, s) in success:
                fname = os.path.basename(p)
                excluded = p in exclude_paths
                w.writerow([
                    fname, s["n_cells"], s.get("has_nan_inf", False),
                    s.get("has_zero_normals", False), excluded,
                    s["pressure_min"], s["pressure_max"], s["pressure_mean"], s["pressure_std"],
                    s["wss_magnitude_min"], s["wss_magnitude_max"],
                    s["wss_magnitude_mean"], s["wss_magnitude_std"],
                ])
        print(f"  Wrote {stats_csv}")

    print(f"\nDone. Wrote {len(to_convert)} zarr groups to {output_dir} "
          f"(excluded {len(exclude_paths)} files)")


if __name__ == "__main__":
    main()
