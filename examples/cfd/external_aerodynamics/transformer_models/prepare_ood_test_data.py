"""
Select 10 random geometries per vehicle class from the driveSim dataset,
merge the 3 surface VTP parts (body + front_wheels + back_wheels) into a
single VTK, and save to test_<class>/{NNN}_speed{S}.vtk.

Usage:
    python prepare_ood_test_data.py
"""
import csv
import os
import random
from collections import defaultdict

import pyvista as pv

CSV_PATH = "/lustre/fsw/coreai_modulus_cae/datasets/driveSim_openfoam/batch_post_legal_3_0.csv"
EXTRACTED_DIR = "/lustre/fsw/coreai_modulus_cae/datasets/driveSim_openfoam/extracted"
OUTPUT_BASE = os.path.dirname(os.path.abspath(__file__))

SURFACE_PARTS = ["aero_suv.vtp", "front_wheels.vtp", "back_wheels.vtp"]
N_PER_CLASS = 10
SEED = 42


def boundary_dir_for_index(idx: int) -> str:
    prefix = f"batch_post_legal_3_0_{idx}"
    return os.path.join(
        EXTRACTED_DIR,
        f"{prefix}_VTK", prefix, "VTK", f"{prefix}_5000", "boundary",
    )


def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    classes = defaultdict(list)
    for row in rows:
        classes[row["Vehicle Class"]].append(row)

    print(f"Classes found: {sorted(classes.keys())}")
    print(f"Selecting {N_PER_CLASS} per class, seed={SEED}\n")

    random.seed(SEED)
    summary = []

    for cls_name in sorted(classes.keys()):
        cls_rows = classes[cls_name]
        dir_name = f"test_{cls_name.lower().replace(' ', '_')}"
        out_dir = os.path.join(OUTPUT_BASE, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        available = [
            r for r in cls_rows
            if os.path.isdir(boundary_dir_for_index(int(r["Index"])))
        ]
        if len(available) < N_PER_CLASS:
            print(f"WARNING: {cls_name} has only {len(available)} available "
                  f"(of {len(cls_rows)} in CSV)")

        selected = random.sample(available, min(N_PER_CLASS, len(available)))
        count = 0

        for i, row in enumerate(selected):
            idx = int(row["Index"])
            speed = int(row["Speed"])
            bdir = boundary_dir_for_index(idx)

            parts = []
            for part_name in SURFACE_PARTS:
                vtp_path = os.path.join(bdir, part_name)
                if not os.path.isfile(vtp_path):
                    print(f"  MISSING: {vtp_path}")
                    continue
                parts.append(pv.read(vtp_path))

            if not parts:
                print(f"  SKIP: no VTP parts for {cls_name} idx={idx}")
                continue

            merged = parts[0] if len(parts) == 1 else pv.merge(parts)

            out_path = os.path.join(out_dir, f"{i:03d}_speed{speed}.vtk")
            merged.save(out_path)
            print(f"  [{cls_name:12s}] idx={idx:3d}  speed={speed:2d}  "
                  f"cells={merged.n_cells:6d}  -> {os.path.basename(out_path)}")
            count += 1

        summary.append((cls_name, count, out_dir))

    print("\n--- Summary ---")
    for cls_name, count, out_dir in summary:
        print(f"  {cls_name:12s}: {count} files -> {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
