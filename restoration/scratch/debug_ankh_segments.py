
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from restoration.extraction import extract_paths, _curvature_at
from restoration.candidates import generate_candidates

def debug_image(image_name):
    print(f"\n--- Debugging {image_name} ---")
    image_path = os.path.join("test_images/damaged_sketches", image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    result = extract_paths(image_path)
    
    for i, path in enumerate(result.paths):
        if path.is_closed: continue
        print(f"Path {i}: segments={len(path.segments)}")
        for j, seg in enumerate(path.segments):
            c_start = _curvature_at(seg.control_points, 0.0)
            c_mid = _curvature_at(seg.control_points, 0.5)
            c_end = _curvature_at(seg.control_points, 1.0)
            print(f"  Seg {j}: curv_start={c_start:.6f}, mid={c_mid:.6f}, end={c_end:.6f}")

if __name__ == "__main__":
    debug_image("damaged_ankh.png")
