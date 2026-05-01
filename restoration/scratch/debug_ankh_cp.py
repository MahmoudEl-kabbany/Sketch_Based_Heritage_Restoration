
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.getcwd())

from restoration.extraction import extract_paths, _curvature_at

def debug_image(image_name):
    image_path = os.path.join("test_images/damaged_sketches", image_name)
    result = extract_paths(image_path)
    
    # Path 0 is likely the main loop
    path = result.paths[0]
    print(f"Path 0, Seg 0 control points:\n{path.segments[0].control_points}")

if __name__ == "__main__":
    debug_image("damaged_ankh.png")
