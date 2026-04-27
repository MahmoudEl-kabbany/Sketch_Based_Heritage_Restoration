import cv2
import numpy as np
from restoration.extraction import extract_paths
from restoration.candidates import _build_path_hulls, _point_in_hull, _path_length

img = cv2.imread('test_images/damaged_sketches/damaged_eye.png', cv2.IMREAD_GRAYSCALE)
ext = extract_paths(img)

hulls = _build_path_hulls(ext.paths)
lengths = {i: _path_length(p) for i, p in enumerate(ext.paths)}

for ep in ext.endpoints:
    plen = lengths[ep.path_index]
    is_noisy = False
    enclosed_by = []
    for p_idx, hull in hulls.items():
        if p_idx != ep.path_index and _point_in_hull(ep.position, hull):
            is_noisy = True
            enclosed_by.append(p_idx)
    if is_noisy:
        print(f"Endpoint {ep.path_index}({ep.end}): length={plen:.1f}, enclosed_by={enclosed_by}")
