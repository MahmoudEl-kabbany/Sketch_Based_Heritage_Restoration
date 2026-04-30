import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import networkx as nx
import numpy as np
from bezier_curves.medial_axis import _extract_vector_boundaries, _compute_voronoi_centerlines, _multilinestring_to_sknw_graph
from bezier_curves.bezier import _prune_skeleton_spurs
import cv2

image_path = 'test_images/damaged_sketches/damaged_star.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
h = img.shape[0]

polygons = _extract_vector_boundaries(image_path, image_height=h, min_area=150.0)
densify_factor = max(3.0, h * 0.0025)
centerlines = _compute_voronoi_centerlines(polygons, densify_factor=densify_factor, min_branch_factor=0.0)
graph = _multilinestring_to_sknw_graph(centerlines)

degrees = dict(graph.degree())
edges = list(graph.edges(keys=True, data=True) if graph.is_multigraph() else graph.edges(data=True))
print("All Terminal edges before pruning:")
for e in edges:
    u, v = e[0], e[1]
    du, dv = degrees.get(u, 0), degrees.get(v, 0)
    if (du == 1 and dv > 1) or (dv == 1 and du > 1):
        data = e[3] if len(e) > 3 else e[2]
        pts = data.get("pts", [])
        if len(pts) > 1:
            diffs = np.diff(np.array(pts), axis=0)
            length = np.sum(np.linalg.norm(diffs, axis=1))
        else:
            length = 0.0
        print(f"  Edge {u}-{v}: length = {length:.2f}, du={du}, dv={dv}")

