import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from bezier_curves.medial_axis import _extract_vector_boundaries, _compute_voronoi_centerlines
import cv2

image_path = 'test_images/damaged_sketches/restoration_test_damaged_big.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
h = img.shape[0]

print("Extracting boundaries...")
t0 = time.time()
polygons = _extract_vector_boundaries(image_path, image_height=h, min_area=150.0)
print(f"Boundaries extracted in {time.time()-t0:.2f}s")
for i, p in enumerate(polygons):
    print(f"Polygon {i}: length={p.length}, area={p.area}, num_coords={len(p.exterior.coords)}")

print("Computing centerlines...")
densify_factor = -1.0
t0 = time.time()
centerlines = _compute_voronoi_centerlines(polygons, densify_factor=densify_factor, min_branch_factor=0.0)
print(f"Centerlines computed in {time.time()-t0:.2f}s")

from bezier_curves.medial_axis import _multilinestring_to_sknw_graph
from bezier_curves.bezier import _prune_skeleton_spurs, _build_skeleton_chains, _estimate_tangent, _fit_cubic_single
from shapely.geometry import LineString

print("Building graph...")
t0 = time.time()
graph = _multilinestring_to_sknw_graph(centerlines)
print(f"Graph built in {time.time()-t0:.2f}s with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

print("Pruning spurs...")
t0 = time.time()
_prune_skeleton_spurs(graph, threshold_length=max(8.0, 12.0 * np.hypot(h, img.shape[1])/1000.0))
print(f"Pruned in {time.time()-t0:.2f}s. Graph now has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

print("Building chains...")
t0 = time.time()
chains = _build_skeleton_chains(graph, follow_junction_continuation=True, junction_min_alignment=-0.15)
print(f"Chains built in {time.time()-t0:.2f}s. {len(chains)} chains found.")

print("Fitting Bezier paths...")
t0 = time.time()
for i, chain in enumerate(chains):
    pts = chain.points
    if len(pts) < 2: continue
    
    if len(pts) > 10:
        ls = LineString(pts).simplify(0.5, preserve_topology=True)
        if ls.is_simple:
            pts = np.array(ls.coords)
    
    if len(pts) < 2: continue
    
    lt = _estimate_tangent(pts, "start", lookahead=8)
    rt = _estimate_tangent(pts, "end", lookahead=8)
    # Using max_error=2.0 -> max_error**2 = 4.0
    cps = _fit_cubic_single(pts, lt, rt, 4.0, straightness_scale=0.75)
print(f"Bezier fitted in {time.time()-t0:.2f}s")

