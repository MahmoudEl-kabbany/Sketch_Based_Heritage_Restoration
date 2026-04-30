import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import networkx as nx
import numpy as np
from bezier_curves.medial_axis import _extract_vector_boundaries, _compute_voronoi_centerlines, _multilinestring_to_sknw_graph
from bezier_curves.bezier import _prune_skeleton_spurs
import cv2

image_path = 'test_images/damaged_sketches/damaged_ankh.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
h = img.shape[0]

print("Extracting boundaries...")
polygons = _extract_vector_boundaries(image_path, image_height=h, min_area=150.0)

print("Computing centerlines...")
densify_factor = max(3.0, h * 0.0025)
centerlines = _compute_voronoi_centerlines(polygons, densify_factor=densify_factor, min_branch_factor=0.0)

print("Building graph...")
graph = _multilinestring_to_sknw_graph(centerlines)

print(f"Graph initially has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
degrees = dict(graph.degree())
deg3_nodes = [n for n, d in degrees.items() if d == 3]
deg1_nodes = [n for n, d in degrees.items() if d == 1]
print(f"Deg 1 nodes: {len(deg1_nodes)}, Deg 3 nodes: {len(deg3_nodes)}")

# Let's see what happens during Phase 0 of _prune_skeleton_spurs
threshold_length = 12.0
is_multi = graph.is_multigraph()

y_branches_found = 0
for node, deg in degrees.items():
    if deg == 3:
        adj_edges = list(graph.edges(node, keys=True, data=True) if is_multi else graph.edges(node, data=True))
        deg1_edges = []
        other_edges = []
        for ae in adj_edges:
            other = ae[1] if ae[0] == node else ae[0]
            if degrees.get(other, 0) == 1:
                deg1_edges.append(ae)
            else:
                other_edges.append(ae)
        
        if len(deg1_edges) == 2 and len(other_edges) == 1:
            e1, e2 = deg1_edges[0], deg1_edges[1]
            n1 = e1[1] if e1[0] == node else e1[0]
            n2 = e2[1] if e2[0] == node else e2[0]
            p1 = np.array(graph.nodes[n1].get("o", [0.0, 0.0]))
            p2 = np.array(graph.nodes[n2].get("o", [0.0, 0.0]))
            d1 = np.linalg.norm(p1 - np.array(graph.nodes[node].get("o", [0.0, 0.0])))
            d2 = np.linalg.norm(p2 - np.array(graph.nodes[node].get("o", [0.0, 0.0])))
            print(f"Y-branch candidate at {node}: d1={d1:.2f}, d2={d2:.2f}")
            if d1 > threshold_length * 1.5 or d2 > threshold_length * 1.5:
                print("  -> Rejected due to length!")

print(f"Found {y_branches_found} valid Y-branches.")

_prune_skeleton_spurs(graph, threshold_length=12.0)
print(f"Graph after pruning has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
