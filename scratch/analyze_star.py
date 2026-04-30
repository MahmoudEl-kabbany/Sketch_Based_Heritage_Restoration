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
print(f"Graph initially has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
print(f"Deg 1 nodes: {list(degrees.values()).count(1)}, Deg 3 nodes: {list(degrees.values()).count(3)}")

edges_before = len(graph.edges())
_prune_skeleton_spurs(graph, threshold_length=12.0)
edges_after = len(graph.edges())

print(f"Graph after pruning has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
print(f"Pruned {edges_before - edges_after} edges.")
