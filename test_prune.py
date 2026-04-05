import cv2
import sknw
from skimage.morphology import skeletonize
import numpy as np

img = cv2.imread('test_images/restoration_test.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
skeleton = skeletonize(binary / 255.0).astype(np.uint8)
graph = sknw.build_sknw(skeleton)
print("Original graph edges:", len(graph.edges()))

# Morphological smoothing before skeletonization
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
skeleton_smoothed = skeletonize(smoothed / 255.0).astype(np.uint8)
graph_smoothed = sknw.build_sknw(skeleton_smoothed)
print("Smoothed graph edges:", len(graph_smoothed.edges()))

# Pruning graph directly
edges_to_remove = []
for s, e in graph.edges():
    pts = graph[s][e]['pts']
    if len(pts) < 10: # threshold for spur length
        edges_to_remove.append((s, e))
print("Short edges:", len(edges_to_remove))

