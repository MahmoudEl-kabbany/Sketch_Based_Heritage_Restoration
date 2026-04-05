import cv2
from bezier_curves.bezier import fit_from_image_skeleton, _draw_paths_on_canvas
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('test_images/restoration_test.png')
paths, adj = fit_from_image_skeleton('test_images/restoration_test.png')
colors = [(np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255)) for _ in paths]
canvas = np.zeros_like(img)
canvas = _draw_paths_on_canvas(canvas, paths, colors, 50, False)

cv2.imwrite('test_images/debug_restoration.png', canvas)
print("Saved debug_restoration.png")
