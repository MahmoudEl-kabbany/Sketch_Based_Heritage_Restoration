import os

import cv2
import numpy as np


image_path = "test_images/khepshef.jpg"
output_dir = "contour_outputs"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# Stage 1 — bilateral filter to suppress surface texture while preserving boundaries
smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Stage 2 — use luminance from LAB and boost local contrast
lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
l_channel = lab[:, :, 0]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(l_channel)

# Stage 3 — threshold to a filled foreground mask instead of detecting thin edges
binary = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    8,
)

# Stage 4 — close gaps, then remove isolated specks
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

# Stage 5 — remove tiny connected components from the binary mask
min_component_area = 150
n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
clean_mask = np.zeros_like(opened)
for i in range(1, n_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
        clean_mask[labels == i] = 255

# Stage 6 — extract only sufficiently large external contours
min_contour_area = 200
contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

# Stage 7 — simplify slightly and draw closed contours
contour_mask = np.zeros_like(clean_mask)
overlay = img.copy()
for contour in filtered_contours:
    epsilon = 0.003 * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    cv2.drawContours(contour_mask, [simplified], -1, 255, 1)
    cv2.drawContours(overlay, [simplified], -1, (0, 255, 0), 2)

cv2.imwrite(os.path.join(output_dir, "01_enhanced_l_channel.png"), enhanced)
cv2.imwrite(os.path.join(output_dir, "02_binary_mask.png"), binary)
cv2.imwrite(os.path.join(output_dir, "03_closed_mask.png"), closed)
cv2.imwrite(os.path.join(output_dir, "04_clean_mask.png"), clean_mask)
cv2.imwrite(os.path.join(output_dir, "05_contours_mask.png"), contour_mask)
cv2.imwrite(os.path.join(output_dir, "06_contours_overlay.png"), overlay)