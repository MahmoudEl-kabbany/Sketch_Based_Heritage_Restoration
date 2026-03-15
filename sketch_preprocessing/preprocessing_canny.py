import cv2

# 1. Load the image directly in grayscale
img = cv2.imread("test_images/khepshef.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Apply a Gaussian Blur to reduce high-frequency noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Apply Otsu's Thresholding to binarize the image (flattening internal gradients)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Apply Morphological Closing to fill microscopic holes and solidify the mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
solid_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 5. Pass the mathematically flattened mask to the Canny Edge Detector
edges = cv2.Canny(solid_mask, 100, 200)

# Save the resulting black-and-white silhouette outline
cv2.imwrite("contour_outputs/output_contours.jpg", edges)