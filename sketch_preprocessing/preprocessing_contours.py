import cv2
import numpy as np

# 1. Load image (grayscale)
img = cv2.imread("test_images/khepshef.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Optional: contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(img)

# 3. Slight blur
blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

# 4. Adaptive thresholding (better for uneven lighting)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# 5. Morphological cleaning
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)  # remove small specs

# 6. Find all contours (including internal/nested ones)
contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 7. Filter very small contours by area
min_area = 100  # increase this value to remove more small contours
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

# 8. Draw filtered contours
if filtered_contours:
    # Draw all detected contours on the original grayscale image converted to BGR
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, filtered_contours, -1, (0,255,0), 2)
    cv2.imwrite("contour_outputs/output_contours.jpg", result)
else:
    print("No contours found above the minimum area threshold.")