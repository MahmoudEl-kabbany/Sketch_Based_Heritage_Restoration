import cv2
import numpy as np

# Load original color image and grayscale version
img = cv2.imread('test_images/aew.webp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

window_name = 'Dual Tuner - Press Q to Save'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ==========================================
# 1. INTERACTIVE UPDATE FUNCTION
# ==========================================
def update_ui(val):
    # Fetch current positions of both trackbars
    blur_val = cv2.getTrackbarPos('Blur Level', window_name)
    min_size = cv2.getTrackbarPos('Min Pixel Area', window_name)
    
    # Convert slider value (0-10) to an odd kernel size (1, 3, 5 ... 21)
    ksize = blur_val * 2 + 1 
    
    # Apply dynamic Median Blur
    blurred = cv2.medianBlur(gray, ksize)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Dilate slightly to connect broken lines
    kernel = np.ones((2, 2), np.uint8)
    closed_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Calculate connected components on the fly
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_edges, connectivity=8)
    
    # Filter out tiny dust lines
    clean_lines = np.zeros_like(gray)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_lines[labels == i] = 255
            
    # Convert back to 3-channel and stitch next to the original
    clean_lines_color = cv2.cvtColor(clean_lines, cv2.COLOR_GRAY2BGR)
    combined_view = cv2.hconcat([img, clean_lines_color])
    
    cv2.imshow(window_name, combined_view)

# ==========================================
# 2. CREATE TRACKBARS
# ==========================================
# Slider for Blur (maps to kernel sizes 1 to 21)
cv2.createTrackbar('Blur Level', window_name, 3, 10, update_ui)

# Slider for Minimum Line Size
cv2.createTrackbar('Min Pixel Area', window_name, 50, 500, update_ui)

# Trigger the UI once to render the first frame
update_ui(0)

# ==========================================
# 3. RUN & SAVE LOOP
# ==========================================
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# Fetch final values
final_blur = cv2.getTrackbarPos('Blur Level', window_name) * 2 + 1
final_size = cv2.getTrackbarPos('Min Pixel Area', window_name)

# Re-generate the final pristine canvas
final_blurred = cv2.medianBlur(gray, final_blur)
final_edges = cv2.Canny(final_blurred, 30, 100)
final_closed = cv2.dilate(final_edges, np.ones((2, 2), np.uint8), iterations=1)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_closed, connectivity=8)

final_canvas = np.zeros_like(gray)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= final_size:
        final_canvas[labels == i] = 255
        
# Save just the black-and-white output
cv2.imwrite("contour_outputs/final_output.jpg", final_canvas)
print(f"Saved successfully!\nOptimized Blur Kernel: {final_blur}\nOptimized Minimum Area: {final_size}")

cv2.destroyAllWindows()