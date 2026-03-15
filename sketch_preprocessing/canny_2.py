import cv2
import numpy as np

# Load the original image in color for the side-by-side display
img_color = cv2.imread("test_images/khepshef.jpg")
# Create the grayscale version for your processing pipeline
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

window_name = 'Edge Tuner - Press Q to Save'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ==========================================
# 1. INTERACTIVE UPDATE FUNCTION
# ==========================================
def update_ui(val):
    # Fetch current positions of the trackbars
    blur_val = cv2.getTrackbarPos('Blur Level', window_name)
    t1 = cv2.getTrackbarPos('Canny Min', window_name)
    t2 = cv2.getTrackbarPos('Canny Max', window_name)
    
    # Convert blur slider (0-15) to an odd number (1, 3, 5, 7...) for the kernel
    ksize = blur_val * 2 + 1 
    
    # --- YOUR EXACT PIPELINE ---
    blurred = cv2.medianBlur(img_gray, ksize)
    edges = cv2.Canny(blurred, threshold1=t1, threshold2=t2)
    
    kernel = np.ones((2, 2), np.uint8)
    clean_lines = cv2.dilate(edges, kernel, iterations=1)
    # ---------------------------
    
    # Convert 1-channel output to 3-channel so it can stitch with the color original
    clean_lines_color = cv2.cvtColor(clean_lines, cv2.COLOR_GRAY2BGR)
    
    # Stitch the images side-by-side
    combined_view = cv2.hconcat([img_color, clean_lines_color])
    
    cv2.imshow(window_name, combined_view)

# ==========================================
# 2. CREATE TRACKBARS
# ==========================================
# Default blur is set to 3 (which calculates to a kernel size of 7, matching your code)
cv2.createTrackbar('Blur Level', window_name, 3, 15, update_ui) 

# Default Canny thresholds set to your code's 30 and 100
cv2.createTrackbar('Canny Min', window_name, 30, 255, update_ui)
cv2.createTrackbar('Canny Max', window_name, 100, 255, update_ui)

# Trigger the UI once to render the first frame
update_ui(0)

# ==========================================
# 3. RUN & SAVE LOOP
# ==========================================
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'): # ESC or 'q' to quit and save
        break

# Fetch the final perfectly tuned values
final_blur = cv2.getTrackbarPos('Blur Level', window_name) * 2 + 1
final_t1 = cv2.getTrackbarPos('Canny Min', window_name)
final_t2 = cv2.getTrackbarPos('Canny Max', window_name)

# Process the final image one last time to save just the outline
final_blurred = cv2.medianBlur(img_gray, final_blur)
final_edges = cv2.Canny(final_blurred, final_t1, final_t2)
final_clean_lines = cv2.dilate(final_edges, np.ones((2, 2), np.uint8), iterations=1)

cv2.imwrite("contour_outputs/final_output.jpg", final_clean_lines)

print(f"Saved successfully!")
print(f"Optimized Blur Kernel: {final_blur}")
print(f"Optimized Canny Thresholds: {final_t1} and {final_t2}")

cv2.destroyAllWindows()