import cv2
import numpy as np

img_color = cv2.imread('test_images/khepshef.jpg')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

window_name = 'Adaptive Tuner - Press Q to Save'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def update_ui(val):
    # 1. Edge Preserving Filter (Smoothes texture, keeps edges sharp)
    # sigma_s controls spatial smoothing, sigma_r controls color smoothing
    smooth_val = cv2.getTrackbarPos('Smooth Textures', window_name)
    smoothed = cv2.edgePreservingFilter(img_color, flags=1, sigma_s=smooth_val, sigma_r=0.4)
    
    # Convert to grayscale for thresholding
    gray_smoothed = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding (Finds lines based on local contrast)
    # Block size must be an odd number (3, 5, 7, etc.)
    block_val = cv2.getTrackbarPos('Line Thickness', window_name)
    block_size = block_val * 2 + 3 
    
    # C is a constant subtracted from the mean. Higher C = less noise, but fainter lines
    c_val = cv2.getTrackbarPos('Noise Filter (C)', window_name)
    
    # Apply the threshold. It automatically outputs white lines on black if we invert it at the end
    lines = cv2.adaptiveThreshold(gray_smoothed, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 
                                  block_size, c_val)
    
    # Convert to 3-channel to stitch
    lines_color = cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR)
    combined_view = cv2.hconcat([img_color, lines_color])
    cv2.imshow(window_name, combined_view)


# Sliders
cv2.createTrackbar('Smooth Textures', window_name, 50, 100, update_ui)
cv2.createTrackbar('Line Thickness', window_name, 4, 15, update_ui) # Maps to block sizes 11, 13, 15...
cv2.createTrackbar('Noise Filter (C)', window_name, 8, 20, update_ui)

update_ui(0)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# Fetch final values and save
final_smooth = cv2.getTrackbarPos('Smooth Textures', window_name)
final_block = cv2.getTrackbarPos('Line Thickness', window_name) * 2 + 3
final_c = cv2.getTrackbarPos('Noise Filter (C)', window_name)

final_smoothed = cv2.edgePreservingFilter(img_color, flags=1, sigma_s=final_smooth, sigma_r=0.4)
final_gray = cv2.cvtColor(final_smoothed, cv2.COLOR_BGR2GRAY)
final_lines = cv2.adaptiveThreshold(final_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, final_block, final_c)

cv2.imwrite('contour_outputs/final_adaptive_lines.jpg', final_lines)
cv2.destroyAllWindows()