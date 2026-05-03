import cv2
from add_gaps import apply_gaps, add_gaps_batch

# 1. Process an image directly (as a numpy array)
img = cv2.imread("test_images/geometric.png", cv2.IMREAD_GRAYSCALE)
# Apply 10 gaps of type 'crack'
processed_img = apply_gaps(img, num_gaps=30, gap_type='box')
cv2.imwrite("output.png", processed_img)

# # 2. Process a list of image paths
# image_paths = ["sketch1.png", "sketch2.png"]
# add_gaps_batch(image_paths, output_dir="results", num_gaps=15, gap_type='circle')
