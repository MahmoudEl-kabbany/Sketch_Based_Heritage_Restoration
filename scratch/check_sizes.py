import cv2
import numpy as np
import time

def check_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        return
    print(f"{path}: shape={img.shape}")

check_image('test_images/damaged_sketches/restoration_test_damaged_big.png')
check_image('test_images/damaged_sketches/damaged_ankh.png')
check_image('test_images/damaged_sketches/damaged_star.png')
