import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

from restoration.pipeline import restore

image_path = 'test_images/damaged_sketches/restoration_test_damaged_big.png'
print(f"Starting restoration for {image_path}")
t0 = time.time()
res = restore(image_path)
print(f"Total time: {time.time()-t0:.2f}s")
