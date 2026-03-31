import os
import sys
import logging
from pipeline import restore, visualise_result, PipelineConfig

logging.basicConfig(level=logging.INFO)

image_path = os.path.join("test_images", "restoration_test_damaged_big.png")
output_dir = "restoration_output_test"

if not os.path.exists(image_path):
    print(f"ERROR: Image not found at {image_path}")
    sys.exit(1)

print(f"Running restoration on: {image_path}")

cfg = PipelineConfig(
    output_dir=output_dir,
    use_skeleton=True,
    efd_order=40
)

result, report = restore(image_path, cfg)
visualise_result(result, report, image_path, output_dir)

print("\n[OK] Restoration complete. Check restoration_output_test/restoration_visualisation.png")
