import os
import glob
from typing import Union, List
from restoration.pipeline import restore_image

def process_images(input_paths: Union[str, List[str]], output_dir: str = "outputs"):
    """
    Run pipeline.py on a single image, a list of images, or a directory.
    """
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif"}
    
    # If a single string is passed, wrap it in a list to process uniformly
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    images_to_process = []
    
    for path in input_paths:
        if not os.path.exists(path):
            print(f"Warning: Path '{path}' does not exist.")
            continue
            
        if os.path.isfile(path):
            images_to_process.append(path)
        elif os.path.isdir(path):
            print(f"Scanning directory '{path}' for images...")
            for ext in valid_extensions:
                # Add glob patterns for both lowercase and uppercase extensions
                images_to_process.extend(glob.glob(os.path.join(path, f"*{ext}")))
                images_to_process.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))

    if not images_to_process:
        print("No valid images found to process.")
        return

    print(f"Found {len(images_to_process)} image(s) to process.")
    
    for img_path in images_to_process:
        print(f"\n---> Processing: {img_path}")
        try:
            restore_image(img_path, output_dir=output_dir)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    # =========================================================================
    # EDIT THESE VARIABLES TO RUN IN YOUR IDE
    # =========================================================================
    
    # You can provide a single image path, a directory of images, or a list 
    # containing multiple files/directories.
    
    # Example 1: Single Image
    # INPUTS = "test_images/damaged_shape.png"
    
    # Example 2: Directory
    # INPUTS = "test_images"
    
    # Example 3: Multiple Images / Mixed
    INPUTS = [
        "test_images/damaged_oval.png",
        "test_images/restoration_test.png",
        "test_images/damaged_bolt.png",
        "test_images/damaged_shape.png",
        "test_images/restoration_small_gaps.png",
        "test_images/restoration_test_damaged.png",
        "test_images/restoration_test_damaged_big.png"
    ]
    
    # Where you want the results saved
    OUTPUT_DIR = "restoration_outputs"
    
    # Run the processing
    process_images(INPUTS, OUTPUT_DIR)

