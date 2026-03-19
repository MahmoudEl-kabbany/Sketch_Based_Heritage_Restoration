"""Bezier curve extraction and visualization.

Main function: process_image(image_path, use_skeleton=False)
  Returns: List[BezierPath] and saves visualization
"""

from bezier_curves.bezier import (
    fit_from_image,
    fit_from_image_skeleton,
    visualize_paths,
    BezierPath,
)
from typing import List
import os


OUTPUT = os.path.join("bezier_curves", "outputs")
os.makedirs(OUTPUT, exist_ok=True)


def process_image(
    image_path: str,
    use_skeleton: bool = False,
    min_contour_area: float = 100.0,
    max_error: float = 5.0,
    save_visualization: bool = True,
) -> List[BezierPath]:
    """
    Extract Bezier curves from a raster image (PNG, JPG, etc.).

    Args:
        image_path: Path to the input image file
        use_skeleton: If True, use skeleton fitting; else use contour fitting
        min_contour_area: Minimum contour area to process (contour mode only)
        max_error: Maximum fitting error threshold for Bezier approximation
        save_visualization: If True, save a visualization PNG to outputs folder

    Returns:
        List[BezierPath]: Fitted Bezier paths extracted from the image

    Example:
        >>> paths = process_image("test_images/sketch.jpg")
        >>> print(f"Found {len(paths)} paths")
        >>> for path in paths:
        ...     print(f"  Path with {path.num_segments} segments")
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Extract Bezier curves
    if use_skeleton:
        paths = fit_from_image_skeleton(image_path, max_error=max_error)
        mode = "skeleton"
    else:
        paths = fit_from_image(
            image_path,
            min_contour_area=min_contour_area,
            max_error=max_error,
        )
        mode = "contour"

    # Print summary
    total_segs = sum(p.num_segments for p in paths)
    print(f"\n{'═' * 60}")
    print(f"  {image_path} ({mode} mode)")
    print(f"  Paths: {len(paths)}  |  Total segments: {total_segs}")
    for i, p in enumerate(paths):
        status = "closed" if p.is_closed else "open"
        print(f"    Path {i}: {p.num_segments} segments ({status})")
    print(f"{'═' * 60}\n")

    # Visualize and save
    if save_visualization:
        name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(OUTPUT, f"{name}_bezier_vis.png")
        visualize_paths(
            paths,
            title=f"Bezier Fit — {name} ({mode})",
            save_path=vis_path,
            show_controls=True,
            image_path=image_path,
            overview=True,
        )

    return paths


if __name__ == "__main__":
    # Example usage
    print("Running bezier extraction tests...\n")

    # Test 1: Contour mode
    test_image = "test_images/bolt.png"
    if os.path.exists(test_image):
        print("▶ Test 1: Contour mode fitting")
        paths = process_image(test_image, use_skeleton=False)
    else:
        print(f"⚠ Test image not found: {test_image}")

    # Test 2: Skeleton mode
    # if os.path.exists(test_image):
    #     print("\n▶ Test 2: Skeleton mode fitting")
    #     paths = process_image(test_image, use_skeleton=True)
    # else:
    #     print(f"⚠ Test image not found for skeleton test: {test_image}")
