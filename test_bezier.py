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
    follow_junction_continuation: bool = True,
    junction_min_alignment: float = -0.30,
    junction_min_score_margin: float = 0.04,
    enable_path_merging: bool = True,
    path_merge_gap_threshold: float = 18.0,
    path_merge_min_alignment: float = -0.20,
    path_merge_min_consistency: float = -0.25,
    path_merge_max_score: float = 20.0,
    path_merge_create_bridge: bool = True,
    save_visualization: bool = True,
) -> List[BezierPath]:
    """
    Extract Bezier curves from a raster image (PNG, JPG, etc.).

    Args:
        image_path: Path to the input image file
        use_skeleton: If True, use skeleton fitting; else use contour fitting
        min_contour_area: Minimum contour area to process (contour mode only)
        max_error: Maximum fitting error threshold for Bezier approximation
        follow_junction_continuation: Continue through junction using best direction
        junction_min_alignment: Minimum dot-product alignment for junction continuation
        junction_min_score_margin: Minimum score margin to accept junction continuation
        enable_path_merging: Enable post-fit path concatenation
        path_merge_gap_threshold: Maximum endpoint distance allowed for path merging
        path_merge_min_alignment: Minimum endpoint-vs-gap alignment for merge candidate
        path_merge_min_consistency: Minimum directional continuity between connected paths
        path_merge_max_score: Maximum allowed merge score (lower is stricter)
        path_merge_create_bridge: Insert short cubic bridge for small residual endpoint gaps
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
        paths, _adjacency = fit_from_image_skeleton(
            image_path,
            max_error=max_error,
            follow_junction_continuation=follow_junction_continuation,
            junction_min_alignment=junction_min_alignment,
            junction_min_score_margin=junction_min_score_margin,
            enable_path_merging=enable_path_merging,
            path_merge_gap_threshold=path_merge_gap_threshold,
            path_merge_min_alignment=path_merge_min_alignment,
            path_merge_min_consistency=path_merge_min_consistency,
            path_merge_max_score=path_merge_max_score,
            path_merge_create_bridge=path_merge_create_bridge,
        )
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
    test_image = "test_images/restoration_test.png"
    # if os.path.exists(test_image):
    #     print("▶ Test 1: Contour mode fitting")
    #     paths = process_image(test_image, use_skeleton=False)
    # else:
    #     print(f"⚠ Test image not found: {test_image}")

    # Test 2: Skeleton mode
    if os.path.exists(test_image):
        print("\n▶ Test 2: Skeleton mode fitting")
        paths = process_image(test_image, use_skeleton=True)
    else:
        print(f"⚠ Test image not found for skeleton test: {test_image}")
