"""Binary sketch renderer for quantitative evaluation.

Renders restoration results as clean black-on-white grayscale images,
matching the format of original sketch images for PSNR/SSIM comparison.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

from bezier_curves.bezier import BezierPath


# ═══════════════════════════════════════════════════════════════════════════
# Line width estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_line_width(
    img_gray: np.ndarray,
    fallback: float = 2.0,
    clamp_min: float = 1.0,
    clamp_max: float = 50.0,
) -> float:
    """Estimate the average line width of a black-on-white sketch image.

    Uses skeleton-based distance transform: the distance at each skeleton
    pixel equals half the local line width.

    Args:
        img_gray: Grayscale image (H, W), uint8.
        fallback: Default width if estimation fails.
        clamp_min: Minimum allowed width.
        clamp_max: Maximum allowed width.

    Returns:
        Estimated line width in pixels.
    """
    if img_gray is None or img_gray.size == 0:
        return fallback

    try:
        # Binarize with Otsu — lines (black) become foreground (255)
        _, binary = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Distance transform on the foreground
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Skeletonize to find medial axis
        skel = skeletonize(binary // 255).astype(np.uint8)

        # Sample distance values along the skeleton
        skel_distances = dist[skel > 0]

        if len(skel_distances) == 0:
            return fallback

        # Median distance at skeleton = half the line width
        avg_half_width = float(np.median(skel_distances))
        avg_width = 2.0 * avg_half_width

        # Clamp to reasonable range
        return float(np.clip(avg_width, clamp_min, clamp_max))

    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# Sketch rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_restored_sketch(
    restored_paths: list[BezierPath],
    image_shape: tuple[int, int],
    line_width: Optional[float] = None,
    original_image: Optional[np.ndarray] = None,
    default_line_width: float = 2.0,
    pts_per_segment: int = 60,
) -> np.ndarray:
    """Render restored Bezier paths as a clean black-on-white sketch image.

    Args:
        restored_paths: List of BezierPath objects from restoration.
        image_shape: (height, width) of the output image.
        line_width: Explicit line width in pixels. If None, auto-estimated
            from original_image or falls back to default_line_width.
        original_image: Optional grayscale original image for auto line
            width estimation.
        default_line_width: Fallback line width if auto-estimation fails
            and no explicit width is given.
        pts_per_segment: Number of sample points per Bezier segment for
            polyline approximation.

    Returns:
        Grayscale uint8 image (H, W) — white background (255) with
        black lines (0), anti-aliased.
    """
    h, w = image_shape[:2]

    # Determine line width
    if line_width is not None:
        thickness = max(1, int(round(line_width)))
    elif original_image is not None:
        estimated = estimate_line_width(original_image, fallback=default_line_width)
        thickness = max(1, int(round(estimated)))
    else:
        thickness = max(1, int(round(default_line_width)))

    # White canvas
    canvas = np.full((h, w), 255, dtype=np.uint8)

    # Draw each path
    for path in restored_paths:
        if not path.segments:
            continue

        pts = path.sample(pts_per_segment=pts_per_segment)
        if len(pts) < 2:
            continue

        # Convert to integer pixel coordinates
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))

        # Draw anti-aliased black polyline
        cv2.polylines(
            canvas,
            [pts_int],
            isClosed=path.is_closed,
            color=0,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# Bridge-on-damaged overlay
# ═══════════════════════════════════════════════════════════════════════════

def render_bridges_on_damaged(
    restored_paths: list[BezierPath],
    damaged_image: np.ndarray,
    line_width: Optional[float] = None,
    original_image: Optional[np.ndarray] = None,
    default_line_width: float = 2.0,
    pts_per_segment: int = 60,
) -> np.ndarray:
    """Draw only restoration bridges onto a copy of the damaged image.

    Only segments with source_type 'bridge' or 'efd_closure' are drawn.
    The damaged image pixels remain untouched everywhere else, making
    PSNR/SSIM comparisons against the original much more meaningful.

    Args:
        restored_paths: List of BezierPath objects from restoration.
        damaged_image: Grayscale damaged image (H, W), uint8.
        line_width: Explicit line width in pixels. If None, auto-estimated
            from original_image or falls back to default_line_width.
        original_image: Optional grayscale original image for auto line
            width estimation.
        default_line_width: Fallback line width if auto-estimation fails
            and no explicit width is given.
        pts_per_segment: Number of sample points per Bezier segment for
            polyline approximation.

    Returns:
        Grayscale uint8 image (H, W) — damaged image with black bridge
        lines drawn on top, anti-aliased.
    """
    # Start from a copy of the damaged image
    canvas = damaged_image.copy()

    # Determine line width
    if line_width is not None:
        thickness = max(1, int(round(line_width)))
    elif original_image is not None:
        estimated = estimate_line_width(original_image, fallback=default_line_width)
        thickness = max(1, int(round(estimated)))
    else:
        thickness = max(1, int(round(default_line_width)))

    # Draw only bridge and efd_closure segments
    for path in restored_paths:
        if not path.segments:
            continue

        for seg in path.segments:
            if seg.source_type not in ("bridge", "efd_closure"):
                continue

            pts = seg.sample(n=pts_per_segment)
            if len(pts) < 2:
                continue

            pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                canvas,
                [pts_int],
                isClosed=False,
                color=0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    return canvas
