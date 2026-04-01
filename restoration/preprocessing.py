"""
R-0: Preprocessing Module
==========================
Binarisation, skeletonisation, and anti-fragmentation cleaning for
the restoration pipeline.

This module is the *restoration-specific* preprocessor.  It does NOT
duplicate the preprocessing already present in ``bezier.py`` or
``efd.py``.  Instead it produces a clean binary image and skeleton
that those modules can consume, plus metadata about stroke thickness
so the pipeline can choose between skeleton-based and contour-based
fitting.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PreprocessResult:
    """Immutable container for preprocessing outputs."""
    gray: np.ndarray           # Original grayscale image
    binary: np.ndarray         # Clean binary image (foreground = 255)
    skeleton: np.ndarray       # 1-pixel skeleton (0/255)
    is_thick_stroke: bool      # True when median stroke width > threshold
    median_stroke_width: float # Estimated median stroke width in pixels
    image_h: int
    image_w: int


def preprocess_for_restoration(
    image_path: str,
    *,
    noise_h: int = 10,
    min_component_fraction: float = 0.001,
    thick_stroke_threshold: float = 4.0,
    spur_length: int = 8,
) -> PreprocessResult:
    """Prepare a damaged sketch image for the restoration pipeline.

    Steps
    -----
    1. Load as grayscale.
    2. Denoise (fastNlMeansDenoising).
    3. Otsu binarisation (robust to varying backgrounds).
    4. Morphological close to heal micro-cracks.
    5. Connected-component filtering (remove tiny fragments).
    6. Skeletonise.
    7. Prune short spurs from the skeleton.
    8. Estimate stroke thickness.

    Parameters
    ----------
    image_path : str
        Path to the damaged image.
    noise_h : int
        Denoising strength.
    min_component_fraction : float
        Components smaller than this fraction of the largest component
        are removed.  Prevents false endpoints.
    thick_stroke_threshold : float
        If the median stroke width exceeds this, the image is flagged
        as thick-stroke (triggering contour-based fitting later).
    spur_length : int
        Maximum length of skeleton branches to prune.

    Returns
    -------
    PreprocessResult
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = gray.shape

    # ── 1. Denoise ──────────────────────────────────────────────────
    denoised = cv2.fastNlMeansDenoising(gray, h=noise_h,
                                         templateWindowSize=7,
                                         searchWindowSize=21)

    # ── 2. Binarise (Otsu) ──────────────────────────────────────────
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 3. Morphological close (heal micro-cracks) ──────────────────
    # Kernel size adaptive to image resolution
    k_size = max(3, int(min(h, w) * 0.005) | 1)  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── 4. Remove small connected components ────────────────────────
    binary = _remove_small_components(binary, min_component_fraction)

    # ── 5. Estimate stroke thickness (before skeletonisation) ───────
    median_sw = _estimate_stroke_width(binary)
    is_thick = median_sw > thick_stroke_threshold

    # ── 6. Skeletonise ──────────────────────────────────────────────
    skel_bool = skeletonize(binary > 0)
    skeleton = (skel_bool.astype(np.uint8)) * 255

    # ── 7. Prune short spurs ────────────────────────────────────────
    skeleton = _prune_spurs(skeleton, spur_length)

    return PreprocessResult(
        gray=gray,
        binary=binary,
        skeleton=skeleton,
        is_thick_stroke=is_thick,
        median_stroke_width=median_sw,
        image_h=h,
        image_w=w,
    )


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _remove_small_components(
    binary: np.ndarray,
    min_fraction: float,
) -> np.ndarray:
    """Remove connected components that are much smaller than the largest."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if n_labels <= 1:
        return binary

    # Areas of each component (label 0 = background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return binary

    max_area = areas.max()
    min_area = max(max_area * min_fraction, 10)  # absolute floor of 10px

    cleaned = np.zeros_like(binary)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Estimate the median stroke width using the distance transform.

    For each foreground pixel, the distance transform gives the distance
    to the nearest background pixel.  At the skeleton, this equals half
    the stroke width.  We approximate by taking the median of all
    nonzero distance-transform values.
    """
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    fg_dists = dt[binary > 0]
    if len(fg_dists) == 0:
        return 1.0
    return float(np.median(fg_dists)) * 2.0


def _prune_spurs(skeleton: np.ndarray, max_spur: int) -> np.ndarray:
    """Remove short terminal branches (spurs) from a skeleton.

    Iteratively removes endpoints whose branch length ≤ max_spur.
    This prevents false endpoints that would generate incorrect bridges.
    """
    skel = skeleton.copy()
    for _ in range(max_spur):
        # An endpoint is a foreground pixel with exactly one 8-neighbour
        endpoints = _find_endpoints(skel)
        if len(endpoints) == 0:
            break
        for y, x in endpoints:
            skel[y, x] = 0
    return skel


def _find_endpoints(skeleton: np.ndarray) -> list:
    """Find all endpoint pixels (exactly 1 foreground 8-neighbour)."""
    # Convolve with a 3×3 all-ones kernel to count neighbours
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    fg = (skeleton > 0).astype(np.uint8)
    neighbour_count = cv2.filter2D(fg, -1, kernel)
    # Endpoints: foreground pixels with exactly 1 neighbour
    ep_mask = (fg == 1) & (neighbour_count == 1)
    ys, xs = np.where(ep_mask)
    return list(zip(ys.tolist(), xs.tolist()))
