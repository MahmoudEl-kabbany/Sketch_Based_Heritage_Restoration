"""Phase 1 — Extraction.

Skeleton-based Bezier path extraction with endpoint tangents, curvatures,
and EFD coefficient extraction.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bezier_curves.bezier import (
    BezierPath,
    BezierSegment,
    fit_from_image_skeleton,
)
from eliptic_fourier_descriptors.efd import (
    reconstruct_contour_efd,
    compute_efd_features,
)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EndpointInfo:
    """Geometric descriptor for one end of a BezierPath."""

    endpoint_id: int        # stable endpoint identity within one extraction
    path_index: int          # index into ExtractionResult.paths
    end: str                 # "start" | "end"
    position: np.ndarray     # (x, y)
    tangent: np.ndarray      # unit tangent pointing *outward* from the path
    curvature: float         # unsigned curvature κ at this endpoint


@dataclass
class ExtractionResult:
    """Everything extracted from one image."""

    paths: List[BezierPath]
    endpoints: List[EndpointInfo]
    efd_contours: List[dict]       # per-contour EFD data
    image_shape: Tuple[int, int]   # (h, w)
    diagonal: float                # sqrt(h² + w²)


# ═══════════════════════════════════════════════════════════════════════════
# Tangent / curvature from Bezier control points
# ═══════════════════════════════════════════════════════════════════════════

def _bezier_derivative_at(cp: np.ndarray, t: float) -> np.ndarray:
    """First derivative of cubic Bezier at parameter *t*.

    cp: (4, 2) control points.
    """
    u = 1.0 - t
    return (
        3.0 * u * u * (cp[1] - cp[0])
        + 6.0 * u * t * (cp[2] - cp[1])
        + 3.0 * t * t * (cp[3] - cp[2])
    )


def _bezier_second_derivative_at(cp: np.ndarray, t: float) -> np.ndarray:
    """Second derivative of cubic Bezier at parameter *t*."""
    u = 1.0 - t
    return (
        6.0 * u * (cp[2] - 2.0 * cp[1] + cp[0])
        + 6.0 * t * (cp[3] - 2.0 * cp[2] + cp[1])
    )


def _curvature_at(cp: np.ndarray, t: float) -> float:
    """Unsigned curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)."""
    d1 = _bezier_derivative_at(cp, t)
    d2 = _bezier_second_derivative_at(cp, t)
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    speed_sq = float(d1[0] ** 2 + d1[1] ** 2)
    if speed_sq < 1e-24:
        return 0.0
    return abs(cross) / (speed_sq ** 1.5)


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint extraction
# ═══════════════════════════════════════════════════════════════════════════

def _compute_endpoint_tangent(path: BezierPath, end: str) -> np.ndarray:
    """Unit tangent pointing *outward* from the path at the given end."""
    if end == "start":
        cp = path.segments[0].control_points
        tangent = _bezier_derivative_at(cp, 0.0)
        return _safe_normalize(-tangent)  # negate: outward from path start
    else:
        cp = path.segments[-1].control_points
        tangent = _bezier_derivative_at(cp, 1.0)
        return _safe_normalize(tangent)   # already points outward


def _compute_endpoint_curvature(path: BezierPath, end: str) -> float:
    """Unsigned curvature at the given end of the path."""
    if end == "start":
        cp = path.segments[0].control_points
        return _curvature_at(cp, 0.0)
    else:
        cp = path.segments[-1].control_points
        return _curvature_at(cp, 1.0)


def _extract_endpoints(paths: List[BezierPath]) -> List[EndpointInfo]:
    """Build EndpointInfo objects for every open path's start and end."""
    endpoints: List[EndpointInfo] = []
    next_endpoint_id = 0
    for idx, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue
        for end in ("start", "end"):
            pos = (
                path.segments[0].control_points[0]
                if end == "start"
                else path.segments[-1].control_points[3]
            )
            endpoints.append(EndpointInfo(
                endpoint_id=next_endpoint_id,
                path_index=idx,
                end=end,
                position=pos.astype(np.float64),
                tangent=_compute_endpoint_tangent(path, end),
                curvature=_compute_endpoint_curvature(path, end),
            ))
            next_endpoint_id += 1
    return endpoints


# ═══════════════════════════════════════════════════════════════════════════
# EFD extraction (wraps existing module)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_efd_contours(
    image_path: str,
    min_contour_area: float = 100.0,
    order: int = 10,
) -> List[dict]:
    """Extract EFD coefficients for closed contours in the image."""
    import pyefd

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    results: List[dict] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        coords = np.squeeze(cnt)
        if coords.ndim != 2 or len(coords) < 5:
            continue
        try:
            coeffs = pyefd.elliptic_fourier_descriptors(
                coords, order=order, normalize=False,
            )
            a0, c0 = pyefd.calculate_dc_coefficients(coords)
            results.append({
                "contour": coords,
                "coeffs": coeffs,
                "locus": (a0, c0),
                "order": order,
            })
        except Exception:
            continue
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_paths(
    image_path: str,
    max_error: float = 2.0,
    spur_threshold: float = 12.0,
    merge_radius: float = 5.0,
) -> ExtractionResult:
    """Extract Bezier paths, endpoints, and EFD data from a raster image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    diagonal = float(np.hypot(h, w))

    # Bezier path extraction via skeleton
    paths, _ = fit_from_image_skeleton(
        image_path,
        max_error=max_error,
        spur_threshold=spur_threshold,
        merge_radius=merge_radius,
    )

    # Endpoint extraction
    endpoints = _extract_endpoints(paths)

    # EFD extraction
    efd_contours = _extract_efd_contours(image_path)

    return ExtractionResult(
        paths=paths,
        endpoints=endpoints,
        efd_contours=efd_contours,
        image_shape=(h, w),
        diagonal=diagonal,
    )
