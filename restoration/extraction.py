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
)
from bezier_curves.medial_axis import fit_from_image_geometric
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
    tangent_confidence: float = 1.0  # confidence in endpoint tangent direction


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


def _dedupe_polyline(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Remove near-duplicate consecutive points from a sampled polyline."""
    if len(points) <= 1:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = diffs > eps
    return points[keep]


def _strip_sway_prefix(points: np.ndarray, end: str, max_strip_fraction: float = 0.15) -> np.ndarray:
    """Walk inward from the endpoint and drop high-curvature sway segments."""
    if len(points) < 10:
        return points

    n_strip_max = int(round(len(points) * max_strip_fraction))
    if n_strip_max < 2:
        return points

    diffs = np.diff(points, axis=0)
    strip_count = 0
    
    if end == "start":
        for i in range(n_strip_max):
            a = diffs[i]
            b = diffs[i + 1]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            if angle > 25.0:
                strip_count = i + 1
            elif i > strip_count + 1:
                break
        if strip_count > 0:
            return points[strip_count:]
    else:
        for i in range(n_strip_max):
            idx = len(diffs) - 1 - i
            a = diffs[idx - 1]
            b = diffs[idx]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            if angle > 25.0:
                strip_count = i + 1
            elif i > strip_count + 1:
                break
        if strip_count > 0:
            return points[:-strip_count]

    return points


def _multiscale_context_tangent(
    sampled: np.ndarray, end: str, outward_ref: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Run PCA at 4 different window sizes and return circular mean of directions."""
    fractions = [0.10, 0.15, 0.20, 0.25]
    directions = []
    weights = []
    n_pts = len(sampled)
    
    for frac in fractions:
        window = int(round(n_pts * frac))
        window = max(8, min(28, max(2, n_pts - 1), window))
        
        if end == "start":
            local = sampled[: window + 1]
        else:
            local = sampled[-(window + 1):]
            
        centered = local - np.mean(local, axis=0)
        cov = centered.T @ centered
        vals, vecs = np.linalg.eigh(cov)
        idx_max = int(np.argmax(vals))
        principal = vecs[:, idx_max]
        
        if float(np.dot(principal, outward_ref)) < 0.0:
            principal = -principal
            
        var_sum = float(np.sum(vals))
        linearity = float(vals[idx_max] / (var_sum + 1e-12))
        directional = max(0.0, float(np.dot(principal, outward_ref)))
        conf = float(np.clip(0.5 * linearity + 0.5 * directional, 0.0, 1.0))
        
        directions.append(principal)
        weights.append(conf)
        
    mean_dir = np.zeros(2, dtype=np.float64)
    for d, w in zip(directions, weights):
        mean_dir += d * w
        
    norm = np.linalg.norm(mean_dir)
    if norm < 1e-12:
        return outward_ref, 0.0
        
    final_tangent = mean_dir / norm
    R = norm / sum(weights) if sum(weights) > 0 else 0.0
    final_conf = float(np.clip(R * np.mean(weights), 0.0, 1.0))
    
    return final_tangent, final_conf


def _estimate_endpoint_context_tangent(
    path: BezierPath,
    end: str,
) -> Tuple[np.ndarray, float]:
    """Estimate endpoint tangent from local sampled context and return confidence."""
    sampled = path.sample(pts_per_segment=24)
    sampled = _dedupe_polyline(sampled)
    # Sway stripping enabled to avoid tangent misalignments
    sampled = _strip_sway_prefix(sampled, end)
    
    if len(sampled) < 3:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    if end == "start":
        path_dir = sampled[min(2, len(sampled)-1)] - sampled[0]
        outward_ref = _safe_normalize(-path_dir)
    else:
        path_dir = sampled[-1] - sampled[max(0, len(sampled)-3)]
        outward_ref = _safe_normalize(path_dir)

    return _multiscale_context_tangent(sampled, end, outward_ref)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint extraction
# ═══════════════════════════════════════════════════════════════════════════

def _compute_endpoint_tangent(path: BezierPath, end: str) -> Tuple[np.ndarray, float]:
    """Outward endpoint tangent blended with local context and confidence."""
    if end == "start":
        cp = path.segments[0].control_points
        derivative_tangent = _safe_normalize(-_bezier_derivative_at(cp, 0.0))
    else:
        cp = path.segments[-1].control_points
        derivative_tangent = _safe_normalize(_bezier_derivative_at(cp, 1.0))

    context_tangent, confidence = _estimate_endpoint_context_tangent(path, end)
    
    if confidence < 0.40:
        blend = float(np.clip(confidence, 0.05, 0.35))
    else:
        blend = float(np.clip(confidence, 0.40, 0.88))
        
    tangent = _safe_normalize((1.0 - blend) * derivative_tangent + blend * context_tangent)
    
    if end == "start":
        interior_dir = _safe_normalize(path.segments[0].control_points[3] - path.segments[0].control_points[0])
    else:
        interior_dir = _safe_normalize(path.segments[-1].control_points[0] - path.segments[-1].control_points[3])
        
    if float(np.dot(tangent, interior_dir)) > 0.50:
        tangent = -tangent
        confidence *= 0.5
        
    return tangent, confidence


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
            tangent, tangent_confidence = _compute_endpoint_tangent(path, end)
            endpoints.append(EndpointInfo(
                endpoint_id=next_endpoint_id,
                path_index=idx,
                end=end,
                position=pos.astype(np.float64),
                tangent=tangent,
                curvature=_compute_endpoint_curvature(path, end),
                tangent_confidence=tangent_confidence,
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
    
    # Scale-invariant thresholding: normalize relative to a 1000px diagonal baseline
    scale_factor = diagonal / 1000.0
    adaptive_spur_threshold = max(8.0, spur_threshold * scale_factor)
    adaptive_merge_radius = max(3.0, merge_radius * scale_factor)

    # Bezier path extraction via pure geometry (Voronoi MAT)
    paths, _ = fit_from_image_geometric(
        image_path,
        image_height=h,
        max_error=max_error,
        min_area=150.0,
        spur_threshold=adaptive_spur_threshold,
        merge_radius=adaptive_merge_radius,
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
