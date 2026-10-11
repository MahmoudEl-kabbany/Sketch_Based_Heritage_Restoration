"""Phase 1 — Extraction.

Skeleton-based Bezier path extraction with endpoint tangents and curvatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, fit_from_image_skeleton
from restoration.geometry import (
    bezier_derivative,
    endpoint_curvature,
    safe_normalize,
    visual_order_key,
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
    image_shape: Tuple[int, int]   # (h, w)
    diagonal: float                # sqrt(h² + w²)
    dist_map: Optional[np.ndarray] = None  # distance map from lines


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint context tangent
# ═══════════════════════════════════════════════════════════════════════════

def _dedupe_polyline(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Remove near-duplicate consecutive points from a sampled polyline."""
    if len(points) <= 1:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = diffs > eps
    return points[keep]


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

    if len(sampled) < 3:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    if end == "start":
        path_dir = sampled[min(2, len(sampled)-1)] - sampled[0]
        outward_ref = safe_normalize(-path_dir)
    else:
        path_dir = sampled[-1] - sampled[max(0, len(sampled)-3)]
        outward_ref = safe_normalize(path_dir)

    return _multiscale_context_tangent(sampled, end, outward_ref)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint extraction
# ═══════════════════════════════════════════════════════════════════════════

def _compute_endpoint_tangent(path: BezierPath, end: str) -> Tuple[np.ndarray, float]:
    """Outward endpoint tangent blended with local context and confidence."""
    if end == "start":
        cp = path.segments[0].control_points
        derivative_tangent = safe_normalize(-bezier_derivative(cp, 0.0))
    else:
        cp = path.segments[-1].control_points
        derivative_tangent = safe_normalize(bezier_derivative(cp, 1.0))

    context_tangent, confidence = _estimate_endpoint_context_tangent(path, end)
    
    if confidence < 0.40:
        blend = float(np.clip(confidence, 0.05, 0.35))
    else:
        blend = float(np.clip(confidence, 0.40, 0.88))
        
    tangent = safe_normalize((1.0 - blend) * derivative_tangent + blend * context_tangent)
    
    if end == "start":
        interior_dir = safe_normalize(path.segments[0].control_points[3] - path.segments[0].control_points[0])
    else:
        interior_dir = safe_normalize(path.segments[-1].control_points[0] - path.segments[-1].control_points[3])
        
    if float(np.dot(tangent, interior_dir)) > 0.50:
        tangent = -tangent
        confidence *= 0.5
        
    return tangent, confidence


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
                curvature=endpoint_curvature(path, end),
                tangent_confidence=tangent_confidence,
            ))
            next_endpoint_id += 1
    return endpoints


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_paths(
    image_path: str,
    max_error: float = 2.0,
    spur_threshold: float = 12.0,
    merge_radius: float = 5.0,
) -> ExtractionResult:
    """Extract Bezier paths, endpoints, and the line distance map from a raster image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    diagonal = float(np.hypot(h, w))

    # Bezier path extraction via skeleton
    paths = fit_from_image_skeleton(
        image_path,
        max_error=max_error,
        spur_threshold=spur_threshold,
        merge_radius=merge_radius,
    )

    # Ensure visual discovery order (top-down, left-right)
    paths.sort(key=visual_order_key)

    # Endpoint extraction
    endpoints = _extract_endpoints(paths)

    # Compute distance map from the original image (lines=0, background=255)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_map = cv2.distanceTransform((255 - binary).astype(np.uint8), cv2.DIST_L2, 5)

    return ExtractionResult(
        paths=paths,
        endpoints=endpoints,
        image_shape=(h, w),
        diagonal=diagonal,
        dist_map=dist_map,
    )
