"""
Bezier Curve Extraction Module
==============================
Contours (OpenCV Nx1x2 arrays) → corner detection → Schneider fit → cubic Bezier

All segments are normalized to cubic Bezier for uniformity.

Output:
  - Python objects (BezierSegment / BezierPath with control points)
  - Visualization on blank canvas with control point overlays
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
import sknw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BezierSegment:
    """A single cubic Bezier segment (4 control points)."""

    control_points: np.ndarray  # shape (4, 2) — P0, P1, P2, P3
    source_type: str = "unknown"  # "svg" | "contour"

    @property
    def start(self) -> np.ndarray:
        return self.control_points[0]

    @property
    def end(self) -> np.ndarray:
        return self.control_points[3]

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter *t* ∈ [0, 1]."""
        u = 1.0 - t
        return (u**3) * self.control_points[0] + 3 * (u**2) * t * self.control_points[1] + 3 * u * (t**2) * self.control_points[2] + (t**3) * self.control_points[3]

    def sample(self, n: int = 50) -> np.ndarray:
        """Return *n* evenly-spaced (x, y) points along the curve."""
        ts = np.linspace(0.0, 1.0, n)
        u = 1.0 - ts
        # Vectorized De Casteljau evaluation
        pts = (
            (u**3)[:, np.newaxis] * self.control_points[0]
            + 3 * (u**2)[:, np.newaxis] * ts[:, np.newaxis] * self.control_points[1]
            + 3 * u[:, np.newaxis] * (ts**2)[:, np.newaxis] * self.control_points[2]
            + (ts**3)[:, np.newaxis] * self.control_points[3]
        )  # shape (n, 2)
        return pts

    def reverse(self) -> None:
        """Reverse the direction of the control points."""
        self.control_points = self.control_points[::-1].copy()


@dataclass
class BezierPath:
    """An ordered sequence of cubic Bezier segments forming a path."""

    segments: List[BezierSegment] = field(default_factory=list)
    is_closed: bool = False
    source_type: str = "unknown"

    @property
    def control_points(self) -> np.ndarray:
        """All control points concatenated — shape (4*N, 2)."""
        if not self.segments:
            return np.empty((0, 2))
        return np.vstack([s.control_points for s in self.segments])

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def sample(self, pts_per_segment: int = 50) -> np.ndarray:
        """Sample all segments and return a single (M, 2) point array."""
        if not self.segments:
            return np.empty((0, 2))
        parts = [s.sample(pts_per_segment) for s in self.segments]
        return np.vstack(parts)

    def reverse(self) -> None:
        """Reverse the entire path sequence and direction."""
        for seg in self.segments:
            seg.reverse()
        self.segments = self.segments[::-1]


# ═══════════════════════════════════════════════════════════════════════════
# Schneider's cubic-Bezier fitting algorithm
# (adapted from the Graphics Gems / "An Algorithm for Automatically
#  Fitting Digitized Curves" — Philip J. Schneider, 1990)
# ═══════════════════════════════════════════════════════════════════════════

def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values by cumulative chord length."""
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlen = np.concatenate(([0.0], np.cumsum(diffs)))
    total = cumlen[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return cumlen / total


def _estimate_tangent(points: np.ndarray, end: str, lookahead: int = 5) -> np.ndarray:
    """Estimate a robust unit tangent at the start or end of a point sequence."""
    if len(points) < 2:
        return np.array([1.0, 0.0])

    window = max(1, min(int(lookahead), len(points) - 1))

    if end == "start":
        local_points = points[: window + 1]
    else:
        local_points = points[-(window + 1):]

    vectors = np.diff(local_points, axis=0)
    if end != "start":
        vectors = -vectors

    # Average local direction vectors to reduce pixel-level jaggedness.
    tangent = np.mean(vectors, axis=0)
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        # Fallback to first non-degenerate direction in the local window.
        for vec in vectors:
            vec_norm = np.linalg.norm(vec)
            if vec_norm >= 1e-12:
                return vec / vec_norm
        return np.array([1.0, 0.0])
    return tangent / norm


def _bezier_point(cp: np.ndarray, t: float) -> np.ndarray:
    """De Casteljau evaluation of a cubic Bezier at parameter *t*."""
    u = 1.0 - t
    return (u ** 3) * cp[0] + 3 * (u ** 2) * t * cp[1] + 3 * u * (t ** 2) * cp[2] + (t ** 3) * cp[3]


def _generate_bezier(
    points: np.ndarray,
    params: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> np.ndarray:
    """Solve for the best-fit cubic Bezier control points (Schneider §4)."""
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    # Build the A matrix (per-point contribution of the two free tangents) - vectorized
    u = 1.0 - params
    t = params
    A = np.zeros((n, 2, 2))
    A[:, 0] = 3.0 * (u**2)[:, np.newaxis] * t[:, np.newaxis] * left_tangent
    A[:, 1] = 3.0 * u[:, np.newaxis] * (t**2)[:, np.newaxis] * right_tangent

    # Build C and X matrices for the 2×2 least-squares system - vectorized with einsum
    C = np.zeros((2, 2))
    C[0, 0] = np.einsum('ij,ij->', A[:, 0, :], A[:, 0, :])
    C[0, 1] = np.einsum('ij,ij->', A[:, 0, :], A[:, 1, :])
    C[1, 0] = C[0, 1]
    C[1, 1] = np.einsum('ij,ij->', A[:, 1, :], A[:, 1, :])

    # Compute tmp and X vectorized
    tmp = (
        points
        - (u**3)[:, np.newaxis] * p0
        - 3 * (u**2)[:, np.newaxis] * t[:, np.newaxis] * p0
        - 3 * u[:, np.newaxis] * (t**2)[:, np.newaxis] * p3
        - (t**3)[:, np.newaxis] * p3
    )
    X = np.array([
        np.einsum('ij,ij->', A[:, 0, :], tmp),
        np.einsum('ij,ij->', A[:, 1, :], tmp)
    ])

    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if abs(det) < 1e-12:
        dist = np.linalg.norm(p3 - p0) / 3.0
        alpha_l = alpha_r = dist
    else:
        alpha_l = (C[1, 1] * X[0] - C[0, 1] * X[1]) / det
        alpha_r = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det

    seg_len = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        dist = seg_len / 3.0
        alpha_l = dist
        alpha_r = dist

    p1 = p0 + alpha_l * left_tangent
    p2 = p3 + alpha_r * right_tangent
    return np.vstack([p0, p1, p2, p3])


def _reparameterize(
    points: np.ndarray, params: np.ndarray, cp: np.ndarray
) -> np.ndarray:
    """Newton-Raphson reparameterization to improve parameter values."""
    u = 1.0 - params  # shape (n,)
    t = params        # shape (n,)

    # Bezier curve value at all parameters
    q = (
        (u**3)[:, np.newaxis] * cp[0]
        + 3 * (u**2)[:, np.newaxis] * t[:, np.newaxis] * cp[1]
        + 3 * u[:, np.newaxis] * (t**2)[:, np.newaxis] * cp[2]
        + (t**3)[:, np.newaxis] * cp[3]
    )  # shape (n, 2)

    # First derivative
    q1 = (
        3.0 * (u**2)[:, np.newaxis] * (cp[1] - cp[0])
        + 6.0 * u[:, np.newaxis] * t[:, np.newaxis] * (cp[2] - cp[1])
        + 3.0 * (t**2)[:, np.newaxis] * (cp[3] - cp[2])
    )  # shape (n, 2)

    # Second derivative
    q2 = (
        6.0 * u[:, np.newaxis] * (cp[2] - 2 * cp[1] + cp[0])
        + 6.0 * t[:, np.newaxis] * (cp[3] - 2 * cp[2] + cp[1])
    )  # shape (n, 2)

    diff = q - points  # shape (n, 2)

    # Newton-Raphson update
    num = np.sum(diff * q1, axis=1)  # shape (n,)
    den = np.sum(q1 * q1, axis=1) + np.sum(diff * q2, axis=1)  # shape (n,)

    new_params = params.copy()
    valid = np.abs(den) > 1e-12
    new_params[valid] = t[valid] - num[valid] / den[valid]
    new_params = np.clip(new_params, 0.0, 1.0)

    return new_params


def _max_error(
    points: np.ndarray, cp: np.ndarray, params: np.ndarray
) -> Tuple[float, int]:
    """Return (max squared error, index of worst point)."""
    # Evaluate Bezier at all parameters using vectorized De Casteljau
    u = 1.0 - params  # shape (n,)
    t = params        # shape (n,)

    # Cubic Bernstein: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
    bezier_pts = (
        (u**3)[:, np.newaxis] * cp[0]
        + 3 * (u**2)[:, np.newaxis] * t[:, np.newaxis] * cp[1]
        + 3 * u[:, np.newaxis] * (t**2)[:, np.newaxis] * cp[2]
        + (t**3)[:, np.newaxis] * cp[3]
    )  # shape (n, 2)

    # Compute all squared errors
    errors = np.sum((bezier_pts - points) ** 2, axis=1)  # shape (n,)
    split_idx = int(np.argmax(errors))
    max_err = float(errors[split_idx])

    return max_err, split_idx


def _fit_error_stats(
    points: np.ndarray, cp: np.ndarray, params: np.ndarray
) -> Tuple[float, float, int]:
    """Return (max squared error, mean squared error, worst index)."""
    u = 1.0 - params
    t = params
    bezier_pts = (
        (u**3)[:, np.newaxis] * cp[0]
        + 3 * (u**2)[:, np.newaxis] * t[:, np.newaxis] * cp[1]
        + 3 * u[:, np.newaxis] * (t**2)[:, np.newaxis] * cp[2]
        + (t**3)[:, np.newaxis] * cp[3]
    )
    errors = np.sum((bezier_pts - points) ** 2, axis=1)
    split_idx = int(np.argmax(errors))
    return float(errors[split_idx]), float(np.mean(errors)), split_idx


def _point_line_distances_sq(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Squared perpendicular distances of points to the line through p0-p1."""
    direction = p1 - p0
    denom = float(np.dot(direction, direction))
    if denom < 1e-12:
        diffs = points - p0
        return np.sum(diffs * diffs, axis=1)

    rel = points - p0
    t = np.sum(rel * direction, axis=1) / denom
    proj = p0 + t[:, np.newaxis] * direction
    diffs = points - proj
    return np.sum(diffs * diffs, axis=1)


def _max_turning_angle_deg(points: np.ndarray) -> float:
    """Maximum local turning angle in degrees for a polyline."""
    if len(points) < 3:
        return 0.0

    diffs = np.diff(points, axis=0)
    angles: List[float] = []
    for i in range(len(diffs) - 1):
        a = diffs[i]
        b = diffs[i + 1]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(dot))))

    if not angles:
        return 0.0
    return max(angles)


def _dominant_turn_index(points: np.ndarray, angle_threshold_deg: float = 70.0) -> Optional[int]:
    """Best split index for a sharp corner, or None if no dominant turn exists."""
    if len(points) < 5:
        return None

    diffs = np.diff(points, axis=0)
    best_angle = angle_threshold_deg
    best_idx: Optional[int] = None
    for i in range(1, len(diffs) - 1):
        a = diffs[i - 1]
        b = diffs[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        if angle > best_angle:
            best_angle = angle
            best_idx = i

    if best_idx is None:
        return None

    # Convert from diff-index space to point-index space.
    return max(1, min(best_idx, len(points) - 2))


def _fit_straight_cubic(p0: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Create an exactly straight cubic Bezier between endpoints."""
    direction = p3 - p0
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0])
        norm = 1.0
    unit = direction / norm
    step = norm / 3.0
    p1 = p0 + unit * step
    p2 = p3 - unit * step
    return np.vstack([p0, p1, p2, p3])


def _is_near_straight(points: np.ndarray, linear_tolerance: float) -> bool:
    """Detect near-linearity from endpoint support line and local turning."""
    if len(points) < 3:
        return True

    d2 = _point_line_distances_sq(points, points[0], points[-1])
    max_dev = math.sqrt(float(np.max(d2)))
    mean_dev = math.sqrt(float(np.mean(d2)))
    turn = _max_turning_angle_deg(points)

    return (
        max_dev <= linear_tolerance
        and mean_dev <= linear_tolerance * 0.55
        and turn <= 24.0
    )


def _fit_cubic_single(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    max_iterations: int = 4,
    straightness_scale: float = 0.35,
    corner_split_angle_deg: float = 72.0,
) -> List[np.ndarray]:
    """Fit a set of points with one or more cubic Bezier curves (recursive).

    Returns a list of (4, 2) control-point arrays.
    """
    if len(points) == 2:
        dist = np.linalg.norm(points[1] - points[0]) / 3.0
        cp = np.vstack([
            points[0],
            points[0] + dist * left_tangent,
            points[1] + dist * right_tangent,
            points[1],
        ])
        return [cp]

    linear_tol = max(0.5, math.sqrt(max_error) * straightness_scale)
    if _is_near_straight(points, linear_tol):
        return [_fit_straight_cubic(points[0], points[-1])]

    corner_idx = _dominant_turn_index(points, angle_threshold_deg=corner_split_angle_deg)
    if corner_idx is not None:
        center_tangent = points[corner_idx + 1] - points[corner_idx - 1]
        norm = np.linalg.norm(center_tangent)
        if norm < 1e-12:
            center_tangent = np.array([1.0, 0.0])
        else:
            center_tangent = center_tangent / norm

        left_curves = _fit_cubic_single(
            points[: corner_idx + 1],
            left_tangent,
            center_tangent,
            max_error,
            max_iterations=max_iterations,
            straightness_scale=straightness_scale,
            corner_split_angle_deg=corner_split_angle_deg,
        )
        right_curves = _fit_cubic_single(
            points[corner_idx:],
            -center_tangent,
            right_tangent,
            max_error,
            max_iterations=max_iterations,
            straightness_scale=straightness_scale,
            corner_split_angle_deg=corner_split_angle_deg,
        )
        return left_curves + right_curves

    params = _chord_length_parameterize(points)
    cp = _generate_bezier(points, params, left_tangent, right_tangent)
    err, mean_err, split_idx = _fit_error_stats(points, cp, params)

    mean_limit = max_error * 0.38

    if err < max_error and mean_err < mean_limit:
        return [cp]

    # Try iterative reparameterization
    if err < max_error * 4.0 or mean_err < mean_limit * 4.0:
        for _ in range(max_iterations):
            params = _reparameterize(points, params, cp)
            cp = _generate_bezier(points, params, left_tangent, right_tangent)
            err, mean_err, split_idx = _fit_error_stats(points, cp, params)
            if err < max_error and mean_err < mean_limit:
                return [cp]

    # Split at the point of maximum error and recurse
    split_idx = max(1, min(split_idx, len(points) - 2))
    center_tangent = points[split_idx + 1] - points[split_idx - 1]
    norm = np.linalg.norm(center_tangent)
    if norm < 1e-12:
        center_tangent = np.array([1.0, 0.0])
    else:
        center_tangent = center_tangent / norm

    left_curves = _fit_cubic_single(
        points[: split_idx + 1],
        left_tangent,
        center_tangent,
        max_error,
        max_iterations=max_iterations,
        straightness_scale=straightness_scale,
        corner_split_angle_deg=corner_split_angle_deg,
    )
    right_curves = _fit_cubic_single(
        points[split_idx:],
        -center_tangent,
        right_tangent,
        max_error,
        max_iterations=max_iterations,
        straightness_scale=straightness_scale,
        corner_split_angle_deg=corner_split_angle_deg,
    )
    return left_curves + right_curves


# ═══════════════════════════════════════════════════════════════════════════
# Contour → Bezier  (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class ContourBezierFitter:
    """Fit cubic Bezier curves to OpenCV contours.

    Pipeline:
      1. Reshape contour to (N, 2)
      2. Simplify with RDP (``cv2.approxPolyDP``) to find corners
      3. Split contour at corners
      4. Fit each sub-segment with Schneider's algorithm
    """

    def __init__(
        self,
        corner_threshold: float = 2.0,
        max_error: float = 5.0,
        tangent_lookahead: int = 5,
    ):
        self.corner_threshold = corner_threshold
        self.max_error_sq = max_error ** 2  # Schneider uses squared error
        self.tangent_lookahead = max(1, int(tangent_lookahead))

    def fit(self, contours: list) -> List[BezierPath]:
        """Fit Bezier paths to a list of OpenCV contours (Nx1x2 arrays)."""
        paths: List[BezierPath] = []
        for contour in contours:
            path = self._fit_single_contour(contour)
            if path is not None:
                paths.append(path)
        return paths

    def _fit_single_contour(self, contour: np.ndarray) -> Optional[BezierPath]:
        points = np.squeeze(contour).astype(np.float64)
        if points.ndim != 2 or len(points) < 2:
            return None

        # Detect if contour is closed
        is_closed = np.allclose(points[0], points[-1], atol=1.0)
        if is_closed and len(points) > 2:
            points = points[:-1]  # remove duplicate closing point

        # Find corner indices via RDP simplification
        corner_indices = self._find_corners(points, is_closed)

        # Split into segments at the corners and fit each
        segments: List[BezierSegment] = []
        n = len(corner_indices)

        if n < 2:
            # No meaningful corners — fit the entire contour as one
            cps_list = self._fit_segment(points, is_closed)
            for cps in cps_list:
                segments.append(BezierSegment(cps, source_type="contour"))
        else:
            for i in range(n - 1):
                start_idx = corner_indices[i]
                end_idx = corner_indices[i + 1]
                seg_points = points[start_idx: end_idx + 1]
                if len(seg_points) < 2:
                    continue
                cps_list = self._fit_segment(seg_points, closed=False)
                for cps in cps_list:
                    segments.append(BezierSegment(cps, source_type="contour"))

            # Close the loop if needed
            if is_closed and n >= 2:
                start_idx = corner_indices[-1]
                end_idx = corner_indices[0]
                seg_points = np.vstack([points[start_idx:], points[: end_idx + 1]])
                if len(seg_points) >= 2:
                    cps_list = self._fit_segment(seg_points, closed=False)
                    for cps in cps_list:
                        segments.append(BezierSegment(cps, source_type="contour"))

        if not segments:
            return None
        return BezierPath(segments, is_closed=is_closed, source_type="contour")

    def _find_corners(self, points: np.ndarray, is_closed: bool) -> List[int]:
        """Use RDP to find corner indices in the contour."""
        # approxPolyDP needs (N, 1, 2)
        contour_cv = points.reshape(-1, 1, 2).astype(np.float32)
        approx = cv2.approxPolyDP(contour_cv, self.corner_threshold, is_closed)
        approx_pts = np.squeeze(approx).astype(np.float64)
        if approx_pts.ndim == 1:
            approx_pts = approx_pts.reshape(1, -1)

        # Compute all distances at once using cdist
        D = cdist(approx_pts, points, metric='euclidean')  # shape (K, N)
        corner_indices = list(np.argmin(D, axis=1))  # shape (K,)

        # Sort and deduplicate
        corner_indices = sorted(set(corner_indices))
        return corner_indices

    def _fit_segment(
        self, points: np.ndarray, closed: bool = False
    ) -> List[np.ndarray]:
        """Fit cubic Beziers to *points* using Schneider's method."""
        if len(points) < 2:
            return []

        left_tangent = _estimate_tangent(points, "start", lookahead=self.tangent_lookahead)
        right_tangent = _estimate_tangent(points, "end", lookahead=self.tangent_lookahead)

        if closed and len(points) > 2:
            # For closed segments make tangents consistent at junction
            wrap_tangent = points[1] - points[-2]
            norm = np.linalg.norm(wrap_tangent)
            if norm > 1e-12:
                wrap_tangent = wrap_tangent / norm
                left_tangent = wrap_tangent
                right_tangent = -wrap_tangent

        return _fit_cubic_single(
            points, left_tangent, right_tangent, self.max_error_sq
        )


# ═══════════════════════════════════════════════════════════════════════════
# Raster image → contour extraction  (mirrors efd.py approach)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_raster_contours(
    image_path: str, min_contour_area: float = 100.0
) -> Tuple[list, Optional[np.ndarray]]:
    """Load a raster image, binarize, and return OpenCV contours."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    # Filter by area
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    return contours, img


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════


def fit_from_contours(
    contours: list,
    corner_threshold: float = 2.0,
    max_error: float = 5.0,
    tangent_lookahead: int = 5,
) -> List[BezierPath]:
    """Fit cubic Bezier paths to a list of OpenCV contours."""
    fitter = ContourBezierFitter(
        corner_threshold=corner_threshold,
        max_error=max_error,
        tangent_lookahead=tangent_lookahead,
    )
    return fitter.fit(contours)


def fit_from_image(
    image_path: str,
    min_contour_area: float = 100.0,
    corner_threshold: float = 2.0,
    max_error: float = 5.0,
    tangent_lookahead: int = 5,
) -> List[BezierPath]:
    """End-to-end: raster image → contours → fitted cubic Bezier paths."""
    contours, _ = _extract_raster_contours(image_path, min_contour_area)
    return fit_from_contours(contours, corner_threshold, max_error, tangent_lookahead)


@dataclass
class _SkeletonEdge:
    """Internal skeleton graph edge with polyline oriented from u to v."""

    edge_id: int
    u: int
    v: int
    points_uv: np.ndarray


@dataclass
class _SkeletonChain:
    """Internal chain traversed over one or more skeleton edges."""

    points: np.ndarray
    node_sequence: List[int]
    is_closed: bool


def _safe_normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (vec / norm).astype(np.float64)


def _node_xy(graph: Any, node_id: int) -> np.ndarray:
    """Best-effort conversion of sknw node position to (x, y)."""
    data = graph.nodes[node_id]
    origin = data.get("o")
    if origin is not None:
        origin_arr = np.asarray(origin, dtype=np.float64)
        if origin_arr.size >= 2:
            return origin_arr[:2][::-1]

    pts = data.get("pts")
    if pts is not None:
        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim == 2 and pts_arr.shape[1] >= 2 and len(pts_arr) > 0:
            return np.mean(pts_arr[:, :2], axis=0)[::-1]

    return np.array([0.0, 0.0], dtype=np.float64)


def _orient_points_between_nodes(pts_xy: np.ndarray, u_xy: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    """Ensure the first point is nearest u and the last point is nearest v."""
    if len(pts_xy) < 2:
        return pts_xy

    score_forward = np.linalg.norm(pts_xy[0] - u_xy) + np.linalg.norm(pts_xy[-1] - v_xy)
    score_reverse = np.linalg.norm(pts_xy[-1] - u_xy) + np.linalg.norm(pts_xy[0] - v_xy)
    if score_reverse + 1e-9 < score_forward:
        return pts_xy[::-1].copy()
    return pts_xy


def _extract_skeleton_edges(graph: Any) -> Tuple[Dict[int, _SkeletonEdge], Dict[int, List[int]]]:
    """Build canonical edge objects and node->edge incidence map from sknw graph."""
    edges: Dict[int, _SkeletonEdge] = {}
    node_to_edges: Dict[int, List[int]] = {}
    eid = 0

    if hasattr(graph, "is_multigraph") and graph.is_multigraph():
        iter_edges = graph.edges(keys=True, data=True)
        for u, v, _k, data in iter_edges:
            pts = data.get("pts", [])
            pts_arr = np.asarray(pts, dtype=np.float64)
            if pts_arr.ndim != 2 or pts_arr.shape[1] < 2 or len(pts_arr) < 2:
                continue
            pts_xy = pts_arr[:, :2][:, ::-1]
            oriented = _orient_points_between_nodes(pts_xy, _node_xy(graph, u), _node_xy(graph, v))

            edges[eid] = _SkeletonEdge(edge_id=eid, u=int(u), v=int(v), points_uv=oriented)
            node_to_edges.setdefault(int(u), []).append(eid)
            node_to_edges.setdefault(int(v), []).append(eid)
            eid += 1
    else:
        iter_edges = graph.edges(data=True)
        for u, v, data in iter_edges:
            pts = data.get("pts", [])
            pts_arr = np.asarray(pts, dtype=np.float64)
            if pts_arr.ndim != 2 or pts_arr.shape[1] < 2 or len(pts_arr) < 2:
                continue
            pts_xy = pts_arr[:, :2][:, ::-1]
            oriented = _orient_points_between_nodes(pts_xy, _node_xy(graph, u), _node_xy(graph, v))

            edges[eid] = _SkeletonEdge(edge_id=eid, u=int(u), v=int(v), points_uv=oriented)
            node_to_edges.setdefault(int(u), []).append(eid)
            node_to_edges.setdefault(int(v), []).append(eid)
            eid += 1

    return edges, node_to_edges


def _edge_other_node(edge: _SkeletonEdge, node: int) -> int:
    return edge.v if node == edge.u else edge.u


def _edge_points_from_node(edge: _SkeletonEdge, start_node: int) -> np.ndarray:
    return edge.points_uv if start_node == edge.u else edge.points_uv[::-1]


def _edge_direction_from_node(edge: _SkeletonEdge, start_node: int, steps: int = 4) -> np.ndarray:
    pts = _edge_points_from_node(edge, start_node)
    if len(pts) < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    k = min(steps, len(pts) - 1)
    direction = pts[k] - pts[0]
    return _safe_normalize_vector(direction)


def _select_continuation_edge(
    node: int,
    incoming_direction: np.ndarray,
    candidate_edge_ids: List[int],
    edges: Dict[int, _SkeletonEdge],
) -> Tuple[Optional[int], float]:
    """Choose edge with best directional continuation score at a junction."""
    best_edge: Optional[int] = None
    best_score = -2.0
    for eid in candidate_edge_ids:
        out_dir = _edge_direction_from_node(edges[eid], node)
        score = float(np.dot(incoming_direction, out_dir))
        if score > best_score:
            best_score = score
            best_edge = eid
    return best_edge, best_score


def _walk_skeleton_chain(
    start_node: int,
    start_edge_id: int,
    edges: Dict[int, _SkeletonEdge],
    node_to_edges: Dict[int, List[int]],
    degree_map: Dict[int, int],
    used_edges: Set[int],
    follow_junction_continuation: bool,
    junction_min_alignment: float,
) -> Optional[_SkeletonChain]:
    """Traverse a chain from a starting node/edge, reusing continuation at junctions."""
    if start_edge_id in used_edges:
        return None

    node_sequence: List[int] = [start_node]
    chain_points: List[np.ndarray] = []

    current_node = start_node
    current_edge_id = start_edge_id
    closed = False

    while True:
        if current_edge_id in used_edges:
            break

        edge = edges[current_edge_id]
        edge_pts = _edge_points_from_node(edge, current_node)
        if len(edge_pts) < 2:
            used_edges.add(current_edge_id)
            break

        used_edges.add(current_edge_id)
        if not chain_points:
            chain_points.extend(edge_pts)
        else:
            chain_points.extend(edge_pts[1:])

        next_node = _edge_other_node(edge, current_node)
        node_sequence.append(next_node)

        k = min(4, len(edge_pts) - 1)
        incoming_direction = _safe_normalize_vector(edge_pts[-1] - edge_pts[-1 - k])

        candidate_edges = [eid for eid in node_to_edges.get(next_node, []) if eid not in used_edges]
        if not candidate_edges:
            if next_node == start_node and len(node_sequence) > 2:
                closed = True
            break

        deg = degree_map.get(next_node, len(node_to_edges.get(next_node, [])))
        if deg == 2:
            current_edge_id = candidate_edges[0]
            current_node = next_node
            continue

        if not follow_junction_continuation:
            break

        chosen_edge, alignment = _select_continuation_edge(
            next_node,
            incoming_direction,
            candidate_edges,
            edges,
        )
        if chosen_edge is None or alignment < junction_min_alignment:
            break

        current_edge_id = chosen_edge
        current_node = next_node

    if len(chain_points) < 2:
        return None

    return _SkeletonChain(
        points=np.asarray(chain_points, dtype=np.float64),
        node_sequence=node_sequence,
        is_closed=closed,
    )


def _build_skeleton_chains(
    graph: Any,
    follow_junction_continuation: bool,
    junction_min_alignment: float,
) -> List[_SkeletonChain]:
    """Convert skeleton graph edges into continuity chains."""
    edges, node_to_edges = _extract_skeleton_edges(graph)
    if not edges:
        return []

    degree_map: Dict[int, int] = {
        node: len(edge_ids) for node, edge_ids in node_to_edges.items()
    }
    used_edges: Set[int] = set()
    chains: List[_SkeletonChain] = []

    # Prefer starting at endpoints / branch nodes, then consume any cycles.
    for node, edge_ids in node_to_edges.items():
        if degree_map.get(node, 0) == 2:
            continue
        for eid in edge_ids:
            if eid in used_edges:
                continue
            chain = _walk_skeleton_chain(
                start_node=node,
                start_edge_id=eid,
                edges=edges,
                node_to_edges=node_to_edges,
                degree_map=degree_map,
                used_edges=used_edges,
                follow_junction_continuation=follow_junction_continuation,
                junction_min_alignment=junction_min_alignment,
            )
            if chain is not None:
                chains.append(chain)

    for eid, edge in edges.items():
        if eid in used_edges:
            continue
        chain = _walk_skeleton_chain(
            start_node=edge.u,
            start_edge_id=eid,
            edges=edges,
            node_to_edges=node_to_edges,
            degree_map=degree_map,
            used_edges=used_edges,
            follow_junction_continuation=True,
            junction_min_alignment=-1.0,
        )
        if chain is not None:
            chains.append(chain)

    return chains


def _build_path_adjacency_from_chains(chains: List[_SkeletonChain]) -> Dict[int, set]:
    """Paths are adjacent if they touch the same skeleton node."""
    node_to_path_indices: Dict[int, Set[int]] = {}
    for path_idx, chain in enumerate(chains):
        for node in set(chain.node_sequence):
            node_to_path_indices.setdefault(int(node), set()).add(path_idx)

    adjacency: Dict[int, set] = {}
    for path_indices in node_to_path_indices.values():
        idx_list = sorted(path_indices)
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                pi, pj = idx_list[i], idx_list[j]
                adjacency.setdefault(pi, set()).add(pj)
                adjacency.setdefault(pj, set()).add(pi)

    return adjacency


def _prune_skeleton_spurs(graph: Any, threshold_length: float = 15.0) -> None:
    """Remove short terminal branches (spurs) often created by corners."""
    changed = True
    while changed:
        changed = False
        degrees = dict(graph.degree())
        edges_to_remove = []

        is_multi = hasattr(graph, "is_multigraph") and graph.is_multigraph()
        edges = graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)

        for edge in edges:
            u, v = edge[0], edge[1]
            deg_u, deg_v = degrees.get(u, 0), degrees.get(v, 0)

            if (deg_u == 1 and deg_v > 1) or (deg_v == 1 and deg_u > 1):
                data = edge[3] if is_multi else edge[2]
                pts = data.get("pts", [])

                if len(pts) <= threshold_length:
                    edges_to_remove.append(edge)

        for edge in edges_to_remove:
            if is_multi:
                graph.remove_edge(edge[0], edge[1], key=edge[2])
            else:
                graph.remove_edge(edge[0], edge[1])
            changed = True


def _merge_connected_paths(paths: List[BezierPath], merge_radius: float) -> List[BezierPath]:
    """Merge separate paths that connect end-to-end into single BezierPath objects."""
    if not paths:
        return []

    merged = list(paths)
    changed = True

    while changed:
        changed = False
        best_pair = None
        best_dist = float('inf')
        best_config = None

        n = len(merged)
        for i in range(n):
            for j in range(i + 1, n):
                p_i = merged[i]
                p_j = merged[j]

                if not p_i.segments or not p_j.segments:
                    continue
                if p_i.is_closed or p_j.is_closed:
                    continue

                i_start = p_i.segments[0].control_points[0]
                i_end = p_i.segments[-1].control_points[3]
                j_start = p_j.segments[0].control_points[0]
                j_end = p_j.segments[-1].control_points[3]

                configs = [
                    (np.linalg.norm(i_end - j_start), False, False),
                    (np.linalg.norm(i_end - j_end), False, True),
                    (np.linalg.norm(i_start - j_start), True, False),
                    (np.linalg.norm(i_start - j_end), True, True)
                ]

                for dist, rev_i, rev_j in configs:
                    if dist <= merge_radius and dist < best_dist:
                        best_dist = dist
                        best_pair = (i, j)
                        best_config = (rev_i, rev_j)

        if best_pair is not None:
            i, j = best_pair
            rev_i, rev_j = best_config
            p_i, p_j = merged[i], merged[j]

            if rev_i: p_i.reverse()
            if rev_j: p_j.reverse()

            midpoint = (p_i.segments[-1].control_points[3] + p_j.segments[0].control_points[0]) / 2.0
            p_i.segments[-1].control_points[3] = midpoint
            p_j.segments[0].control_points[0] = midpoint

            new_path = BezierPath(
                segments=p_i.segments + p_j.segments,
                is_closed=False,
                source_type=p_i.source_type
            )

            d_close = np.linalg.norm(new_path.segments[0].control_points[0] - new_path.segments[-1].control_points[3])
            if d_close <= merge_radius:
                new_path.is_closed = True
                midpoint = (new_path.segments[0].control_points[0] + new_path.segments[-1].control_points[3]) / 2.0
                new_path.segments[0].control_points[0] = midpoint
                new_path.segments[-1].control_points[3] = midpoint

            merged.pop(j)
            merged.pop(i)
            merged.append(new_path)
            changed = True

    return merged


def fit_from_image_skeleton(
    image_path: str,
    max_error: float = 5.0,
    tangent_lookahead: int = 5,
    merge_radius: float = 15.0,
    follow_junction_continuation: bool = True,
    junction_min_alignment: float = -0.30,
    spur_threshold: float = 15.0,
) -> Tuple[List[BezierPath], Dict[int, set]]:
    """End-to-end: raster image → skeleton → graph → fitted cubic Bezier paths."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary / 255.0).astype(np.uint8)
    graph = sknw.build_sknw(skeleton)

    _prune_skeleton_spurs(graph, threshold_length=spur_threshold)

    chains = _build_skeleton_chains(
        graph,
        follow_junction_continuation=follow_junction_continuation,
        junction_min_alignment=junction_min_alignment,
    )

    paths: List[BezierPath] = []
    lookahead = max(1, int(tangent_lookahead))
    for chain in chains:
        pts = chain.points
        if len(pts) < 2:
            continue

        left_tangent = _estimate_tangent(pts, "start", lookahead=lookahead)
        right_tangent = _estimate_tangent(pts, "end", lookahead=lookahead)
        cps_list = _fit_cubic_single(pts, left_tangent, right_tangent, max_error ** 2)

        segments = [BezierSegment(cps, source_type="skeleton") for cps in cps_list]
        if not segments:
            continue

        paths.append(
            BezierPath(
                segments=segments,
                is_closed=chain.is_closed,
                source_type="skeleton",
            )
        )

    merged_paths = _merge_connected_paths(paths, merge_radius=merge_radius)

    return merged_paths, {}


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════


def _draw_paths_on_canvas(
    canvas: np.ndarray,
    paths: List[BezierPath],
    colors: Tuple[Tuple[float, float, float], ...],
    pts_per_segment: int,
    show_controls: bool,
) -> np.ndarray:
    """Render Bezier paths onto an image-space canvas using OpenCV."""
    rendered = canvas.copy()
    curve_palette_bgr = [
        (255, 255, 0),   # neon cyan
        (0, 255, 255),   # bright yellow
        (255, 140, 0),   # bright orange
        (50, 255, 50),   # bright lime
        (255, 60, 255),  # bright magenta
        (100, 200, 255), # light orange-blue tint
    ]
    control_line_bgr = (180, 180, 180)
    control_point_bgr = (255, 255, 255)

    for idx, bp in enumerate(paths):
        color_bgr = curve_palette_bgr[idx % len(curve_palette_bgr)]
        for seg in bp.segments:
            pts = np.round(seg.sample(pts_per_segment)).astype(np.int32).reshape((-1, 1, 2))
            if len(pts) >= 2:
                cv2.polylines(rendered, [pts], False, color_bgr, 2, lineType=cv2.LINE_AA)

            if not show_controls:
                continue

            cp = np.round(seg.control_points).astype(np.int32)
            cv2.line(rendered, tuple(cp[0]), tuple(cp[1]), control_line_bgr, 1, lineType=cv2.LINE_AA)
            cv2.line(rendered, tuple(cp[2]), tuple(cp[3]), control_line_bgr, 1, lineType=cv2.LINE_AA)
            for point in cp:
                cv2.circle(rendered, tuple(point), 3, control_point_bgr, -1, lineType=cv2.LINE_AA)

    return rendered


def _visualize_paths_single_panel(
    paths: List[BezierPath],
    title: str,
    save_path: Optional[str],
    show_controls: bool,
    pts_per_segment: int,
) -> None:
    """Fallback single-panel plot for callers without source-image context."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_title(title)
    ax.title.set_color("white")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    all_points = []
    bright_colors = [
        "#00FFFF",  # neon cyan
        "#FFFF00",  # bright yellow
        "#39FF14",  # neon green
        "#FF5F1F",  # bright orange
        "#FF2CF0",  # bright magenta
        "#6EE7FF",  # bright sky
        "#FFD166",  # warm bright yellow
        "#8BFF9E",  # mint bright
        "#FF8BA7",  # bright rose
        "#B794FF",  # bright violet
    ]
    for idx, bp in enumerate(paths):
        color = bright_colors[idx % len(bright_colors)]
        for seg in bp.segments:
            pts = seg.sample(pts_per_segment)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0)
            all_points.append(pts)

            if not show_controls:
                continue

            cp = seg.control_points
            all_points.append(cp)
            ax.plot(
                [cp[0, 0], cp[1, 0]], [cp[0, 1], cp[1, 1]],
                color="#CCCCCC", linewidth=0.9, linestyle="--", alpha=0.8,
            )
            ax.plot(
                [cp[2, 0], cp[3, 0]], [cp[2, 1], cp[3, 1]],
                color="#CCCCCC", linewidth=0.9, linestyle="--", alpha=0.8,
            )
            ax.plot(
                cp[:, 0], cp[:, 1], "o",
                color="#FFFFFF", markersize=3.5, alpha=0.9,
            )

    if all_points:
        all_pts_array = np.concatenate(all_points, axis=0)
        x_min, y_min = all_pts_array.min(axis=0)
        x_max, y_max = all_pts_array.max(axis=0)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        padding_x = x_range * 0.1
        padding_y = y_range * 0.1
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Matplotlib uses Cartesian y-up; image-derived data is y-down.
    if paths and paths[0].source_type in {"contour", "skeleton"}:
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"  -> Saved visualization -> {save_path}")
        plt.close(fig)
        return

    plt.show()


def _visualize_paths_overview(
    paths: List[BezierPath],
    title: str,
    save_path: Optional[str],
    image_path: str,
    show_controls: bool,
    pts_per_segment: int,
) -> None:
    """Render an EFD-style 3-panel overview for Bezier results."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        _visualize_paths_single_panel(paths, title, save_path, show_controls, pts_per_segment)
        return

    h, w = img_bgr.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].title.set_color("white")
    axes[0].axis("off")

    colors = plt.cm.tab10.colors
    curves_only = _draw_paths_on_canvas(
        np.zeros((h, w, 3), dtype=np.uint8),
        paths,
        colors,
        pts_per_segment,
        show_controls=False,
    )
    axes[1].imshow(cv2.cvtColor(curves_only, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Bezier Curves Only ({len(paths)})")
    axes[1].title.set_color("white")
    axes[1].axis("off")

    curves_with_controls = _draw_paths_on_canvas(
        np.zeros((h, w, 3), dtype=np.uint8),
        paths,
        colors,
        pts_per_segment,
        show_controls=show_controls,
    )
    axes[2].imshow(cv2.cvtColor(curves_with_controls, cv2.COLOR_BGR2RGB))
    if show_controls:
        axes[2].set_title("Bezier Curves + Controls")
    else:
        axes[2].set_title("Bezier Curves Overview")
    axes[2].title.set_color("white")
    axes[2].axis("off")

    suptitle = fig.suptitle(title)
    suptitle.set_color("white")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"  -> Saved visualization -> {save_path}")
        plt.close(fig)
        return

    plt.show()


def _comparison_save_path_from_overview_path(save_path: str) -> str:
    """Derive a companion compare filename from the main overview path."""
    directory, filename = os.path.split(save_path)
    stem, ext = os.path.splitext(filename)
    if stem.endswith("_bezier_vis"):
        stem = stem[: -len("_bezier_vis")]
    if not ext:
        ext = ".png"
    return os.path.join(directory, f"{stem}_bezier_compare{ext}")


def _save_original_vs_bezier_controls(
    paths: List[BezierPath],
    image_path: str,
    save_path: str,
    pts_per_segment: int,
) -> None:
    """Save a 2-panel comparison image: original and Bezier+controls."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    h, w = img_bgr.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].title.set_color("white")
    axes[0].axis("off")

    curves_with_controls = _draw_paths_on_canvas(
        np.zeros((h, w, 3), dtype=np.uint8),
        paths,
        plt.cm.tab10.colors,
        pts_per_segment,
        show_controls=True,
    )
    axes[1].imshow(cv2.cvtColor(curves_with_controls, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Bezier Curves + Controls")
    axes[1].title.set_color("white")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"  -> Saved comparison visualization -> {save_path}")
    plt.close(fig)

def visualize_paths(
    paths: List[BezierPath],
    title: str = "Bezier Curves",
    save_path: Optional[str] = None,
    show_controls: bool = True,
    pts_per_segment: int = 80,
    image_path: Optional[str] = None,
    overview: bool = True,
) -> None:
    """Plot Bezier paths with optional EFD-style overview rendering.

    Args:
        paths:           list of BezierPath objects
        title:           figure title
        save_path:       if given, save figure to this file
        show_controls:   draw control points and handle lines
        pts_per_segment: sampling density per segment
        image_path:      source raster used for overview panel 1
        overview:        when True and image_path is available, use 3-panel layout
    """
    if overview and image_path:
        _visualize_paths_overview(
            paths,
            title,
            save_path,
            image_path,
            show_controls,
            pts_per_segment,
        )

        if save_path:
            compare_path = _comparison_save_path_from_overview_path(save_path)
            _save_original_vs_bezier_controls(
                paths,
                image_path,
                compare_path,
                pts_per_segment,
            )
        return

    _visualize_paths_single_panel(paths, title, save_path, show_controls, pts_per_segment)


# ═══════════════════════════════════════════════════════════════════════════
# Quick-run CLI
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(paths: List[BezierPath], label: str) -> None:
    total_segs = sum(p.num_segments for p in paths)
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  Paths: {len(paths)}  |  Total segments: {total_segs}")
    for i, p in enumerate(paths):
        status = "closed" if p.is_closed else "open"
        print(f"    Path {i}: {p.num_segments} segments ({status})")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit cubic Bezier curves from raster plans.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--no-skeleton", action="store_false", dest="skeleton", help="Disable skeleton-based fitting")
    parser.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area for contour extraction",
    )
    parser.add_argument(
        "--tangent-lookahead",
        type=int,
        default=5,
        help="Lookahead window used for tangent estimation",
    )
    parser.set_defaults(skeleton=True)
    args = parser.parse_args()

    if args.skeleton:
        paths, _adjacency = fit_from_image_skeleton(
            args.image_path,
            tangent_lookahead=args.tangent_lookahead,
        )
        label = f"Skeleton fitting: {args.image_path}"
    else:
        paths = fit_from_image(
            args.image_path,
            min_contour_area=args.min_area,
            tangent_lookahead=args.tangent_lookahead,
        )
        label = f"Contour fitting: {args.image_path}"

    _print_summary(paths, label)

    if paths:
        visualize_paths(paths, title=label, save_path=os.path.join(OUTPUT_DIR, "bezier_vis.png"))
