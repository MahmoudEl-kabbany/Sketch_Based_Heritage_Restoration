"""
Bezier Curve Extraction Module
==============================
Skeleton image -> graph edges -> Schneider fit -> cubic Bezier

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
from scipy.ndimage import gaussian_filter, maximum_filter
from skimage.morphology import skeletonize
import sknw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing helpers (improvements #1–#4)
# ═══════════════════════════════════════════════════════════════════════════

def _round_to_odd(x: float) -> int:
    """Round to the nearest odd integer ≥ 3."""
    v = max(3, int(round(x)))
    return v if v % 2 == 1 else v + 1


def _estimate_median_stroke_width(binary: np.ndarray) -> float:
    """Estimate median stroke width from distance transform peaks.

    Stroke width ≈ 2× the distance transform value at skeleton pixels.
    Returns a fallback of 2.0 when the image has too few foreground pixels.
    """
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skel = skeletonize(binary // 255).astype(np.uint8)
    ridge_values = dist[skel > 0]
    if len(ridge_values) < 10:
        return 2.0
    return float(np.median(ridge_values)) * 2.0


def _robust_binarize(gray: np.ndarray) -> np.ndarray:
    """Two-pass binarization: CLAHE + Otsu, with adaptive fallback.

    CLAHE normalizes local contrast to handle uneven backgrounds common in
    heritage photographs (stone walls, parchment stains, shadow gradients).
    When Otsu produces an implausible foreground ratio (<2% or >60%), an
    adaptive Gaussian threshold is used instead.
    """
    # Pass 1: CLAHE to normalize local contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Pass 2: Otsu on the equalized image
    _, binary_otsu = cv2.threshold(
        equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Sanity check: if Otsu captures < 2% or > 60% foreground,
    # fall back to adaptive thresholding
    fg_ratio = np.count_nonzero(binary_otsu) / max(binary_otsu.size, 1)
    if fg_ratio < 0.02 or fg_ratio > 0.60:
        binary_adaptive = cv2.adaptiveThreshold(
            equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=31, C=8,
        )
        return binary_adaptive

    return binary_otsu


def _remove_small_components(binary: np.ndarray, min_area: int = 8) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels.

    Unlike median blur, this preserves thin stroke endpoints while
    eliminating isolated noise blobs.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    clean = np.zeros_like(binary)
    for i in range(1, n_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def _anisotropic_diffusion(
    gray: np.ndarray,
    n_iter: int = 10,
    kappa: float = 30.0,
    gamma: float = 0.15,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion — smooths texture, preserves edges.

    Useful as a pre-filter for photographic heritage inputs where stone,
    plaster, or parchment texture creates noise that survives binarization.
    On clean high-contrast sketches the filter is effectively a no-op.
    """
    img = gray.astype(np.float64)
    for _ in range(n_iter):
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)

        img += gamma * (
            c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w
        )

    return np.clip(img, 0, 255).astype(np.uint8)


def _chaikin_smooth(pts: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Chaikin corner-cutting subdivision: converges to a quadratic B-spline.

    Produces C1-smooth output while preserving the original endpoints.
    Better than a uniform moving average for handling diagonal staircase
    artifacts common in skeleton pixel traces.
    """
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        q = np.empty((2 * len(pts) - 2, 2), dtype=np.float64)
        q[0::2] = 0.75 * pts[:-1] + 0.25 * pts[1:]
        q[1::2] = 0.25 * pts[:-1] + 0.75 * pts[1:]
        # Preserve original endpoints exactly
        q[0] = pts[0]
        q[-1] = pts[-1]
        pts = q
    return pts



# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BezierSegment:
    """A single cubic Bezier segment (4 control points)."""

    control_points: np.ndarray  # shape (4, 2) — P0, P1, P2, P3
    source_type: str = "unknown"  # "svg" | "skeleton"

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


def _estimate_tangent(points: np.ndarray, end: str, lookahead: int = 10, skip_tip: int = 2) -> np.ndarray:
    """Estimate a robust unit tangent at the start or end of a point sequence.

    Uses exponential distance weighting so nearer samples contribute more,
    while the wider window (default 10) provides stability against pixel-level
    jaggedness and junction artifacts.
    
    The skip_tip parameter allows ignoring the extreme terminal pixels to avoid
    directional bias from noisy endpoints.
    """
    if len(points) < 2:
        return np.array([1.0, 0.0])

    window = max(1, min(int(lookahead), len(points) - 1))

    if end == "start":
        local_points = points[skip_tip : skip_tip + window + 1] if len(points) > skip_tip + 1 else points[: window + 1]
    else:
        local_points = points[-(skip_tip + window + 1) : -skip_tip] if len(points) > skip_tip + 1 else points[-(window + 1):]

    vectors = np.diff(local_points, axis=0)
    if end != "start":
        vectors = -vectors

    n_vecs = len(vectors)
    if n_vecs == 0:
        return np.array([1.0, 0.0])

    # Exponential distance weighting: nearer vectors get higher weight.
    # decay_rate chosen so the farthest vector has ~30% of the nearest's weight.
    decay_rate = 1.2 / max(n_vecs - 1, 1)
    weights = np.exp(-decay_rate * np.arange(n_vecs))

    # Down-weight vectors at high local curvature (turning angle > 25°).
    for i in range(n_vecs - 1):
        a = vectors[i]
        b = vectors[i + 1]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-12 and nb > 1e-12:
            dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            if angle > 25.0:
                # Reduce influence of vectors around the turning point.
                weights[i] *= 0.3
                weights[i + 1] *= 0.3

    tangent = np.zeros(2, dtype=np.float64)
    for vec, w in zip(vectors, weights):
        tangent += vec * w

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

    # Stop once the update has effectively converged so higher-level fitting
    # loops do not keep iterating on numerically stagnant parameters.
    if np.max(np.abs(new_params - params)) < 1e-4:
        return params

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
        and mean_dev <= linear_tolerance * 0.70
        and turn <= 32.0
    )


def _fit_cubic_single(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    max_iterations: int = 4,
    straightness_scale: float = 0.75,
    corner_split_angle_deg: float = 72.0,
    line_preference_improvement: float = 0.15,
    recursion_depth: int = 0,
) -> List[np.ndarray]:
    """Fit a set of points with one or more cubic Bezier curves (recursive).

    Returns a list of (4, 2) control-point arrays.
    """
    max_recursion_depth = 48
    if recursion_depth >= max_recursion_depth:
        return [_fit_straight_cubic(points[0], points[-1])]

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
            line_preference_improvement=line_preference_improvement,
            recursion_depth=recursion_depth + 1,
        )
        right_curves = _fit_cubic_single(
            points[corner_idx:],
            -center_tangent,
            right_tangent,
            max_error,
            max_iterations=max_iterations,
            straightness_scale=straightness_scale,
            corner_split_angle_deg=corner_split_angle_deg,
            line_preference_improvement=line_preference_improvement,
            recursion_depth=recursion_depth + 1,
        )
        return left_curves + right_curves

    params = _chord_length_parameterize(points)
    straight_cp = _fit_straight_cubic(points[0], points[-1])
    straight_err, straight_mean_err, _ = _fit_error_stats(points, straight_cp, params)

    cp = _generate_bezier(points, params, left_tangent, right_tangent)
    err, mean_err, split_idx = _fit_error_stats(points, cp, params)

    # Prefer an exact straight cubic unless the flexible cubic significantly improves fit.
    if straight_err <= max_error * 1.35:
        improvement = (straight_mean_err - mean_err) / max(straight_mean_err, 1e-12)
        if improvement < line_preference_improvement:
            return [straight_cp]

    mean_limit = max_error * 0.38

    if err < max_error and mean_err < mean_limit:
        return [cp]

    # Try iterative reparameterization
    # ── Improvement #9: oscillation early-exit ──
    if err < max_error * 4.0 or mean_err < mean_limit * 4.0:
        prev_err = err
        for _ in range(max_iterations):
            next_params = _reparameterize(points, params, cp)
            if np.max(np.abs(next_params - params)) < 1e-4:
                params = next_params
                break
            params = next_params
            cp = _generate_bezier(points, params, left_tangent, right_tangent)
            err, mean_err, split_idx = _fit_error_stats(points, cp, params)
            if err >= prev_err:  # Oscillation detected → stop
                break
            prev_err = err
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
        line_preference_improvement=line_preference_improvement,
        recursion_depth=recursion_depth + 1,
    )
    right_curves = _fit_cubic_single(
        points[split_idx:],
        -center_tangent,
        right_tangent,
        max_error,
        max_iterations=max_iterations,
        straightness_scale=straightness_scale,
        corner_split_angle_deg=corner_split_angle_deg,
        line_preference_improvement=line_preference_improvement,
        recursion_depth=recursion_depth + 1,
    )
    return left_curves + right_curves


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


def _select_continuation_by_curvature(
    node: int,
    incoming_direction: np.ndarray,
    candidate_edge_ids: List[int],
    edges: Dict[int, _SkeletonEdge],
    degree: int,
) -> Tuple[Optional[int], float]:
    """
    At degree-2 or degree-3 nodes, use direction alignment (existing behaviour).
    At degree-4+ nodes (true X-junctions), also score by curvature continuity:
    prefer the outgoing edge whose initial direction produces the smallest
    absolute change in curvature, computed as the angular difference between
    the incoming direction and the outgoing direction (smaller turn = smoother).
    Return (best_edge_id, alignment_score).
    """
    if degree < 4:
        best_edge = None
        best_score = -2.0
        for eid in candidate_edge_ids:
            out_dir = _edge_direction_from_node(edges[eid], node)
            score = float(np.dot(incoming_direction, out_dir))
            if score > best_score:
                best_score = score
                best_edge = eid
        return best_edge, best_score

    best_edge = None
    best_score = -2.0
    for eid in candidate_edge_ids:
        out_dir = _edge_direction_from_node(edges[eid], node)
        dot = float(np.clip(np.dot(incoming_direction, out_dir), -1.0, 1.0))
        turning_angle = float(np.arccos(dot))
        curvature_continuity = 1.0 - (turning_angle / np.pi)
        direction_alignment = dot
        score = 0.5 * direction_alignment + 0.5 * curvature_continuity
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

        deg = degree_map.get(next_node, len(node_to_edges.get(next_node, [])))
        chosen_edge, alignment = _select_continuation_by_curvature(
            next_node,
            incoming_direction,
            candidate_edges,
            edges,
            deg,
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


def _prune_skeleton_spurs(
    graph: Any,
    threshold_length: float = 15.0,
    protect_isolated: bool = True,
) -> None:
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

            # Do not prune if it is the only edge connected to the junction node,
            # which would isolate real short strokes that happen to end at a junction.
            if protect_isolated:
                if deg_u == 1 and deg_v > 1:
                    # Only prune if the junction node (v) has at least 3 other edges
                    # remaining after this removal (i.e. degree > 2 after pruning).
                    if degrees.get(v, 0) <= 2:
                        continue
                if deg_v == 1 and deg_u > 1:
                    if degrees.get(u, 0) <= 2:
                        continue

            if (deg_u == 1 and deg_v > 1) or (deg_v == 1 and deg_u > 1):
                data = edge[3] if is_multi else edge[2]
                pts = data.get("pts", [])

                # Use geometric arc length instead of raw point count for
                # orientation-independent spur measurement (#8).
                pts_arr = np.asarray(pts, dtype=np.float64)
                if pts_arr.ndim == 2 and len(pts_arr) >= 2:
                    arc_length = float(np.sum(np.linalg.norm(
                        np.diff(pts_arr, axis=0), axis=1
                    )))
                else:
                    arc_length = float(len(pts))

                if arc_length <= threshold_length:
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

        def endpoint_tangent(path: BezierPath, end: str) -> np.ndarray:
            if end == "start":
                cp_vec = path.segments[0].control_points[0] - path.segments[0].control_points[1]
                tangent = _safe_normalize_vector(cp_vec)
                if np.linalg.norm(cp_vec) < 1e-12:
                    sampled = path.sample(6)
                    if len(sampled) >= 2:
                        tangent = _safe_normalize_vector(sampled[1] - sampled[0])
                return tangent

            cp_vec = path.segments[-1].control_points[3] - path.segments[-1].control_points[2]
            tangent = _safe_normalize_vector(cp_vec)
            if np.linalg.norm(cp_vec) < 1e-12:
                sampled = path.sample(6)
                if len(sampled) >= 2:
                    tangent = _safe_normalize_vector(sampled[-1] - sampled[-2])
            return tangent

        tangent_cache: Dict[Tuple[int, str], np.ndarray] = {}
        for idx, path in enumerate(merged):
            tangent_cache[(idx, "start")] = endpoint_tangent(path, "start")
            tangent_cache[(idx, "end")] = endpoint_tangent(path, "end")

        # Build spatial hash grid to avoid O(n²) pairwise checks
        grid_cell_size = max(1.0, merge_radius)
        grid: Dict[Tuple[int, int], List[int]] = {}
        endpoint_positions: Dict[int, List[np.ndarray]] = {}
        
        for idx, path in enumerate(merged):
            if not path.segments or path.is_closed:
                continue
            
            start_pos = path.segments[0].control_points[0]
            end_pos = path.segments[-1].control_points[3]
            endpoint_positions[idx] = [start_pos, end_pos]
            
            # Hash both start and end points
            for pos in [start_pos, end_pos]:
                gx = int(np.floor(pos[0] / grid_cell_size))
                gy = int(np.floor(pos[1] / grid_cell_size))
                key = (gx, gy)
                if key not in grid:
                    grid[key] = []
                if idx not in grid[key]:
                    grid[key].append(idx)
        
        # Check only pairs from nearby grid cells
        candidate_pairs: Set[Tuple[int, int]] = set()
        for cell_indices in grid.values():
            for i in range(len(cell_indices)):
                for j in range(i + 1, len(cell_indices)):
                    idx_i, idx_j = cell_indices[i], cell_indices[j]
                    candidate_pairs.add((min(idx_i, idx_j), max(idx_i, idx_j)))
        
        # Also check adjacent grid cells for robustness
        checked_pairs: Set[Tuple[int, int]] = set()
        for (gx, gy), cell_indices in grid.items():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (gx + dx, gy + dy)
                    if neighbor_key in grid and neighbor_key != (gx, gy):
                        for i in cell_indices:
                            for j in grid[neighbor_key]:
                                if i < j:
                                    candidate_pairs.add((i, j))
                                elif i > j:
                                    candidate_pairs.add((j, i))
        
        for i, j in candidate_pairs:
            if (i, j) in checked_pairs:
                continue
            checked_pairs.add((i, j))
            
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
                if dist > merge_radius:
                    continue

                if not rev_i and not rev_j:
                    dot = float(np.dot(tangent_cache[(i, "end")], tangent_cache[(j, "start")]))
                elif not rev_i and rev_j:
                    dot = float(np.dot(tangent_cache[(i, "end")], tangent_cache[(j, "end")]))
                elif rev_i and not rev_j:
                    dot = float(np.dot(tangent_cache[(i, "start")], tangent_cache[(j, "start")]))
                else:
                    dot = float(np.dot(tangent_cache[(i, "start")], tangent_cache[(j, "end")]))

                if dot < -0.4 and dist < best_dist:
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
        else:
            changed = False

    return merged


def _refine_paths_differentiable(
    paths: List[BezierPath],
    binary_mask: np.ndarray,
    n_iterations: int = 6,
    step_size: float = 0.35,
    sample_count: int = 40,
) -> List[BezierPath]:
    """
    Nudge internal control points P1 and P2 of each segment toward the true
    centerline using a simple numerical gradient descent against the binary mask.

    For each segment, sample `sample_count` points along the curve. For each
    sampled point, look up the value in the distance transform of the binary mask
    (foreground=255 after THRESH_BINARY_INV). The gradient of this distance
    transform at each sample point indicates the direction toward the nearest
    foreground pixel. Accumulate the mean gradient over all sample points and
    apply it as a small displacement to P1 and P2 only (never P0 or P3, which
    are topology-anchoring endpoints).

    Steps:
    1. Compute dist_transform = cv2.distanceTransform(
           (255 - binary_mask).astype(np.uint8), cv2.DIST_L2, 5
       )
       where binary_mask is the thresholded foreground (255=line, 0=background).
    2. Compute the x and y Sobel gradients of dist_transform:
           grad_x = cv2.Sobel(dist_transform, cv2.CV_64F, 1, 0, ksize=3)
           grad_y = cv2.Sobel(dist_transform, cv2.CV_64F, 0, 1, ksize=3)
    3. For each iteration (n_iterations total):
       For each path, for each segment:
         a. Sample `sample_count` points along the segment using segment.sample().
         b. For each sampled point (clipped to image bounds), read grad_x and
            grad_y at that pixel location.
         c. Compute mean_gradient = mean over all sample points of [gx, gy].
         d. Negate mean_gradient (we want to move toward lower distance, i.e.
            toward the line).
         e. Displace P1 by: P1 -= step_size * mean_gradient
            Displace P2 by: P2 -= step_size * mean_gradient
         f. After displacing, re-clamp P1 and P2 so they remain within image
            bounds (0 to w-1 for x, 0 to h-1 for y).
    4. Return the modified paths list.
    """
    if not paths:
        return paths

    dist_transform = cv2.distanceTransform((255 - binary_mask).astype(np.uint8), cv2.DIST_L2, 5)
    grad_x = cv2.Sobel(dist_transform, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist_transform, cv2.CV_64F, 0, 1, ksize=3)
    h, w = binary_mask.shape[:2]

    for _ in range(n_iterations):
        for path in paths:
            for segment in path.segments:
                sampled = segment.sample(sample_count)
                if len(sampled) == 0:
                    continue

                sampled_xy = np.round(sampled).astype(np.int32)
                sampled_xy[:, 0] = np.clip(sampled_xy[:, 0], 0, w - 1)
                sampled_xy[:, 1] = np.clip(sampled_xy[:, 1], 0, h - 1)

                gradients = np.column_stack([
                    grad_x[sampled_xy[:, 1], sampled_xy[:, 0]],
                    grad_y[sampled_xy[:, 1], sampled_xy[:, 0]],
                ])
                mean_gradient = -np.mean(gradients, axis=0)

                control_points = segment.control_points.astype(np.float64, copy=True)
                control_points[1] -= step_size * mean_gradient
                control_points[2] -= step_size * mean_gradient
                control_points[1, 0] = np.clip(control_points[1, 0], 0.0, float(w - 1))
                control_points[1, 1] = np.clip(control_points[1, 1], 0.0, float(h - 1))
                control_points[2, 0] = np.clip(control_points[2, 0], 0.0, float(w - 1))
                control_points[2, 1] = np.clip(control_points[2, 1], 0.0, float(h - 1))
                segment.control_points = control_points

    return paths


def fit_from_image_skeleton(
    image_path: str,
    max_error: float = 1.5,
    tangent_lookahead: int = 8,
    straightness_scale: float = 0.75,
    merge_radius: float = 5.0,
    follow_junction_continuation: bool = True,
    junction_min_alignment: float = -0.15,
    spur_threshold: float = 25.0,
) -> Tuple[List[BezierPath], Dict[int, set]]:
    """End-to-end: raster image → skeleton → graph → fitted cubic Bezier paths."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ── Improvement #4: anisotropic diffusion for photographic inputs ──
    # Smooths stone/plaster texture while preserving drawn edges.
    # Detects whether the image is "photographic" (high local variance) and
    # applies diffusion only when beneficial.
    local_var = float(np.std(img.astype(np.float64)))
    if local_var > 45.0:
        img_filtered = _anisotropic_diffusion(img, n_iter=8, kappa=30.0, gamma=0.12)
    else:
        img_filtered = img

    # ── Improvement #2: robust CLAHE-based binarization ──
    binary = _robust_binarize(img_filtered)

    # ── Improvement #3: CC-based denoising (replaces median blur) ──
    # Estimate stroke width before denoising so we can scale the min area.
    stroke_width = _estimate_median_stroke_width(binary)
    min_component_area = max(8, int(stroke_width ** 2 * 0.5))
    binary = _remove_small_components(binary, min_area=min_component_area)

    # ── Improvement #1: adaptive morphological kernels ──
    close_ksize = _round_to_odd(stroke_width * 0.6)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    # Skip dilation for thick strokes (>6 px) — they don't need reconnection
    # and dilation would merge parallel nearby strokes.
    if stroke_width <= 6.0:
        dilate_ksize = max(2, _round_to_odd(stroke_width * 0.3))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        binary = cv2.dilate(binary, dilate_kernel, iterations=1)

    skeleton = _udf_skeleton(binary)
    graph = sknw.build_sknw(skeleton)

    _prune_skeleton_spurs(graph, threshold_length=spur_threshold, protect_isolated=True)

    # Remove degree-0 isolate nodes left by spur pruning.
    isolates = [n for n, d in dict(graph.degree()).items() if d == 0]
    graph.remove_nodes_from(isolates)

    chains = _build_skeleton_chains(
        graph,
        follow_junction_continuation=follow_junction_continuation,
        junction_min_alignment=junction_min_alignment,
    )

    paths: List[BezierPath] = []
    lookahead = max(1, int(tangent_lookahead))
    for chain in chains:
        pts = chain.points
        
        # For damaged sketches, trim only 1 pixel from each end.
        # Trimming 3 pixels discards real stroke data when strokes are short.
        trim_len = 1 if len(pts) > 6 else 0
        if not chain.is_closed and trim_len > 0 and len(pts) > (trim_len * 2) + 2:
            pts = pts[trim_len:-trim_len]

        # ── Improvement #6: Chaikin subdivision smoothing ──
        # Replaces the moving-average smoother. Chaikin corner-cutting
        # naturally handles diagonal staircase artifacts and produces
        # C1-smooth polylines without a fixed window size.
        if len(pts) >= 5:
            original_start = pts[0].copy()
            original_end = pts[-1].copy()
            pts = _chaikin_smooth(pts, iterations=1)
            # Preserve original endpoints exactly; only smooth the interior.
            pts[0] = original_start
            pts[-1] = original_end

        if len(pts) < 2:
            continue

        # ── Improvement #7: adaptive max_error per chain arc length ──
        chain_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if chain_length < 20:
            adaptive_error = max_error * 0.6  # tighter for short chains
        elif chain_length > 200:
            adaptive_error = max_error * 1.4  # looser for long smooth curves
        else:
            adaptive_error = max_error

        left_tangent = _estimate_tangent(pts, "start", lookahead=lookahead)
        right_tangent = _estimate_tangent(pts, "end", lookahead=lookahead)
        cps_list = _fit_cubic_single(
            pts,
            left_tangent,
            right_tangent,
            adaptive_error ** 2,
            straightness_scale=straightness_scale,
        )

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

    # Compute binary mask for differentiable refinement.
    # Use the same robust binarization for consistency.
    binary_refine = _robust_binarize(img)
    merged_paths = _refine_paths_differentiable(merged_paths, binary_refine)

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
    if paths and paths[0].source_type in {"skeleton"}:
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


def _udf_skeleton(binary: np.ndarray, udf_blur_sigma: float = 1.2) -> np.ndarray:
    """
    Produce a robust skeleton from a binary foreground mask (foreground=255)
    suitable for damaged sketches with broken lines and gaps.

    Strategy: compute both a UDF ridge skeleton AND a morphological skeleton,
    then take their union. This ensures that regions where the UDF ridge
    detector finds no ridge (because a stroke is too thin or damaged) are
    still covered by the morphological skeleton, while regions where the UDF
    ridge is cleaner (at junctions and thick strokes) benefit from its
    topological accuracy.

    Steps:
    1. Compute morphological skeleton as a fallback:
        from skimage.morphology import skeletonize as ski_skeletonize
        morph_skel = ski_skeletonize(binary // 255).astype(np.uint8)

    2. Compute the Euclidean distance transform of the foreground:
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    3. Smooth the distance field:
        from scipy.ndimage import gaussian_filter
        dist_smooth = gaussian_filter(dist, sigma=udf_blur_sigma)

    4. Detect local maxima of the smoothed distance field:
        from scipy.ndimage import maximum_filter
        local_max = (dist_smooth == maximum_filter(dist_smooth, size=3))

    5. Keep only maxima on the foreground with dist > 0.3 (slightly above zero
    to exclude noisy single-pixel foreground hits):
        ridge = (local_max & (dist > 0.3)).astype(np.uint8)

    6. Thin the ridge to one pixel wide:
        ridge_thin = ski_skeletonize(ridge).astype(np.uint8)

    7. Take the union of both skeletons:
        combined = np.clip(morph_skel + ridge_thin, 0, 1).astype(np.uint8)

    8. Thin the union once more to remove any 2-pixel-wide seams introduced
    by the union operation:
        final = ski_skeletonize(combined).astype(np.uint8)

    9. Return final (values 0 or 1).
    """
    morph_skel = skeletonize(binary // 255).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_smooth = gaussian_filter(dist, sigma=udf_blur_sigma)
    local_max = dist_smooth == maximum_filter(dist_smooth, size=3)

    # ── Improvement #5: hysteresis thresholding for UDF ridges ──
    # A single hard threshold (dist > 0.3) loses thin strokes entirely.
    # Instead, use a high/low threshold pair (like Canny edge detection):
    # strong ridges seed the result, weak ridges extend them if connected.
    fg_dists = dist[dist > 0]
    if len(fg_dists) > 10:
        high_threshold = max(0.5, float(np.percentile(fg_dists, 25)))
    else:
        high_threshold = 0.5
    low_threshold = high_threshold * 0.4

    strong_ridge = (local_max & (dist > high_threshold)).astype(np.uint8)
    weak_ridge = (local_max & (dist > low_threshold)).astype(np.uint8)

    # Keep weak ridge components only if they touch a strong ridge pixel.
    from scipy.ndimage import label as ndimage_label
    labeled_weak, n_features = ndimage_label(weak_ridge)
    for i in range(1, n_features + 1):
        component_mask = labeled_weak == i
        if np.any(strong_ridge[component_mask]):
            strong_ridge[component_mask] = 1

    ridge = strong_ridge
    ridge_thin = skeletonize(ridge).astype(np.uint8)
    combined = np.clip(morph_skel + ridge_thin, 0, 1).astype(np.uint8)
    final = skeletonize(combined).astype(np.uint8)
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit cubic Bezier curves from raster images using skeleton fitting."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--tangent-lookahead",
        type=int,
        default=5,
        help="Lookahead window used for tangent estimation",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=2.0,
        help="Maximum geometric error (pixels) before splitting",
    )
    parser.add_argument(
        "--straightness-scale",
        type=float,
        default=0.75,
        help="Near-straight tolerance scale (higher favors straight cubics)",
    )
    parser.add_argument(
        "--merge-radius",
        type=float,
        default=5.0,
        help="Maximum pixel distance to merge endpoints",
    )
    parser.add_argument(
        "--spur-threshold",
        type=float,
        default=25.0,
        help="Maximum length of branches to prune",
    )
    args = parser.parse_args()

    paths, _adjacency = fit_from_image_skeleton(
        args.image_path,
        max_error=args.max_error,
        tangent_lookahead=args.tangent_lookahead,
        straightness_scale=args.straightness_scale,
        merge_radius=args.merge_radius,
        spur_threshold=args.spur_threshold,
    )
    label = f"Skeleton fitting: {args.image_path}"

    _print_summary(paths, label)

    if paths:
        visualize_paths(paths, title=label, save_path=os.path.join(OUTPUT_DIR, "bezier_vis.png"))
