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
from typing import Dict, List, Optional, Tuple

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


def _fit_cubic_single(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    max_iterations: int = 4,
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

    params = _chord_length_parameterize(points)
    cp = _generate_bezier(points, params, left_tangent, right_tangent)
    err, split_idx = _max_error(points, cp, params)

    if err < max_error:
        return [cp]

    # Try iterative reparameterization
    if err < max_error * 4.0:
        for _ in range(max_iterations):
            params = _reparameterize(points, params, cp)
            cp = _generate_bezier(points, params, left_tangent, right_tangent)
            err, split_idx = _max_error(points, cp, params)
            if err < max_error:
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
        points[: split_idx + 1], left_tangent, center_tangent, max_error
    )
    right_curves = _fit_cubic_single(
        points[split_idx:], -center_tangent, right_tangent, max_error
    )
    return left_curves + right_curves


# ═══════════════════════════════════════════════════════════════════════════
# Contour → Bezier  (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

def _find_corners_angle(
    points: np.ndarray,
    angle_threshold_deg: float = 45.0,
    smoothing_window: int = 3,
) -> List[int]:
    """Detect corner indices using turning-angle analysis.

    Returns indices where the local direction changes by more than
    `angle_threshold_deg`. Inflection points (curvature sign-flip without
    a sharp turn) are NOT flagged as corners.
    """
    if smoothing_window > 1 and len(points) > smoothing_window * 2:
        from scipy.ndimage import uniform_filter1d
        pts = uniform_filter1d(points.astype(np.float64), size=smoothing_window, axis=0)
    else:
        pts = points.astype(np.float64)

    corners = [0]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))

        # Skip inflection points: curvature sign flips but no sharp turn
        w = max(1, smoothing_window)
        if i >= w and i < len(pts) - w:
            def _cross(a, b): return a[0] * b[1] - a[1] * b[0]
            before = _cross(pts[i] - pts[i - w], pts[i + 1] - pts[i])
            after  = _cross(pts[i] - pts[i - 1], pts[i + w] - pts[i])
            if (before * after) < 0:   # inflection, not a corner
                continue

        if angle > angle_threshold_deg:
            corners.append(i)

    corners.append(len(points) - 1)
    return corners

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
        """Detect corners using both RDP simplification and turning-angle analysis.

        RDP gives scale-sensitive corners; angle analysis catches sharp bends
        that RDP may miss. The union of both sets is returned, sorted and
        deduplicated. Inflection points are excluded by _find_corners_angle.
        """
        # --- RDP-based corners (existing approach) ---
        contour_cv = points.reshape(-1, 1, 2).astype(np.float32)
        approx = cv2.approxPolyDP(contour_cv, self.corner_threshold, is_closed)
        approx_pts = np.squeeze(approx).astype(np.float64)
        if approx_pts.ndim == 1:
            approx_pts = approx_pts.reshape(1, -1)
        D = cdist(approx_pts, points, metric='euclidean')
        rdp_indices = set(np.argmin(D, axis=1).tolist())

        # --- Angle-based corners ---
        angle_indices = set(_find_corners_angle(points, angle_threshold_deg=45.0))

        corner_indices = sorted(rdp_indices | angle_indices)
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


def _chain_skeleton_edges(graph) -> List[np.ndarray]:
    """Walk the skeleton graph and merge edges through degree-2 nodes.

    A degree-2 node is an interior pass-through point of a stroke, not a
    junction. Chaining through these collapses each physical stroke into a
    single point array regardless of how many corners it has.

    Traversal starts only at degree-1 (tip) or degree-3+ (junction/branch)
    nodes. Isolated loops (all degree-2) are handled separately.

    Returns
    -------
    List of (N, 2) float64 arrays in (x, y) order, one per logical stroke.
    """
    visited_edges: set = set()
    chains: List[np.ndarray] = []

    def _edge_key(a, b):
        return (min(a, b), max(a, b))

    def _pts_for_edge(u, v):
        """Return edge point array oriented from u toward v, in (x, y)."""
        raw = graph[u][v].get('pts', np.array([]))
        if isinstance(raw, list):
            raw = np.array(raw, dtype=np.float64)
        else:
            raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] < 1:
            # Fall back to node positions
            nu = np.array(graph.nodes[u]['o'], dtype=np.float64)
            nv = np.array(graph.nodes[v]['o'], dtype=np.float64)
            raw = np.vstack([nu, nv])
        # sknw stores (row, col) = (y, x); flip to (x, y)
        raw = raw[:, ::-1]
        # Orient: ensure raw[0] is closer to node u than raw[-1]
        nu = np.array(graph.nodes[u]['o'][::-1], dtype=np.float64)  # (x,y)
        if np.linalg.norm(raw[0] - nu) > np.linalg.norm(raw[-1] - nu):
            raw = raw[::-1]
        return raw

    def _unit(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        if n < 1e-9:
            return np.array([1.0, 0.0])
        return vec / n

    def _concat_parts(parts: List[np.ndarray]) -> np.ndarray:
        chain = parts[0].copy()
        for part in parts[1:]:
            if len(part) == 0:
                continue
            if np.linalg.norm(chain[-1] - part[0]) <= 1.5:
                chain = np.vstack([chain, part[1:]])
            elif np.linalg.norm(chain[-1] - part[-1]) <= 1.5:
                part = part[::-1]
                chain = np.vstack([chain, part[1:]])
            else:
                chain = np.vstack([chain, part])
        return chain

    def _choose_next_neighbor(cur: int, prev: int, nexts: List[int], incoming_vec: np.ndarray) -> int:
        """Select continuation edge with best directional coherence."""
        incoming = _unit(incoming_vec)
        best_score = -np.inf
        best_neighbor = nexts[0]
        for n in nexts:
            cand = _pts_for_edge(cur, n)
            if len(cand) >= 2:
                out_vec = cand[1] - cand[0]
            else:
                out_vec = (
                    np.array(graph.nodes[n]['o'][::-1], dtype=np.float64)
                    - np.array(graph.nodes[cur]['o'][::-1], dtype=np.float64)
                )
            score = float(np.dot(incoming, _unit(out_vec)))
            if score > best_score:
                best_score = score
                best_neighbor = n
        return best_neighbor

    def _walk_chain(start: int, neighbor: int) -> Optional[np.ndarray]:
        key = _edge_key(start, neighbor)
        if key in visited_edges:
            return None

        chain_parts: List[np.ndarray] = []
        prev, cur = start, neighbor
        while True:
            k = _edge_key(prev, cur)
            if k in visited_edges:
                break

            visited_edges.add(k)
            cur_part = _pts_for_edge(prev, cur)
            chain_parts.append(cur_part)

            if graph.degree(cur) != 2:
                break  # stop at tip or junction

            nexts = [n for n in graph.neighbors(cur) if _edge_key(cur, n) not in visited_edges]
            if not nexts:
                break

            if len(cur_part) >= 2:
                incoming_vec = cur_part[-1] - cur_part[-2]
            else:
                incoming_vec = (
                    np.array(graph.nodes[cur]['o'][::-1], dtype=np.float64)
                    - np.array(graph.nodes[prev]['o'][::-1], dtype=np.float64)
                )

            nxt = _choose_next_neighbor(cur, prev, nexts, incoming_vec)
            prev, cur = cur, nxt

        if not chain_parts:
            return None
        return _concat_parts(chain_parts)

    # Walk from every non-degree-2 node
    start_nodes = [n for n in graph.nodes() if graph.degree(n) != 2]
    if not start_nodes:
        start_nodes = list(graph.nodes())  # isolated loops fallback

    for start in start_nodes:
        for neighbor in list(graph.neighbors(start)):
            chain = _walk_chain(start, neighbor)
            if chain is not None and len(chain) >= 2:
                chains.append(chain)

    # Catch any unvisited edges (e.g. isolated loops)
    for s, e in graph.edges():
        if _edge_key(s, e) in visited_edges:
            continue
        chain = _walk_chain(s, e)
        if chain is not None and len(chain) >= 2:
            chains.append(chain)

    return chains


def fit_from_image_skeleton(
    image_path: str,
    max_error: float = 5.0,
    tangent_lookahead: int = 5,
    merge_radius: float = 3.0,
) -> Tuple[List[BezierPath], Dict[int, set]]:
    """End-to-end: raster image → skeleton → graph → fitted cubic Bezier paths.

    Preferred over fit_from_image for line drawings, diagrams, and strokes,
    where findContours would produce two parallel outlines instead of a
    single centreline.

    Returns
    -------
    paths : List[BezierPath]
        Fitted B\u00e9zier paths.
    adjacency : Dict[int, set]
        Mapping path_index → set of path_indices that share a skeleton
        node (i.e. paths that are already connected and should not be
        bridged by the gap detector).
    """
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Binarize with Otsu threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Skeletonize
    skeleton = skeletonize(binary / 255.0).astype(np.uint8)

    # Build graph from skeleton
    graph = sknw.build_sknw(skeleton)

    # Process edges — use chained strokes instead of raw graph edges so that
    # corners within a stroke don't fragment it into multiple BezierPaths.
    chains = _chain_skeleton_edges(graph)

    paths = []
    for pts in chains:
        if len(pts) < 2:
            continue
        lookahead = max(1, int(tangent_lookahead))
        left_tangent  = _estimate_tangent(pts, "start", lookahead=lookahead)
        right_tangent = _estimate_tangent(pts, "end",   lookahead=lookahead)
        cps_list = _fit_cubic_single(pts, left_tangent, right_tangent, max_error ** 2)
        segments = [BezierSegment(cps, source_type="skeleton") for cps in cps_list]
        if segments:
            paths.append(BezierPath(segments, is_closed=False, source_type="skeleton"))

    # Endpoint snapping: pull close endpoints to a shared midpoint
    if merge_radius > 0:
        ends = []  # (path_idx, seg_end: 'start'|'end', cp_array, cp_index)
        for i, p in enumerate(paths):
            ends.append((i, 'start', p.segments[0].control_points,  0))
            ends.append((i, 'end',   p.segments[-1].control_points, 3))
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                ia, _, cp_a, idx_a = ends[a]
                ib, _, cp_b, idx_b = ends[b]
                if ia == ib:
                    continue
                d = np.linalg.norm(cp_a[idx_a] - cp_b[idx_b])
                if 0 < d <= merge_radius:
                    mid = (cp_a[idx_a] + cp_b[idx_b]) / 2.0
                    cp_a[idx_a] = mid
                    cp_b[idx_b] = mid

    adjacency: Dict[int, set] = {}   # no longer skeleton-node-based; kept for API compat
    return paths, adjacency


def merge_nearby_paths(
    paths: List[BezierPath],
    gap_threshold: float = 20.0,
    angle_threshold_deg: float = 60.0,
    small_fragment_max_segments: int = 2,
) -> List[BezierPath]:
    """Chain BezierPaths whose endpoints are spatially close and directionally compatible.

    Two paths are merged when:
      - An endpoint of path A is within `gap_threshold` pixels of an endpoint of path B.
      - The outgoing tangent directions at those endpoints point roughly toward each other
        (their dot product < -(cos of angle_threshold_deg)), meaning the angle between
        their outgoing tangents is > (180 - angle_threshold_deg).

    Merging uses union-find so transitive chains (A→B→C) are handled in one pass.

    Parameters
    ----------
    paths              : list of BezierPath objects to consider.
    gap_threshold      : max endpoint distance (pixels) to consider merging.
    angle_threshold_deg: directional compatibility gate (degrees). Tighter values
                         prevent merging strokes that are parallel-but-close.

    Returns
    -------
    A new list of BezierPath objects with compatible neighbours merged.
    """
    n = len(paths)
    if n == 0:
        return paths

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    def _reverse_path(path: BezierPath) -> BezierPath:
        """Reverse path direction while preserving cubic geometry."""
        reversed_segments = [
            BezierSegment(seg.control_points[[3, 2, 1, 0]].copy(), source_type=seg.source_type)
            for seg in reversed(path.segments)
        ]
        return BezierPath(
            segments=reversed_segments,
            is_closed=path.is_closed,
            source_type=path.source_type,
        )

    def _clone_path(path: BezierPath) -> BezierPath:
        cloned_segments = [
            BezierSegment(seg.control_points.copy(), source_type=seg.source_type)
            for seg in path.segments
        ]
        return BezierPath(
            segments=cloned_segments,
            is_closed=path.is_closed,
            source_type=path.source_type,
        )

    # Build endpoint table: (path_idx, side, point, outward_tangent)
    def _outward_tangent(seg_cp: np.ndarray, side: str) -> np.ndarray:
        if side == 'start':
            t = seg_cp[0] - seg_cp[3]
        else:
            t = seg_cp[3] - seg_cp[0]
        n_ = np.linalg.norm(t)
        return t / n_ if n_ > 1e-9 else np.array([1.0, 0.0])

    endpoints = []
    for i, p in enumerate(paths):
        endpoints.append((i, 'start', p.segments[0].control_points[0],
                          _outward_tangent(p.segments[0].control_points,  'start')))
        endpoints.append((i, 'end',   p.segments[-1].control_points[3],
                          _outward_tangent(p.segments[-1].control_points, 'end')))

    cos_gate = -np.cos(np.radians(angle_threshold_deg))  # negative = facing each other

    for a in range(len(endpoints)):
        for b in range(a + 1, len(endpoints)):
            ia, side_a, pa, ta = endpoints[a]
            ib, side_b, pb, tb = endpoints[b]
            if find(ia) == find(ib):
                continue
            gap = pb - pa
            gap_norm = np.linalg.norm(gap)
            if gap_norm > gap_threshold:
                continue

            facing_each_other = np.dot(ta, tb) <= cos_gate
            if not facing_each_other:
                # Aggressive fallback for tiny orphan fragments.
                small_fragment = min(paths[ia].num_segments, paths[ib].num_segments) <= small_fragment_max_segments
                if not small_fragment:
                    continue
                if gap_norm < 1e-9:
                    union(ia, ib)
                    continue
                gap_dir = gap / gap_norm
                aligns_with_gap = np.dot(ta, gap_dir) > 0.0 and np.dot(tb, -gap_dir) > 0.0
                if not aligns_with_gap:
                    continue

            union(ia, ib)

    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    def _chain_group(group_indices: List[int]) -> BezierPath:
        """Greedily order paths in a group by chaining nearest endpoints."""
        if len(group_indices) == 1:
            return _clone_path(paths[group_indices[0]])
        remaining = [_clone_path(paths[i]) for i in group_indices]
        ordered = [remaining.pop(0)]
        while remaining:
            tail = ordered[-1].segments[-1].control_points[3]
            best_dist, best_idx, flip = float('inf'), 0, False
            for j, cand in enumerate(remaining):
                d_start = np.linalg.norm(tail - cand.segments[0].control_points[0])
                d_end   = np.linalg.norm(tail - cand.segments[-1].control_points[3])
                if d_start < best_dist:
                    best_dist, best_idx, flip = d_start, j, False
                if d_end < best_dist:
                    best_dist, best_idx, flip = d_end,   j, True
            chosen = remaining.pop(best_idx)
            if flip:
                chosen = _reverse_path(chosen)

            # Snap adjoining endpoints for continuity in the merged chain.
            tail_cp = ordered[-1].segments[-1].control_points
            head_cp = chosen.segments[0].control_points
            d = np.linalg.norm(tail_cp[3] - head_cp[0])
            if 0 < d <= gap_threshold:
                mid = (tail_cp[3] + head_cp[0]) / 2.0
                tail_cp[3] = mid
                head_cp[0] = mid

            ordered.append(chosen)
        all_segs = []
        for p in ordered:
            all_segs.extend(p.segments)
        return BezierPath(all_segs, is_closed=False,
                          source_type=paths[group_indices[0]].source_type)

    return [_chain_group(list(idxs)) for idxs in groups.values()]


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
    parser.add_argument(
        "--merge-radius",
        type=float,
        default=12.0,
        help="Endpoint snapping radius for skeleton paths",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=30.0,
        help="Gap threshold for merge_nearby_paths",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=48.0,
        help="Directional threshold for merge_nearby_paths (degrees)",
    )
    parser.set_defaults(skeleton=True)
    args = parser.parse_args()

    if args.skeleton:
        paths, _adjacency = fit_from_image_skeleton(
            args.image_path,
            tangent_lookahead=args.tangent_lookahead,
            merge_radius=args.merge_radius,
        )
        if args.gap_threshold > 0:
            paths = merge_nearby_paths(
                paths,
                gap_threshold=args.gap_threshold,
                angle_threshold_deg=args.angle_threshold,
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
