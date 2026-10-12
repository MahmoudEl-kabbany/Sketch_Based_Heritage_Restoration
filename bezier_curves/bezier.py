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
import numpy as np
from skimage.morphology import skeletonize
import sknw

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Sizes used while extracting paths are multiples of the local stroke radius,
# so that thin and thick drawings are treated alike.
#  - A terminal skeleton branch no longer than this many radii does not reach
#    beyond the body of the stroke it hangs off: it is a thinning artefact.
SPUR_RADIUS_FACTOR = 1.5
#  - A longer terminal branch is still an artefact if it runs into a corner of
#    the outline: its radius shrinks to under this share of the radius at its
#    junction, where a real stroke stub keeps its width. Oblique cut ends give
#    such branches, up to this many radii long.
CORNER_TIP_RADIUS_RATIO = 0.5
CORNER_BRANCH_MAX_RADII = 3.0
#  - At a stroke's free end the skeleton leaves the centre line: at a blunt end
#    it runs into a corner of the outline, 1.4 radii of skeleton for one radius
#    of stroke. The skeleton is trusted from this many stroke radii behind the
#    tip, and never within a few pixels of it.
END_TRIM_RADII = 1.5
END_TRIM_MIN_PX = 3.0
#  - Blobs smaller than this many squared line widths are specks, not strokes.
SPECK_AREA_WIDTHS2 = 0.5

# sknw compiles its numba functions without an on-disk cache, which costs a few
# seconds in every new process. Turn the cache on so only the first run compiles.
for _name in ("mark", "idx2rc", "fill", "trace", "parse_struc"):
    _compiled = getattr(sknw.sknw, _name, None)
    if hasattr(_compiled, "enable_caching"):
        _compiled.enable_caching()


def bezier_eval(cp: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Evaluate the cubic Bezier with control points *cp* (4, 2) at parameters *ts* (n,)."""
    t = ts[:, np.newaxis]
    u = 1.0 - t
    return (u**3) * cp[0] + 3 * (u**2) * t * cp[1] + 3 * u * (t**2) * cp[2] + (t**3) * cp[3]


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

    def sample(self, n: int = 50) -> np.ndarray:
        """Return *n* evenly-spaced (x, y) points along the curve."""
        return bezier_eval(self.control_points, np.linspace(0.0, 1.0, n))

    def reverse(self) -> None:
        """Reverse the direction of the control points."""
        self.control_points = self.control_points[::-1].copy()


@dataclass
class BezierPath:
    """An ordered sequence of cubic Bezier segments forming a path."""

    segments: List[BezierSegment] = field(default_factory=list)
    is_closed: bool = False
    source_type: str = "unknown"
    # A free end is where a stroke stops; an end that is not free sits on a
    # junction with other strokes and is connected, not open.
    start_free: bool = True
    end_free: bool = True

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
        self.start_free, self.end_free = self.end_free, self.start_free


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


def _turning_angles_deg(vectors: np.ndarray) -> np.ndarray:
    """Angle between each consecutive pair of vectors (NaN where one is degenerate)."""
    if len(vectors) < 2:
        return np.empty(0, dtype=np.float64)

    a, b = vectors[:-1], vectors[1:]
    na = np.sqrt(np.einsum("ij,ij->i", a, a))
    nb = np.sqrt(np.einsum("ij,ij->i", b, b))
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.einsum("ij,ij->i", a / na[:, np.newaxis], b / nb[:, np.newaxis])
        angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    angles[(na < 1e-12) | (nb < 1e-12)] = np.nan
    return angles


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
    for i in np.flatnonzero(_turning_angles_deg(vectors) > 25.0):
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

    q = bezier_eval(cp, params)  # shape (n, 2)

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


def _fit_error_stats(
    points: np.ndarray, cp: np.ndarray, params: np.ndarray
) -> Tuple[float, float, int]:
    """Return (max squared error, mean squared error, worst index)."""
    errors = np.sum((bezier_eval(cp, params) - points) ** 2, axis=1)
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
    angles = _turning_angles_deg(np.diff(points, axis=0))
    angles = angles[~np.isnan(angles)]
    return float(angles.max()) if len(angles) else 0.0


def _dominant_turn_index(points: np.ndarray, angle_threshold_deg: float = 70.0) -> Optional[int]:
    """Best split index for a sharp corner, or None if no dominant turn exists."""
    if len(points) < 5:
        return None

    # The turn at point i is the angle between the steps into and out of it;
    # the turn at the last interior point is not considered.
    angles = _turning_angles_deg(np.diff(points, axis=0))[:-1]
    angles = np.where(np.isnan(angles), -np.inf, angles)
    if angles.max() <= angle_threshold_deg:
        return None

    best_idx = int(np.argmax(angles)) + 1  # first point with the sharpest turn
    return max(1, min(best_idx, len(points) - 2))


def fit_straight_cubic(p0: np.ndarray, p3: np.ndarray) -> np.ndarray:
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


# Fitting constants (Schneider with a preference for exactly straight segments).
_REPARAM_ITERATIONS = 4
_CORNER_SPLIT_ANGLE_DEG = 72.0
_LINE_PREFERENCE_IMPROVEMENT = 0.15


def _fit_cubic_single(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    straightness_scale: float = 0.75,
) -> List[np.ndarray]:
    """Fit a set of points with one or more cubic Bezier curves (recursive).

    Returns a list of (4, 2) control-point arrays.
    """
    def split_at(idx: int) -> List[np.ndarray]:
        """Fit both sides of points[idx] separately, joined by a shared tangent."""
        center_tangent = points[idx + 1] - points[idx - 1]
        norm = np.linalg.norm(center_tangent)
        if norm < 1e-12:
            center_tangent = np.array([1.0, 0.0])
        else:
            center_tangent = center_tangent / norm
        left_curves = _fit_cubic_single(
            points[: idx + 1], left_tangent, center_tangent, max_error, straightness_scale,
        )
        right_curves = _fit_cubic_single(
            points[idx:], -center_tangent, right_tangent, max_error, straightness_scale,
        )
        return left_curves + right_curves

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
        return [fit_straight_cubic(points[0], points[-1])]

    corner_idx = _dominant_turn_index(points, angle_threshold_deg=_CORNER_SPLIT_ANGLE_DEG)
    if corner_idx is not None:
        return split_at(corner_idx)

    params = _chord_length_parameterize(points)
    straight_cp = fit_straight_cubic(points[0], points[-1])
    straight_err, straight_mean_err, _ = _fit_error_stats(points, straight_cp, params)

    cp = _generate_bezier(points, params, left_tangent, right_tangent)
    err, mean_err, split_idx = _fit_error_stats(points, cp, params)

    # Prefer an exact straight cubic unless the flexible cubic significantly improves fit.
    if straight_err <= max_error * 1.35:
        improvement = (straight_mean_err - mean_err) / max(straight_mean_err, 1e-12)
        if improvement < _LINE_PREFERENCE_IMPROVEMENT:
            return [straight_cp]

    mean_limit = max_error * 0.38

    if err < max_error and mean_err < mean_limit:
        return [cp]

    # Try iterative reparameterization
    if err < max_error * 4.0 or mean_err < mean_limit * 4.0:
        for _ in range(_REPARAM_ITERATIONS):
            params = _reparameterize(points, params, cp)
            cp = _generate_bezier(points, params, left_tangent, right_tangent)
            err, mean_err, split_idx = _fit_error_stats(points, cp, params)
            if err < max_error and mean_err < mean_limit:
                return [cp]

    # Split at the point of maximum error and recurse
    return split_at(max(1, min(split_idx, len(points) - 2)))


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
    start_free: bool = False   # the chain starts at a degree-1 node of the skeleton graph
    end_free: bool = False     # ... and ends at one


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

    is_multi = hasattr(graph, "is_multigraph") and graph.is_multigraph()
    for edge in (graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)):
        u, v, data = edge[0], edge[1], edge[-1]
        pts_arr = np.asarray(data.get("pts", []), dtype=np.float64)
        if pts_arr.ndim != 2 or pts_arr.shape[1] < 2 or len(pts_arr) < 2:
            continue
        pts_xy = pts_arr[:, :2][:, ::-1]
        oriented = _orient_points_between_nodes(pts_xy, _node_xy(graph, u), _node_xy(graph, v))

        eid = len(edges)
        edges[eid] = _SkeletonEdge(edge_id=eid, u=int(u), v=int(v), points_uv=oriented)
        node_to_edges.setdefault(int(u), []).append(eid)
        node_to_edges.setdefault(int(v), []).append(eid)

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

    # A chain that returns to its first node is a loop, whatever else meets there.
    end_node = node_sequence[-1]
    closed = closed or (end_node == start_node and len(chain_points) > 2)
    return _SkeletonChain(
        points=np.asarray(chain_points, dtype=np.float64),
        node_sequence=node_sequence,
        is_closed=closed,
        start_free=not closed and degree_map.get(start_node, 0) == 1,
        end_free=not closed and degree_map.get(end_node, 0) == 1,
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


def _radius_at(radius_map: np.ndarray, xy: np.ndarray, fallback: float) -> float:
    """Local stroke radius (distance to the background) at an (x, y) position."""
    h, w = radius_map.shape
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    if 0 <= x < w and 0 <= y < h and radius_map[y, x] > 0:
        return float(radius_map[y, x])
    return fallback


def _prune_skeleton_spurs(graph: Any, radius_map: np.ndarray, fallback_radius: float) -> None:
    """Remove terminal branches that do not reach beyond the stroke they hang off.

    Thinning leaves short branches at corners, bumps and blunt stroke ends. Such
    a branch is no longer than about one stroke radius at its junction, while a
    real stroke stub sticks out of the stroke's body.
    """
    is_multi = hasattr(graph, "is_multigraph") and graph.is_multigraph()
    changed = True
    while changed:
        changed = False
        degrees = dict(graph.degree())
        edges = graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)

        edges_to_remove = []
        for edge in edges:
            u, v = edge[0], edge[1]
            if degrees.get(u, 0) == 1 and degrees.get(v, 0) >= 3:
                tip, junction = u, v
            elif degrees.get(v, 0) == 1 and degrees.get(u, 0) >= 3:
                tip, junction = v, u
            else:
                continue
            pts = np.asarray(edge[-1].get("pts", []), dtype=np.float64)
            length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
            junction_radius = _radius_at(radius_map, _node_xy(graph, junction), fallback_radius)
            tip_radius = _radius_at(radius_map, _node_xy(graph, tip), fallback_radius)
            runs_into_corner = (
                tip_radius <= CORNER_TIP_RADIUS_RATIO * junction_radius
                and length <= CORNER_BRANCH_MAX_RADII * junction_radius
            )
            if length <= SPUR_RADIUS_FACTOR * junction_radius or runs_into_corner:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            if is_multi:
                graph.remove_edge(edge[0], edge[1], key=edge[2])
            else:
                graph.remove_edge(edge[0], edge[1])
            changed = True


def stroke_radius(points: np.ndarray, radius_map: np.ndarray, fallback: float) -> float:
    """Typical radius of a stroke: the median distance to the background along its centre line."""
    h, w = radius_map.shape
    xy = np.round(points).astype(int)
    values = radius_map[np.clip(xy[:, 1], 0, h - 1), np.clip(xy[:, 0], 0, w - 1)]
    values = values[values > 0]
    return float(np.median(values)) if len(values) else fallback


def _trim_free_ends(chain: _SkeletonChain, radius_map: np.ndarray, fallback_radius: float) -> np.ndarray:
    """Chain points without the stretch at each free end where the skeleton leaves the centre line.

    The stretch is END_TRIM_RADII radii of the stroke long. The stroke's
    radius is taken along the whole chain: at the tip itself the radius says
    nothing, it is zero in a corner. Ends on a junction are kept as they are.
    """
    pts = chain.points
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    max_drop = len(pts) // 4   # never trim away more than a quarter of the chain per end
    reach = max(END_TRIM_RADII * stroke_radius(pts, radius_map, fallback_radius), END_TRIM_MIN_PX)

    def points_within_reach(step_lengths: np.ndarray) -> int:
        return min(int(np.searchsorted(np.cumsum(step_lengths), reach)), max_drop)

    first = points_within_reach(steps) if chain.start_free else 0
    last = points_within_reach(steps[::-1]) if chain.end_free else 0
    return pts[first:len(pts) - last]


def stroke_geometry(image_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Binarize a sketch and measure its strokes.

    Returns (binary, radius_map, line_width): the stroke mask (255 on strokes)
    with specks removed, the distance of every stroke pixel to the background,
    and the typical line width (twice the median radius along the skeleton).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def measure(mask: np.ndarray) -> Tuple[np.ndarray, float]:
        radius_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        on_skeleton = radius_map[skeletonize(mask > 0)]
        return radius_map, (2.0 * float(np.median(on_skeleton)) if len(on_skeleton) else 2.0)

    radius_map, line_width = measure(binary)

    # Remove specks: blobs much smaller than a piece of stroke one line width long.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    specks = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < SPECK_AREA_WIDTHS2 * line_width ** 2]
    if specks:
        binary[np.isin(labels, specks)] = 0
        radius_map, line_width = measure(binary)

    return binary, radius_map, line_width


def fit_from_image_skeleton(
    image_path: str,
    max_error: float = 2.0,
    tangent_lookahead: int = 8,
    straightness_scale: float = 0.75,
    follow_junction_continuation: bool = True,
    junction_min_alignment: float = -0.15,
    geometry: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
) -> List[BezierPath]:
    """End-to-end: raster image → skeleton → graph → fitted cubic Bezier paths.

    Each path records whether its ends are free (the stroke stops there) or
    sit on a junction with other strokes. *geometry* is the result of
    stroke_geometry(image_path) if the caller already has it.
    """
    binary, radius_map, line_width = geometry or stroke_geometry(image_path)
    graph = sknw.build_sknw(skeletonize(binary > 0).astype(np.uint8))

    fallback_radius = line_width / 2.0
    _prune_skeleton_spurs(graph, radius_map, fallback_radius)

    chains = _build_skeleton_chains(
        graph,
        follow_junction_continuation=follow_junction_continuation,
        junction_min_alignment=junction_min_alignment,
    )

    paths: List[BezierPath] = []
    lookahead = max(1, int(tangent_lookahead))
    for chain in chains:
        pts = _trim_free_ends(chain, radius_map, fallback_radius)
        if len(pts) < 2:
            continue

        left_tangent = _estimate_tangent(pts, "start", lookahead=lookahead)
        right_tangent = _estimate_tangent(pts, "end", lookahead=lookahead)
        cps_list = _fit_cubic_single(
            pts,
            left_tangent,
            right_tangent,
            max_error ** 2,
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
                start_free=chain.start_free,
                end_free=chain.end_free,
            )
        )

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

_CURVE_COLORS_BGR = [
    (255, 255, 0),   # neon cyan
    (0, 255, 255),   # bright yellow
    (255, 140, 0),   # bright orange
    (50, 255, 50),   # bright lime
    (255, 60, 255),  # bright magenta
    (100, 200, 255), # light orange-blue tint
]


def _draw_paths_on_canvas(
    shape: Tuple[int, int],
    paths: List[BezierPath],
    pts_per_segment: int,
    show_controls: bool,
) -> np.ndarray:
    """Render Bezier paths, optionally with control handles, on a black BGR canvas."""
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for idx, bp in enumerate(paths):
        color = _CURVE_COLORS_BGR[idx % len(_CURVE_COLORS_BGR)]
        for seg in bp.segments:
            pts = np.round(seg.sample(pts_per_segment)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, color, 2, lineType=cv2.LINE_AA)
            if not show_controls:
                continue
            cp = [tuple(int(v) for v in point) for point in np.round(seg.control_points)]
            cv2.line(canvas, cp[0], cp[1], (180, 180, 180), 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, cp[2], cp[3], (180, 180, 180), 1, lineType=cv2.LINE_AA)
            for point in cp:
                cv2.circle(canvas, point, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return canvas


def visualize_paths(
    paths: List[BezierPath],
    title: str = "Bezier Curves",
    save_path: Optional[str] = None,
    show_controls: bool = True,
    pts_per_segment: int = 80,
    image_path: Optional[str] = None,
) -> None:
    """Plot Bezier paths as side-by-side panels on a black background.

    Panels: the source image (when *image_path* is given), the curves alone,
    and the curves with control points and handles (when *show_controls*).
    The figure is saved to *save_path* if given, otherwise shown.
    """
    import matplotlib.pyplot as plt

    image = cv2.imread(image_path) if image_path else None
    if image is not None:
        shape = image.shape[:2]
    else:
        # No source image: size the canvas to fit the control points.
        extent = np.vstack([p.control_points for p in paths] + [np.ones((1, 2))]).max(axis=0)
        shape = (int(extent[1]) + 20, int(extent[0]) + 20)

    panels = [] if image is None else [("Original Image", image)]
    panels.append((f"Bezier Curves Only ({len(paths)})",
                   _draw_paths_on_canvas(shape, paths, pts_per_segment, show_controls=False)))
    if show_controls:
        panels.append(("Bezier Curves + Controls",
                       _draw_paths_on_canvas(shape, paths, pts_per_segment, show_controls=True)))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
    fig.patch.set_facecolor("black")
    for ax, (name, panel) in zip(axes[0], panels):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(name, color="white")
        ax.axis("off")
    fig.suptitle(title, color="white")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"  -> Saved visualization -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


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
        free = sum((p.start_free, p.end_free)) if not p.is_closed else 0
        print(f"    Path {i}: {p.num_segments} segments ({status}, {free} free end(s))")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit cubic Bezier curves from raster images using skeleton fitting."
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--tangent-lookahead", type=int, default=5,
                        help="Lookahead window used for tangent estimation")
    parser.add_argument("--max-error", type=float, default=2.0,
                        help="Maximum geometric error (pixels) before splitting")
    parser.add_argument("--straightness-scale", type=float, default=0.75,
                        help="Near-straight tolerance scale (higher favors straight cubics)")
    args = parser.parse_args()

    paths = fit_from_image_skeleton(
        args.image_path,
        max_error=args.max_error,
        tangent_lookahead=args.tangent_lookahead,
        straightness_scale=args.straightness_scale,
    )
    label = f"Skeleton fitting: {args.image_path}"

    _print_summary(paths, label)

    if paths:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.image_path))[0]
        visualize_paths(
            paths,
            title=label,
            save_path=os.path.join(OUTPUT_DIR, f"{name}_bezier_vis.png"),
            image_path=args.image_path,
        )
