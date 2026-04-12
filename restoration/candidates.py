"""Phase 2 — Candidate Connection Generation.

KD-tree endpoint search, good-continuation and extension-intersection tests,
with support for both straight and curved extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.extraction import EndpointInfo, ExtractionResult


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectionCandidate:
    """One possible bridge between two endpoints."""

    id: int
    ep_a: EndpointInfo
    ep_b: EndpointInfo
    scenario: str              # "continuation" | "extension_intersection"
    bridge_points: np.ndarray  # (N, 2) sampled bridge
    bridge_bezier: List[BezierSegment]
    distance: float
    tier: int = 1              # 1 = normal, 2 = extended-radius
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _ray_point_distance(origin: np.ndarray, direction: np.ndarray,
                         point: np.ndarray) -> Tuple[float, float]:
    """Distance from *point* to the ray (origin, direction).

    Returns (perpendicular_distance, parameter_along_ray).
    """
    diff = point - origin
    t = float(np.dot(diff, direction))
    if t < 0:
        return float(np.linalg.norm(diff)), t
    proj = origin + t * direction
    return float(np.linalg.norm(point - proj)), t


def _line_line_intersection(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersection of two rays (2D). Returns None if parallel."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-10:
        return None
    diff = p2 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / det
    s = (diff[0] * d1[1] - diff[1] * d1[0]) / det
    # Both parameters must be positive (intersection ahead of both rays)
    if t < 0 or s < 0:
        return None
    return p1 + t * d1


def _extrapolate_bezier(
    cp: np.ndarray, direction: str, steps: int = 10, dt: float = 0.05,
) -> np.ndarray:
    """Sample a cubic Bezier beyond its [0,1] domain.

    direction: "forward" → t = 1+dt, 1+2dt, ...
               "backward" → t = -dt, -2dt, ...
    Returns (steps, 2) array.
    """
    points = []
    for i in range(1, steps + 1):
        if direction == "forward":
            t = 1.0 + i * dt
        else:
            t = -i * dt
        u = 1.0 - t
        pt = (
            u**3 * cp[0]
            + 3 * u**2 * t * cp[1]
            + 3 * u * t**2 * cp[2]
            + t**3 * cp[3]
        )
        points.append(pt)
    return np.array(points, dtype=np.float64)


def _nearest_approach(pts_a: np.ndarray, pts_b: np.ndarray,
                       threshold: float) -> Optional[np.ndarray]:
    """Find the point of nearest approach between two point sets.

    Returns the midpoint if the minimum distance < threshold, else None.
    """
    from scipy.spatial.distance import cdist
    D = cdist(pts_a, pts_b)
    min_idx = np.unravel_index(D.argmin(), D.shape)
    if D[min_idx] < threshold:
        return (pts_a[min_idx[0]] + pts_b[min_idx[1]]) / 2.0
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Scenario tests
# ═══════════════════════════════════════════════════════════════════════════

def _test_good_continuation(
    ep_a: EndpointInfo, ep_b: EndpointInfo, tolerance: float,
) -> bool:
    """True if A's tangent ray passes within *tolerance* of B's position."""
    perp_dist, t = _ray_point_distance(
        ep_a.position, ep_a.tangent, ep_b.position,
    )
    return perp_dist < tolerance and t > 0


def _test_extension_intersection_linear(
    ep_a: EndpointInfo, ep_b: EndpointInfo, max_reach: float,
) -> Optional[np.ndarray]:
    """Line-line intersection of two tangent rays. Returns point or None."""
    pt = _line_line_intersection(
        ep_a.position, ep_a.tangent,
        ep_b.position, ep_b.tangent,
    )
    if pt is None:
        return None
    # Intersection must be within reasonable reach of both endpoints
    if (np.linalg.norm(pt - ep_a.position) > max_reach
            or np.linalg.norm(pt - ep_b.position) > max_reach):
        return None
    return pt


def _test_extension_intersection_curved(
    ep_a: EndpointInfo, ep_b: EndpointInfo,
    path_a: BezierPath, path_b: BezierPath,
    convergence_threshold: float,
) -> Optional[np.ndarray]:
    """Extrapolate both curves beyond their endpoints, find convergence."""
    # Get the relevant terminal segment
    if ep_a.end == "end":
        cp_a = path_a.segments[-1].control_points
        dir_a = "forward"
    else:
        cp_a = path_a.segments[0].control_points
        dir_a = "backward"

    if ep_b.end == "end":
        cp_b = path_b.segments[-1].control_points
        dir_b = "forward"
    else:
        cp_b = path_b.segments[0].control_points
        dir_b = "backward"

    ext_a = _extrapolate_bezier(cp_a, dir_a, steps=15, dt=0.05)
    ext_b = _extrapolate_bezier(cp_b, dir_b, steps=15, dt=0.05)

    return _nearest_approach(ext_a, ext_b, convergence_threshold)


# ═══════════════════════════════════════════════════════════════════════════
# Bridge construction (preliminary — refined in synthesis.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_continuation_bridge(
    ep_a: EndpointInfo, ep_b: EndpointInfo,
) -> List[BezierSegment]:
    """G1-continuous cubic Bezier bridge between two endpoints."""
    p0 = ep_a.position.copy()
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        chord = 1.0

    alpha_a = chord / 3.0
    alpha_b = chord / 3.0
    if ep_a.curvature > 1e-6:
        alpha_a = min(alpha_a, 1.0 / (3.0 * ep_a.curvature))
    if ep_b.curvature > 1e-6:
        alpha_b = min(alpha_b, 1.0 / (3.0 * ep_b.curvature))

    p1 = p0 + alpha_a * ep_a.tangent
    p2 = p3 + alpha_b * (-ep_b.tangent)

    cp = np.vstack([p0, p1, p2, p3])
    seg = BezierSegment(control_points=cp, source_type="bridge")
    return [seg]


def _build_intersection_bridge(
    ep_a: EndpointInfo, ep_b: EndpointInfo, intersection: np.ndarray,
    path_a: Optional[BezierPath] = None,
    path_b: Optional[BezierPath] = None,
) -> List[BezierSegment]:
    """Two-segment bridge through an intersection/convergence point."""
    segments: List[BezierSegment] = []
    I = intersection.copy()

    for ep, target, path, is_first in [
        (ep_a, I, path_a, True),
        (EndpointInfo(
            endpoint_id=-1,
            path_index=ep_b.path_index, end=ep_b.end,
            position=I, tangent=_safe_normalize(ep_b.position - I),
            curvature=0.0,
        ), ep_b.position, path_b, False),
    ]:
        p0 = ep.position.copy()
        p3 = target.copy()
        chord = float(np.linalg.norm(p3 - p0))
        if chord < 1e-6:
            continue

        alpha = chord / 3.0
        if is_first:
            if ep_a.curvature > 1e-6:
                alpha_start = min(alpha, 1.0 / (3.0 * ep_a.curvature))
            else:
                alpha_start = alpha
            p1 = p0 + alpha_start * ep_a.tangent
            p2 = p3 - alpha * _safe_normalize(p3 - p0)
        else:
            p1 = p0 + alpha * _safe_normalize(p3 - p0)
            if ep_b.curvature > 1e-6:
                alpha_end = min(alpha, 1.0 / (3.0 * ep_b.curvature))
            else:
                alpha_end = alpha
            p2 = p3 + alpha_end * (-ep_b.tangent)

        cp = np.vstack([p0, p1, p2, p3])
        segments.append(BezierSegment(control_points=cp, source_type="bridge"))

    return segments


def _sample_bridge(segments: List[BezierSegment], n: int = 50) -> np.ndarray:
    """Sample a bridge composed of one or more BezierSegments."""
    if not segments:
        return np.empty((0, 2))
    parts = [seg.sample(n) for seg in segments]
    return np.vstack(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_candidates(
    result: ExtractionResult,
    lookahead_fraction: float = 0.15,
    max_per_endpoint: int = 5,
) -> List[ConnectionCandidate]:
    """Generate connection candidates for all open endpoints.

    Returns candidates sorted by distance (proximity-first).
    """
    endpoints = result.endpoints
    if len(endpoints) < 2:
        return []

    positions = np.array([ep.position for ep in endpoints])
    tree = cKDTree(positions)
    radius_t1 = result.diagonal * lookahead_fraction
    radius_t2 = result.diagonal * lookahead_fraction * 2.0
    continuation_tolerance = max(10.0, radius_t1 * 0.15)
    convergence_threshold = max(8.0, radius_t1 * 0.12)

    candidates: List[ConnectionCandidate] = []
    cid = 0

    # Track per-endpoint candidate counts for the cap
    ep_candidate_count: Dict[int, int] = {i: 0 for i in range(len(endpoints))}
    seen_pairs: set = set()

    # Query all pairs within Tier 2 radius (superset of Tier 1)
    for i, ep_a in enumerate(endpoints):
        neighbours = tree.query_ball_point(ep_a.position, radius_t2)
        # Sort by distance
        dists = [(j, float(np.linalg.norm(endpoints[j].position - ep_a.position)))
                 for j in neighbours if j != i]
        dists.sort(key=lambda x: x[1])

        for j, dist in dists:
            ep_b = endpoints[j]

            # Skip same-path endpoints
            if ep_a.path_index == ep_b.path_index:
                continue

            # Skip duplicate pairs
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue

            # Enforce per-endpoint cap
            if ep_candidate_count[i] >= max_per_endpoint:
                break
            if ep_candidate_count[j] >= max_per_endpoint:
                continue

            tier = 1 if dist <= radius_t1 else 2

            path_a = result.paths[ep_a.path_index]
            path_b = result.paths[ep_b.path_index]

            # --- Scenario 1: Good Continuation ---
            tol = continuation_tolerance if tier == 1 else continuation_tolerance * 0.5
            if _test_good_continuation(ep_a, ep_b, tol):
                bridge_segs = _build_continuation_bridge(ep_a, ep_b)
                candidates.append(ConnectionCandidate(
                    id=cid,
                    ep_a=ep_a, ep_b=ep_b,
                    scenario="continuation",
                    bridge_points=_sample_bridge(bridge_segs),
                    bridge_bezier=bridge_segs,
                    distance=dist,
                    tier=tier,
                ))
                cid += 1
                ep_candidate_count[i] += 1
                ep_candidate_count[j] += 1
                seen_pairs.add(pair_key)
                continue  # prefer continuation over intersection

            # --- Scenario 2: Extension Intersection ---
            # Try linear intersection first
            max_reach = radius_t1 * 1.5 if tier == 1 else radius_t2
            intersection = _test_extension_intersection_linear(
                ep_a, ep_b, max_reach,
            )

            # If linear fails, try curved extrapolation
            if intersection is None:
                intersection = _test_extension_intersection_curved(
                    ep_a, ep_b, path_a, path_b,
                    convergence_threshold=convergence_threshold,
                )

            if intersection is not None:
                bridge_segs = _build_intersection_bridge(
                    ep_a, ep_b, intersection, path_a, path_b,
                )
                if bridge_segs:
                    candidates.append(ConnectionCandidate(
                        id=cid,
                        ep_a=ep_a, ep_b=ep_b,
                        scenario="extension_intersection",
                        bridge_points=_sample_bridge(bridge_segs),
                        bridge_bezier=bridge_segs,
                        distance=dist,
                        tier=tier,
                    ))
                    cid += 1
                    ep_candidate_count[i] += 1
                    ep_candidate_count[j] += 1
                    seen_pairs.add(pair_key)

    # Sort by distance
    candidates.sort(key=lambda c: c.distance)
    return candidates
