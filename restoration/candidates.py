"""Phase 2 — Candidate Connection Generation.

KD-tree endpoint search, good-continuation and extension-intersection tests,
with support for both straight and curved extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from bezier_curves.bezier import BezierPath, BezierSegment, bezier_eval
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.geometry import g1_control_points, polyline_length, safe_normalize


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectionCandidate:
    """One possible bridge between two endpoints."""

    id: int
    ep_a: EndpointInfo
    ep_b: EndpointInfo
    scenario: str              # "continuation" | "extension_intersection" | "self_closure"
    bridge_points: np.ndarray  # (N, 2) sampled bridge
    bridge_bezier: List[BezierSegment]
    distance: float
    tier: int = 1              # 1 = normal, 2 = extended-radius
    bilateral_alignment: float = 0.0
    misalignment_deg: float = 180.0
    spur_involved: bool = False
    same_path_closure: bool = False
    intersection_point: Optional[np.ndarray] = None
    extension_quality: float = 0.0
    relaxed_tier2_extension: bool = False
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pair_alignment_metrics(ep_a: EndpointInfo, ep_b: EndpointInfo) -> Tuple[float, float, float, float]:
    """Directional compatibility metrics for endpoint pairing."""
    direction_ab = ep_b.position - ep_a.position
    n = np.linalg.norm(direction_ab)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 180.0

    dir_ab = direction_ab / n
    forward_a = float(np.dot(ep_a.tangent, dir_ab))
    forward_b = float(np.dot(ep_b.tangent, -dir_ab))
    bilateral = min(forward_a, forward_b)
    anti_parallel = float(np.clip(np.dot(ep_a.tangent, -ep_b.tangent), -1.0, 1.0))
    misalignment = float(np.degrees(np.arccos(anti_parallel)))
    return forward_a, forward_b, bilateral, misalignment


def _passes_direction_gates(
    forward_a: float,
    forward_b: float,
    bilateral: float,
    misalignment_deg: float,
    tier: int,
    same_path: bool = False,
    tangent_confidence_a: float = 1.0,
    tangent_confidence_b: float = 1.0,
) -> bool:
    """Conservative geometric gate to reject ambiguous pairings."""
    if same_path:
        min_forward = -0.05
        min_bilateral = -0.10
        max_misalignment = 120.0
    elif tier == 1:
        min_forward = 0.05
        min_bilateral = 0.20
        max_misalignment = 90.0
        if misalignment_deg < 25.0:
            min_forward = -0.80
            min_bilateral = -0.80
    else:
        min_forward = 0.15
        min_bilateral = 0.25
        max_misalignment = 72.0
        if misalignment_deg < 25.0:
            min_forward = -0.80
            min_bilateral = -0.80

    if forward_a < min_forward or forward_b < min_forward:
        return False
    if bilateral < min_bilateral:
        return False
    if misalignment_deg > max_misalignment:
        same_path_high_conf = (
            same_path
            and misalignment_deg <= 150.0
            and bilateral >= -0.02
            and min(tangent_confidence_a, tangent_confidence_b) >= 0.55
        )
        if not same_path_high_conf:
            return False

    if same_path:
        low_context_conf = min(tangent_confidence_a, tangent_confidence_b) < 0.35
        if low_context_conf and misalignment_deg > 95.0:
            return False

    return True


MAX_REJECTED_PAIR_RECORDS = 80

# Optional per-pair diagnostics fields, in report order, with their rounding.
_PAIR_FIELD_DIGITS = {
    "tier": None,
    "forward_a": 4,
    "forward_b": 4,
    "bilateral_alignment": 4,
    "misalignment_deg": 2,
    "extension_pregate_pass": None,
    "intersection_available": None,
}


def _new_diagnostics(radius_t1: float, radius_t2: float) -> Dict[str, Any]:
    """Empty candidate-generation diagnostics: aggregate counts plus a capped rejection sample."""
    def section(first_counter: str) -> Dict[str, Any]:
        return {
            first_counter: 0,
            "eligible_pairs": 0,
            "generated_pairs": 0,
            "rejection_reason_counts": {},
            "rejected_pairs_total": 0,
            "rejected_pairs": [],
        }

    return {
        "radius_px": {
            "tier1": round(float(radius_t1), 2),
            "tier2": round(float(radius_t2), 2),
        },
        "max_rejected_pair_records": MAX_REJECTED_PAIR_RECORDS,
        "same_path": section("evaluated_paths"),
        "cross_path": section("pairs_within_tier2"),
    }


def _reject(
    section: Dict[str, Any],
    reasons: List[str],
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    distance: float,
    **fields: Any,
) -> None:
    """Count the rejection *reasons* for one endpoint pair and keep a capped sample of it."""
    counts = section["rejection_reason_counts"]
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    section["rejected_pairs_total"] += 1
    if len(section["rejected_pairs"]) >= MAX_REJECTED_PAIR_RECORDS:
        return

    payload: Dict[str, Any] = {
        "endpoint_a": {"path_index": int(ep_a.path_index), "end": str(ep_a.end)},
        "endpoint_b": {"path_index": int(ep_b.path_index), "end": str(ep_b.end)},
        "distance_px": round(float(distance), 2),
        "reasons": list(reasons),
    }
    for key, digits in _PAIR_FIELD_DIGITS.items():
        if key not in fields:
            continue
        value = fields[key]
        if key == "tier":
            payload[key] = int(value)
        elif digits is None:
            payload[key] = bool(value)
        else:
            payload[key] = round(float(value), digits)
    section["rejected_pairs"].append(payload)


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
    i = np.arange(1, steps + 1)
    return bezier_eval(cp, 1.0 + i * dt if direction == "forward" else -i * dt)


def _nearest_approach(pts_a: np.ndarray, pts_b: np.ndarray,
                       threshold: float) -> Optional[np.ndarray]:
    """Find the point of nearest approach between two point sets.

    Returns the midpoint if the minimum distance < threshold, else None.
    """
    D = cdist(pts_a, pts_b)
    min_idx = np.unravel_index(D.argmin(), D.shape)
    if D[min_idx] < threshold:
        return (pts_a[min_idx[0]] + pts_b[min_idx[1]]) / 2.0
    return None


def _extension_alignment_quality(
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    intersection: np.ndarray,
) -> float:
    """Quality of endpoint-to-intersection alignment for extension closures."""
    dir_a = safe_normalize(intersection - ep_a.position)
    dir_b = safe_normalize(intersection - ep_b.position)
    align_a = max(0.0, float(np.dot(ep_a.tangent, dir_a)))
    align_b = max(0.0, float(np.dot(ep_b.tangent, dir_b)))

    dist_a = float(np.linalg.norm(intersection - ep_a.position))
    dist_b = float(np.linalg.norm(intersection - ep_b.position))
    balance = 1.0 - (abs(dist_a - dist_b) / max(dist_a + dist_b, 1e-6))

    conf_a = float(np.clip(ep_a.tangent_confidence, 0.0, 1.0))
    conf_b = float(np.clip(ep_b.tangent_confidence, 0.0, 1.0))
    context_conf = min(conf_a, conf_b)

    quality = 0.55 * min(align_a, align_b) + 0.25 * max(0.0, balance) + 0.20 * context_conf

    # Curved-curved endpoints are more prone to spurious straight extensions.
    curvature_product = float(max(ep_a.curvature, 0.0) * max(ep_b.curvature, 0.0))
    curvature_penalty = float(np.clip((curvature_product - 1e-5) / 4e-5, 0.0, 1.0))
    quality *= (1.0 - 0.25 * curvature_penalty)

    return float(np.clip(quality, 0.0, 1.0))


def _is_high_curvature_endpoint(ep: EndpointInfo, threshold: float = 0.008) -> bool:
    """True if endpoint curvature indicates a smooth curved boundary."""
    return float(ep.curvature) >= float(threshold)


def _distance_to_aabb(point: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> float:
    """Euclidean distance from point to axis-aligned bounding box (0 if inside)."""
    dx = max(float(min_xy[0] - point[0]), 0.0, float(point[0] - max_xy[0]))
    dy = max(float(min_xy[1] - point[1]), 0.0, float(point[1] - max_xy[1]))
    return float(np.hypot(dx, dy))


def _safe_same_path_extension_intersection(
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    path: BezierPath,
    path_length: float,
    gap_distance: float,
    max_reach: float,
    convergence_threshold: float,
    bilateral_alignment: float,
    misalignment_deg: float,
) -> Tuple[Optional[np.ndarray], float]:
    """Return a guarded same-path extension intersection and quality."""
    context_conf = min(
        float(np.clip(ep_a.tangent_confidence, 0.0, 1.0)),
        float(np.clip(ep_b.tangent_confidence, 0.0, 1.0)),
    )
    both_high_curvature = _is_high_curvature_endpoint(ep_a) and _is_high_curvature_endpoint(ep_b)

    if both_high_curvature and bilateral_alignment < 0.35 and context_conf < 0.65:
        return None, 0.0

    # Hard guard: very weak bilateral support with near-opposed tangents is unreliable.
    if bilateral_alignment < 0.28 and misalignment_deg > 118.0:
        return None, 0.0

    # Long-gap same-path extension on weakly aligned endpoints is a common spike artifact.
    gap_ratio = float(gap_distance / max(path_length, 1e-6))
    if (gap_ratio > 0.095 or gap_distance > 90.0) and bilateral_alignment < 0.50 and misalignment_deg > 105.0:
        return None, 0.0

    if both_high_curvature:
        # For curved pairs, prefer curved extrapolation with tighter convergence.
        intersection = _test_extension_intersection_curved(
            ep_a,
            ep_b,
            path,
            path,
            convergence_threshold=convergence_threshold * 0.70,
            steps=25,
            dt=0.04,
        )
        if intersection is None:
            intersection = _test_extension_intersection_linear(ep_a, ep_b, max_reach=max_reach * 0.85)
    else:
        intersection = _test_extension_intersection_linear(ep_a, ep_b, max_reach=max_reach)
        if intersection is None:
            intersection = _test_extension_intersection_curved(
                ep_a,
                ep_b,
                path,
                path,
                convergence_threshold=convergence_threshold * 0.85,
                steps=18,
                dt=0.05,
            )

    if intersection is None:
        return None, 0.0

    dist_a = float(np.linalg.norm(intersection - ep_a.position))
    dist_b = float(np.linalg.norm(intersection - ep_b.position))
    if dist_a < 0.25 * gap_distance or dist_b < 0.25 * gap_distance:
        return None, 0.0
    if dist_a > max_reach or dist_b > max_reach:
        return None, 0.0

    if both_high_curvature:
        max_reasonable = max(1.35 * gap_distance, 0.28 * max(path_length, 1e-6))
    else:
        max_reasonable = max(1.8 * gap_distance, 0.42 * max(path_length, 1e-6))
    if max(dist_a, dist_b) > max_reasonable:
        return None, 0.0

    quality = _extension_alignment_quality(ep_a, ep_b, intersection)

    samples = path.sample(pts_per_segment=25)
    if len(samples) >= 4 and both_high_curvature:
        min_xy = np.min(samples, axis=0)
        max_xy = np.max(samples, axis=0)
        outside = _distance_to_aabb(intersection, min_xy, max_xy)
        if outside > max(8.0, 0.55 * gap_distance):
            return None, 0.0

    if len(samples) >= 12:
        trim = max(3, int(round(len(samples) * 0.12)))
        core = samples[trim:-trim] if len(samples) > (2 * trim) else np.empty((0, 2))
        if len(core) > 0:
            min_core_dist = float(np.min(np.linalg.norm(core - intersection, axis=1)))
            if min_core_dist < max(6.0, 0.012 * max(path_length, 1.0)) and quality < 0.60:
                return None, 0.0

    if quality < 0.25:
        return None, quality

    return intersection, quality


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
    steps: int = 15,
    dt: float = 0.05,
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

    ext_a = _extrapolate_bezier(cp_a, dir_a, steps=steps, dt=dt)
    ext_b = _extrapolate_bezier(cp_b, dir_b, steps=steps, dt=dt)

    return _nearest_approach(ext_a, ext_b, convergence_threshold)


def _find_extension_intersection(
    ep_a: EndpointInfo, ep_b: EndpointInfo,
    path_a: BezierPath, path_b: BezierPath,
    max_reach: float,
    convergence_threshold: float,
) -> Optional[np.ndarray]:
    """Linear tangent-ray intersection, falling back to curved extrapolation."""
    intersection = _test_extension_intersection_linear(ep_a, ep_b, max_reach)
    if intersection is None:
        intersection = _test_extension_intersection_curved(
            ep_a, ep_b, path_a, path_b,
            convergence_threshold=convergence_threshold,
        )
    return intersection


# ═══════════════════════════════════════════════════════════════════════════
# Bridge-body crossing filter
# ═══════════════════════════════════════════════════════════════════════════

def _build_path_samples_cache(
    paths: List[BezierPath],
    pts_per_segment: int = 16,
) -> Dict[int, np.ndarray]:
    """Pre-sample all path bodies for fast proximity tests."""
    cache: Dict[int, np.ndarray] = {}
    for idx, path in enumerate(paths):
        if not path.segments:
            continue
        pts = path.sample(pts_per_segment=pts_per_segment)
        if len(pts) >= 2:
            cache[idx] = pts
    return cache


def _bridge_crosses_paths(
    bridge_points: np.ndarray,
    path_samples: Dict[int, np.ndarray],
    exclude_path_indices: Set[int],
    proximity_px: float = 4.0,
) -> int:
    """Count how many distinct path bodies the bridge passes near.

    A bridge is considered to "cross" a path if any bridge sample point
    is within *proximity_px* of any path body sample point, for paths
    not in *exclude_path_indices*.

    Returns the number of distinct crossed path indices.
    """
    if len(bridge_points) < 2:
        return 0

    crossed: Set[int] = set()
    for path_idx, body_pts in path_samples.items():
        if path_idx in exclude_path_indices:
            continue
        if len(body_pts) < 2:
            continue

        # Quick AABB pre-filter to avoid expensive distance checks.
        bridge_min = np.min(bridge_points, axis=0)
        bridge_max = np.max(bridge_points, axis=0)
        body_min = np.min(body_pts, axis=0)
        body_max = np.max(body_pts, axis=0)
        if (bridge_max[0] + proximity_px < body_min[0]
                or body_max[0] + proximity_px < bridge_min[0]
                or bridge_max[1] + proximity_px < body_min[1]
                or body_max[1] + proximity_px < bridge_min[1]):
            continue

        # Subsample bridge for speed if it has many points.
        bp = bridge_points
        if len(bp) > 30:
            step = max(1, len(bp) // 30)
            bp = bp[::step]

        D = cdist(bp, body_pts)
        if float(np.min(D)) < proximity_px:
            crossed.add(path_idx)
            if len(crossed) >= 2:
                return len(crossed)

    return len(crossed)


# ═══════════════════════════════════════════════════════════════════════════
# Bridge construction (preliminary — refined in synthesis.py)
# ═══════════════════════════════════════════════════════════════════════════

def _build_continuation_bridge(
    ep_a: EndpointInfo, ep_b: EndpointInfo,
) -> List[BezierSegment]:
    """G1-continuous cubic Bezier bridge between two endpoints."""
    cp = g1_control_points(
        ep_a.position, ep_a.tangent, ep_a.curvature,
        ep_b.position, ep_b.tangent, ep_b.curvature,
    )
    return [BezierSegment(control_points=cp, source_type="bridge")]


def _build_intersection_bridge(
    ep_a: EndpointInfo, ep_b: EndpointInfo, intersection: np.ndarray,
) -> List[BezierSegment]:
    """Two-segment bridge through an intersection/convergence point."""
    segments: List[BezierSegment] = []
    for p0, p3, ep, leaves_endpoint in (
        (ep_a.position, intersection, ep_a, True),
        (intersection, ep_b.position, ep_b, False),
    ):
        chord = float(np.linalg.norm(p3 - p0))
        if chord < 1e-6:
            continue

        alpha = chord / 3.0
        alpha_ep = min(alpha, 1.0 / (3.0 * ep.curvature)) if ep.curvature > 1e-6 else alpha
        if leaves_endpoint:
            p1 = p0 + alpha_ep * ep.tangent
            p2 = p3 - alpha * safe_normalize(p3 - p0)
        else:
            p1 = p0 + alpha * safe_normalize(p3 - p0)
            # ep.tangent points outward, into the gap: the handle belongs on that side.
            p2 = p3 + alpha_ep * ep.tangent

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
    self_closure_gap_ratio: float = 0.22,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[ConnectionCandidate]:
    """Generate connection candidates for all open endpoints.

    If *diagnostics* is given it is filled in place with rejection statistics.
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

    diag = _new_diagnostics(radius_t1, radius_t2)
    same_diag, cross_diag = diag["same_path"], diag["cross_path"]

    candidates: List[ConnectionCandidate] = []

    # Track per-endpoint candidate counts for the cap
    ep_candidate_count = [0] * len(endpoints)
    seen_pairs: Set[Tuple[int, int]] = set()

    endpoint_idx_by_key: Dict[Tuple[int, str], int] = {
        (ep.path_index, ep.end): idx for idx, ep in enumerate(endpoints)
    }

    path_lengths = [polyline_length(path.sample(pts_per_segment=20)) for path in result.paths]
    spur_length_soft = max(22.0, result.diagonal * 0.008)
    spur_length_hard = max(36.0, result.diagonal * 0.012)
    is_spur_path = [
        plen <= spur_length_soft or (len(path.segments) <= 2 and plen <= spur_length_hard)
        for path, plen in zip(result.paths, path_lengths)
    ]

    open_path_count = sum(1 for p in result.paths if not p.is_closed)
    sparse_scene = len(endpoints) <= 8 and open_path_count <= 3

    # Pre-sample all path bodies for bridge-crossing checks.
    path_samples_cache = _build_path_samples_cache(result.paths, pts_per_segment=16)
    # Distance threshold for strict (>=1) crossing suppression:
    # bridges shorter than this use the lenient (>=2) threshold.
    crossing_long_threshold = max(120.0, result.diagonal * 0.05)

    # Same-path near-closure candidates (critical for broken circles/loops).
    for path_idx, path in enumerate(result.paths):
        if path.is_closed or not path.segments:
            continue
        same_diag["evaluated_paths"] += 1
        idx_start = endpoint_idx_by_key.get((path_idx, "start"))
        idx_end = endpoint_idx_by_key.get((path_idx, "end"))
        if idx_start is None or idx_end is None:
            continue

        ep_start = endpoints[idx_start]
        ep_end = endpoints[idx_end]
        dist = float(np.linalg.norm(ep_end.position - ep_start.position))
        path_len = max(path_lengths[path_idx], 1e-6)
        gap_ratio = dist / path_len

        reasons: List[str] = []
        if gap_ratio > self_closure_gap_ratio:
            reasons.append("same_gap_ratio_exceeds_cap")
        if dist > (radius_t1 * 1.25):
            reasons.append("same_distance_exceeds_cap")
        if not reasons and (
            ep_candidate_count[idx_start] >= max_per_endpoint
            or ep_candidate_count[idx_end] >= max_per_endpoint
        ):
            reasons.append("same_endpoint_cap")
        if reasons:
            _reject(same_diag, reasons, ep_end, ep_start, dist, tier=1)
            continue

        same_diag["eligible_pairs"] += 1

        fwd_a, fwd_b, bilateral, misalignment_deg = _pair_alignment_metrics(ep_end, ep_start)
        if not _passes_direction_gates(
            fwd_a,
            fwd_b,
            bilateral,
            misalignment_deg,
            tier=1,
            same_path=True,
            tangent_confidence_a=ep_end.tangent_confidence,
            tangent_confidence_b=ep_start.tangent_confidence,
        ):
            reasons.append("same_direction_gate")
        elif (gap_ratio > 0.095 or dist > 90.0) and bilateral < 0.50 and misalignment_deg > 105.0:
            reasons.append("same_long_gap_guard")
        elif gap_ratio > 0.10 and bilateral < 0.25 and misalignment_deg > 120.0:
            reasons.append("same_corner_guard")
        if reasons:
            _reject(
                same_diag, reasons, ep_end, ep_start, dist, tier=1,
                forward_a=fwd_a, forward_b=fwd_b,
                bilateral_alignment=bilateral, misalignment_deg=misalignment_deg,
            )
            continue

        common = dict(
            ep_a=ep_end,
            ep_b=ep_start,
            distance=dist,
            tier=1,
            bilateral_alignment=bilateral,
            misalignment_deg=misalignment_deg,
            spur_involved=is_spur_path[path_idx],
            same_path_closure=True,
        )

        # Keep a conservative continuation self-closure candidate as fallback.
        bridge_segs = _build_continuation_bridge(ep_end, ep_start)
        candidates.append(ConnectionCandidate(
            id=len(candidates),
            scenario="self_closure",
            bridge_points=_sample_bridge(bridge_segs),
            bridge_bezier=bridge_segs,
            **common,
        ))

        # Add same-path extension candidate when a stable convergence point exists.
        max_reach_same = min(radius_t1 * 1.8, max(radius_t1 * 0.75, dist * 1.8))
        intersection, ext_quality = _safe_same_path_extension_intersection(
            ep_end,
            ep_start,
            path,
            path_length=path_len,
            gap_distance=dist,
            max_reach=max_reach_same,
            convergence_threshold=convergence_threshold * 0.85,
            bilateral_alignment=bilateral,
            misalignment_deg=misalignment_deg,
        )
        if intersection is not None:
            ext_bridge = _build_intersection_bridge(ep_end, ep_start, intersection)
            if ext_bridge:
                candidates.append(ConnectionCandidate(
                    id=len(candidates),
                    scenario="extension_intersection",
                    bridge_points=_sample_bridge(ext_bridge),
                    bridge_bezier=ext_bridge,
                    intersection_point=intersection.copy(),
                    extension_quality=float(ext_quality),
                    **common,
                ))

        ep_candidate_count[idx_start] += 1
        ep_candidate_count[idx_end] += 1
        seen_pairs.add((min(idx_start, idx_end), max(idx_start, idx_end)))
        same_diag["generated_pairs"] += 1

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

            cross_diag["pairs_within_tier2"] += 1

            # Enforce per-endpoint cap
            if ep_candidate_count[i] >= max_per_endpoint:
                _reject(cross_diag, ["cross_endpoint_cap_source"], ep_a, ep_b, dist)
                continue
            if ep_candidate_count[j] >= max_per_endpoint:
                _reject(cross_diag, ["cross_endpoint_cap_target"], ep_a, ep_b, dist)
                continue

            tier = 1 if dist <= radius_t1 else 2
            fwd_a, fwd_b, bilateral, misalignment_deg = _pair_alignment_metrics(ep_a, ep_b)
            spur_involved = is_spur_path[ep_a.path_index] or is_spur_path[ep_b.path_index]
            path_a = result.paths[ep_a.path_index]
            path_b = result.paths[ep_b.path_index]
            max_reach = radius_t1 * 1.5 if tier == 1 else radius_t2

            strict_direction_ok = _passes_direction_gates(
                fwd_a, fwd_b, bilateral, misalignment_deg, tier=tier,
            )
            min_conf = min(ep_a.tangent_confidence, ep_b.tangent_confidence)
            best_forward = max(fwd_a, fwd_b)
            worst_forward = min(fwd_a, fwd_b)
            extension_pregate_t1 = (
                tier == 1
                and not spur_involved
                and dist <= max(80.0, radius_t1 * 0.42)
                and 92.0 <= misalignment_deg <= 165.0
                and best_forward >= 0.28
                and worst_forward >= -0.22
                and bilateral >= -0.22
                and min_conf >= 0.50
            )

            # Narrow Tier-2 fallback for sparse straight-edged scenes (e.g., staggered rectangle sides).
            low_curvature_pair = max(ep_a.curvature, ep_b.curvature) <= 0.0045
            extension_pregate_t2 = (
                tier == 2
                and sparse_scene
                and not spur_involved
                and low_curvature_pair
                and dist >= max(140.0, radius_t1 * 0.75)
                and 86.0 <= misalignment_deg <= 112.0
                and best_forward >= 0.90
                and worst_forward >= -0.04
                and bilateral >= -0.04
                and min_conf >= 0.58
            )
            extension_pregate_ok = extension_pregate_t1 or extension_pregate_t2
            relaxed_tier2_extension = (extension_pregate_t2 and not strict_direction_ok)

            # Diagnostics fields shared by every rejection below.
            fields = dict(
                tier=tier,
                forward_a=fwd_a,
                forward_b=fwd_b,
                bilateral_alignment=bilateral,
                misalignment_deg=misalignment_deg,
            )

            if not strict_direction_ok and not extension_pregate_ok:
                reasons = ["cross_direction_gate"]
                has_intersection = False
                if diagnostics is not None:
                    # Only the diagnostics care whether the rejected rays would have met.
                    has_intersection = _find_extension_intersection(
                        ep_a, ep_b, path_a, path_b, max_reach, convergence_threshold,
                    ) is not None
                    if has_intersection:
                        reasons.append("cross_direction_gate_with_intersection")
                _reject(
                    cross_diag, reasons, ep_a, ep_b, dist, **fields,
                    extension_pregate_pass=extension_pregate_ok,
                    intersection_available=has_intersection,
                )
                continue

            cross_diag["eligible_pairs"] += 1
            fields_ext = dict(fields, extension_pregate_pass=extension_pregate_ok)

            if spur_involved:
                if dist > radius_t1 * 0.65:
                    _reject(cross_diag, ["cross_spur_distance_gate"], ep_a, ep_b, dist, **fields_ext)
                    continue
                if bilateral < 0.45 or misalignment_deg > 64.0:
                    _reject(cross_diag, ["cross_spur_geometry_gate"], ep_a, ep_b, dist, **fields_ext)
                    continue

            # --- Scenario 1: Good Continuation ---
            tol = continuation_tolerance if tier == 1 else continuation_tolerance * 0.5
            good_cont = _test_good_continuation(ep_a, ep_b, tol) and _test_good_continuation(ep_b, ep_a, tol)
            parallel_offset = (tier == 1 and misalignment_deg < 35.0)

            extra: Dict[str, Any] = {}
            if strict_direction_ok and (good_cont or parallel_offset):
                scenario = "continuation"  # preferred over intersection
                bridge_segs = _build_continuation_bridge(ep_a, ep_b)
            else:
                # --- Scenario 2: Extension Intersection ---
                scenario = "extension_intersection"
                intersection = _find_extension_intersection(
                    ep_a, ep_b, path_a, path_b, max_reach, convergence_threshold,
                )
                fields = dict(fields_ext, intersection_available=intersection is not None)
                if intersection is None:
                    _reject(cross_diag, ["cross_no_extension_intersection"], ep_a, ep_b, dist, **fields)
                    continue
                bridge_segs = _build_intersection_bridge(ep_a, ep_b, intersection)
                if not bridge_segs:
                    _reject(cross_diag, ["cross_extension_bridge_empty"], ep_a, ep_b, dist, **fields)
                    continue
                extra = dict(
                    intersection_point=intersection.copy(),
                    extension_quality=_extension_alignment_quality(ep_a, ep_b, intersection),
                    relaxed_tier2_extension=relaxed_tier2_extension,
                )

            # Bridge-body crossing filter.
            # Long bridges are suppressed if they cross even 1 path body,
            # since legitimate connections rarely span that far across other shapes.
            # Short bridges tolerate 1 crossing (junction proximity).
            bridge_pts = _sample_bridge(bridge_segs)
            n_crossed = _bridge_crosses_paths(
                bridge_pts, path_samples_cache, {ep_a.path_index, ep_b.path_index},
            )
            if n_crossed >= (1 if dist > crossing_long_threshold else 2):
                _reject(cross_diag, ["cross_bridge_body_crossing"], ep_a, ep_b, dist, **fields)
                continue

            candidates.append(ConnectionCandidate(
                id=len(candidates),
                ep_a=ep_a, ep_b=ep_b,
                scenario=scenario,
                bridge_points=bridge_pts,
                bridge_bezier=bridge_segs,
                distance=dist,
                tier=tier,
                bilateral_alignment=bilateral,
                misalignment_deg=misalignment_deg,
                spur_involved=spur_involved,
                **extra,
            ))
            ep_candidate_count[i] += 1
            ep_candidate_count[j] += 1
            seen_pairs.add(pair_key)
            cross_diag["generated_pairs"] += 1

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(diag)

    # Keep geometry-consistent candidates first before scoring.
    candidates.sort(key=lambda c: (c.distance, c.misalignment_deg, -c.bilateral_alignment, c.id))
    return candidates
