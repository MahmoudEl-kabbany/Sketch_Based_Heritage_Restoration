"""Phase 2 — Candidate Connection Generation.

KD-tree endpoint search, good-continuation and extension-intersection tests,
with support for both straight and curved extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    score: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _path_length(path: BezierPath, pts_per_segment: int = 20) -> float:
    """Approximate arc length of one path by sampling."""
    if not path.segments:
        return 0.0
    pts = path.sample(pts_per_segment=pts_per_segment)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


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
        min_bilateral = 0.02
        max_misalignment = 90.0
    else:
        min_forward = 0.15
        min_bilateral = 0.10
        max_misalignment = 72.0

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


def _init_candidate_diagnostics(
    diagnostics: Optional[Dict[str, Any]],
    radius_t1: float,
    radius_t2: float,
) -> Optional[Dict[str, Any]]:
    """Initialize optional candidate-generation diagnostics payload.

    Keep this compact: only aggregate counts and a small sample of rejections.
    """
    if diagnostics is None:
        return None

    diagnostics.clear()
    diagnostics.update({
        "radius_px": {
            "tier1": round(float(radius_t1), 2),
            "tier2": round(float(radius_t2), 2),
        },
        "max_rejected_pair_records": 80,
        "same_path": {
            "evaluated_paths": 0,
            "eligible_pairs": 0,
            "generated_pairs": 0,
            "rejection_reason_counts": {},
            "rejected_pairs_total": 0,
            "rejected_pairs": [],
        },
        "cross_path": {
            "pairs_within_tier2": 0,
            "eligible_pairs": 0,
            "generated_pairs": 0,
            "rejection_reason_counts": {},
            "rejected_pairs_total": 0,
            "rejected_pairs": [],
        },
    })
    return diagnostics


def _diag_increment_reason(section: Dict[str, Any], reason: str) -> None:
    """Increment a rejection reason counter in diagnostics."""
    counts = section.setdefault("rejection_reason_counts", {})
    counts[reason] = int(counts.get(reason, 0)) + 1


def _diag_record_rejection(
    section: Dict[str, Any],
    payload: Dict[str, Any],
    max_records: int,
) -> None:
    """Record one rejected pair payload with capped storage."""
    section["rejected_pairs_total"] = int(section.get("rejected_pairs_total", 0)) + 1
    records = section.setdefault("rejected_pairs", [])
    if len(records) < max_records:
        records.append(payload)


def _diag_pair_payload(
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    distance: float,
    reasons: List[str],
    tier: Optional[int] = None,
    forward_a: Optional[float] = None,
    forward_b: Optional[float] = None,
    bilateral: Optional[float] = None,
    misalignment_deg: Optional[float] = None,
    extension_pregate_pass: Optional[bool] = None,
    intersection_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a compact JSON-friendly diagnostics payload for a rejected pair."""
    payload: Dict[str, Any] = {
        "endpoint_a": {
            "path_index": int(ep_a.path_index),
            "end": str(ep_a.end),
        },
        "endpoint_b": {
            "path_index": int(ep_b.path_index),
            "end": str(ep_b.end),
        },
        "distance_px": round(float(distance), 2),
        "reasons": list(reasons),
    }
    if tier is not None:
        payload["tier"] = int(tier)
    if forward_a is not None:
        payload["forward_a"] = round(float(forward_a), 4)
    if forward_b is not None:
        payload["forward_b"] = round(float(forward_b), 4)
    if bilateral is not None:
        payload["bilateral_alignment"] = round(float(bilateral), 4)
    if misalignment_deg is not None:
        payload["misalignment_deg"] = round(float(misalignment_deg), 2)
    if extension_pregate_pass is not None:
        payload["extension_pregate_pass"] = bool(extension_pregate_pass)
    if intersection_available is not None:
        payload["intersection_available"] = bool(intersection_available)
    return payload


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


def _extension_alignment_quality(
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    intersection: np.ndarray,
) -> float:
    """Quality of endpoint-to-intersection alignment for extension closures."""
    dir_a = _safe_normalize(intersection - ep_a.position)
    dir_b = _safe_normalize(intersection - ep_b.position)
    align_a = max(0.0, float(np.dot(ep_a.tangent, dir_a)))
    align_b = max(0.0, float(np.dot(ep_b.tangent, dir_b)))

    dist_a = float(np.linalg.norm(intersection - ep_a.position))
    dist_b = float(np.linalg.norm(intersection - ep_b.position))
    balance = 1.0 - (abs(dist_a - dist_b) / max(dist_a + dist_b, 1e-6))

    conf_a = float(np.clip(getattr(ep_a, "tangent_confidence", 1.0), 0.0, 1.0))
    conf_b = float(np.clip(getattr(ep_b, "tangent_confidence", 1.0), 0.0, 1.0))
    context_conf = min(conf_a, conf_b)

    quality = 0.55 * min(align_a, align_b) + 0.25 * max(0.0, balance) + 0.20 * context_conf

    # Curved-curved endpoints are more prone to spurious straight extensions.
    curvature_product = float(max(ep_a.curvature, 0.0) * max(ep_b.curvature, 0.0))
    curvature_penalty = float(np.clip((curvature_product - 1e-5) / 4e-5, 0.0, 1.0))
    quality *= (1.0 - 0.25 * curvature_penalty)

    return float(np.clip(quality, 0.0, 1.0))


def _is_high_curvature_endpoint(ep: EndpointInfo, threshold: float = 0.008) -> bool:
    """True if endpoint curvature indicates a smooth curved boundary."""
    return float(getattr(ep, "curvature", 0.0)) >= float(threshold)


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
        float(np.clip(getattr(ep_a, "tangent_confidence", 1.0), 0.0, 1.0)),
        float(np.clip(getattr(ep_b, "tangent_confidence", 1.0), 0.0, 1.0)),
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
    self_closure_gap_ratio: float = 0.22,
    diagnostics: Optional[Dict[str, Any]] = None,
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

    diag = _init_candidate_diagnostics(diagnostics, radius_t1, radius_t2)
    max_diag_records = int(diag.get("max_rejected_pair_records", 80)) if diag is not None else 80
    same_diag = diag.get("same_path") if diag is not None else None
    cross_diag = diag.get("cross_path") if diag is not None else None

    candidates: List[ConnectionCandidate] = []
    cid = 0

    # Track per-endpoint candidate counts for the cap
    ep_candidate_count: Dict[int, int] = {i: 0 for i in range(len(endpoints))}
    seen_pairs: Set[Tuple[int, int]] = set()

    endpoint_idx_by_key: Dict[Tuple[int, str], int] = {
        (ep.path_index, ep.end): idx for idx, ep in enumerate(endpoints)
    }

    path_lengths: Dict[int, float] = {
        idx: _path_length(path) for idx, path in enumerate(result.paths)
    }
    spur_length_soft = max(22.0, result.diagonal * 0.008)
    spur_length_hard = max(36.0, result.diagonal * 0.012)
    is_spur_path: Dict[int, bool] = {}
    for idx, path in enumerate(result.paths):
        plen = path_lengths.get(idx, 0.0)
        few_segments = len(path.segments) <= 2
        is_spur_path[idx] = (plen <= spur_length_soft) or (few_segments and plen <= spur_length_hard)

    # Same-path near-closure candidates (critical for broken circles/loops).
    for path_idx, path in enumerate(result.paths):
        if path.is_closed or not path.segments:
            continue
        if same_diag is not None:
            same_diag["evaluated_paths"] = int(same_diag.get("evaluated_paths", 0)) + 1
        idx_start = endpoint_idx_by_key.get((path_idx, "start"))
        idx_end = endpoint_idx_by_key.get((path_idx, "end"))
        if idx_start is None or idx_end is None:
            continue

        ep_start = endpoints[idx_start]
        ep_end = endpoints[idx_end]
        dist = float(np.linalg.norm(ep_end.position - ep_start.position))
        path_len = max(path_lengths.get(path_idx, 0.0), 1e-6)
        gap_ratio = dist / path_len

        same_reasons: List[str] = []
        if gap_ratio > self_closure_gap_ratio:
            same_reasons.append("same_gap_ratio_exceeds_cap")
        if dist > (radius_t1 * 1.25):
            same_reasons.append("same_distance_exceeds_cap")

        if same_reasons:
            if same_diag is not None:
                for reason in same_reasons:
                    _diag_increment_reason(same_diag, reason)
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=same_reasons,
                        tier=1,
                    ),
                    max_records=max_diag_records,
                )
            continue

        if ep_candidate_count[idx_start] >= max_per_endpoint or ep_candidate_count[idx_end] >= max_per_endpoint:
            if same_diag is not None:
                _diag_increment_reason(same_diag, "same_endpoint_cap")
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=["same_endpoint_cap"],
                        tier=1,
                    ),
                    max_records=max_diag_records,
                )
            continue

        if same_diag is not None:
            same_diag["eligible_pairs"] = int(same_diag.get("eligible_pairs", 0)) + 1

        fwd_a, fwd_b, bilateral, misalignment_deg = _pair_alignment_metrics(ep_end, ep_start)
        if not _passes_direction_gates(
            fwd_a,
            fwd_b,
            bilateral,
            misalignment_deg,
            tier=1,
            same_path=True,
            tangent_confidence_a=float(getattr(ep_end, "tangent_confidence", 1.0)),
            tangent_confidence_b=float(getattr(ep_start, "tangent_confidence", 1.0)),
        ):
            if same_diag is not None:
                _diag_increment_reason(same_diag, "same_direction_gate")
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=["same_direction_gate"],
                        tier=1,
                        forward_a=fwd_a,
                        forward_b=fwd_b,
                        bilateral=bilateral,
                        misalignment_deg=misalignment_deg,
                    ),
                    max_records=max_diag_records,
                )
            continue

        if (gap_ratio > 0.095 or dist > 90.0) and bilateral < 0.50 and misalignment_deg > 105.0:
            if same_diag is not None:
                _diag_increment_reason(same_diag, "same_long_gap_guard")
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=["same_long_gap_guard"],
                        tier=1,
                        forward_a=fwd_a,
                        forward_b=fwd_b,
                        bilateral=bilateral,
                        misalignment_deg=misalignment_deg,
                    ),
                    max_records=max_diag_records,
                )
            continue

        if gap_ratio > 0.10 and bilateral < 0.25 and misalignment_deg > 120.0:
            if same_diag is not None:
                _diag_increment_reason(same_diag, "same_corner_guard")
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=["same_corner_guard"],
                        tier=1,
                        forward_a=fwd_a,
                        forward_b=fwd_b,
                        bilateral=bilateral,
                        misalignment_deg=misalignment_deg,
                    ),
                    max_records=max_diag_records,
                )
            continue

        pair_key = (min(idx_start, idx_end), max(idx_start, idx_end))
        if pair_key in seen_pairs:
            if same_diag is not None:
                _diag_increment_reason(same_diag, "same_pair_seen")
                _diag_record_rejection(
                    same_diag,
                    _diag_pair_payload(
                        ep_end,
                        ep_start,
                        dist,
                        reasons=["same_pair_seen"],
                        tier=1,
                    ),
                    max_records=max_diag_records,
                )
            continue

        added_for_pair = False

        # Keep a conservative continuation self-closure candidate as fallback.
        bridge_segs = _build_continuation_bridge(ep_end, ep_start)
        candidates.append(ConnectionCandidate(
            id=cid,
            ep_a=ep_end,
            ep_b=ep_start,
            scenario="self_closure",
            bridge_points=_sample_bridge(bridge_segs),
            bridge_bezier=bridge_segs,
            distance=dist,
            tier=1,
            bilateral_alignment=bilateral,
            misalignment_deg=misalignment_deg,
            spur_involved=is_spur_path.get(path_idx, False),
            same_path_closure=True,
            extension_quality=0.0,
        ))
        cid += 1
        added_for_pair = True

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
            ext_bridge = _build_intersection_bridge(ep_end, ep_start, intersection, path, path)
            if ext_bridge:
                candidates.append(ConnectionCandidate(
                    id=cid,
                    ep_a=ep_end,
                    ep_b=ep_start,
                    scenario="extension_intersection",
                    bridge_points=_sample_bridge(ext_bridge),
                    bridge_bezier=ext_bridge,
                    distance=dist,
                    tier=1,
                    bilateral_alignment=bilateral,
                    misalignment_deg=misalignment_deg,
                    spur_involved=is_spur_path.get(path_idx, False),
                    same_path_closure=True,
                    intersection_point=intersection.copy(),
                    extension_quality=float(ext_quality),
                ))
                cid += 1
                added_for_pair = True

        if added_for_pair:
            ep_candidate_count[idx_start] += 1
            ep_candidate_count[idx_end] += 1
            seen_pairs.add(pair_key)
            if same_diag is not None:
                same_diag["generated_pairs"] = int(same_diag.get("generated_pairs", 0)) + 1

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

            if cross_diag is not None:
                cross_diag["pairs_within_tier2"] = int(cross_diag.get("pairs_within_tier2", 0)) + 1

            # Enforce per-endpoint cap
            if ep_candidate_count[i] >= max_per_endpoint:
                if cross_diag is not None:
                    _diag_increment_reason(cross_diag, "cross_endpoint_cap_source")
                    _diag_record_rejection(
                        cross_diag,
                        _diag_pair_payload(
                            ep_a,
                            ep_b,
                            dist,
                            reasons=["cross_endpoint_cap_source"],
                        ),
                        max_records=max_diag_records,
                    )
                continue
            if ep_candidate_count[j] >= max_per_endpoint:
                if cross_diag is not None:
                    _diag_increment_reason(cross_diag, "cross_endpoint_cap_target")
                    _diag_record_rejection(
                        cross_diag,
                        _diag_pair_payload(
                            ep_a,
                            ep_b,
                            dist,
                            reasons=["cross_endpoint_cap_target"],
                        ),
                        max_records=max_diag_records,
                    )
                continue

            tier = 1 if dist <= radius_t1 else 2
            fwd_a, fwd_b, bilateral, misalignment_deg = _pair_alignment_metrics(ep_a, ep_b)
            spur_involved = is_spur_path.get(ep_a.path_index, False) or is_spur_path.get(ep_b.path_index, False)

            strict_direction_ok = _passes_direction_gates(
                fwd_a,
                fwd_b,
                bilateral,
                misalignment_deg,
                tier=tier,
                same_path=False,
                tangent_confidence_a=float(getattr(ep_a, "tangent_confidence", 1.0)),
                tangent_confidence_b=float(getattr(ep_b, "tangent_confidence", 1.0)),
            )
            min_conf = min(
                float(getattr(ep_a, "tangent_confidence", 1.0)),
                float(getattr(ep_b, "tangent_confidence", 1.0)),
            )
            best_forward = max(fwd_a, fwd_b)
            worst_forward = min(fwd_a, fwd_b)
            extension_pregate_ok = (
                tier == 1
                and not spur_involved
                and dist <= max(80.0, radius_t1 * 0.42)
                and 92.0 <= misalignment_deg <= 165.0
                and best_forward >= 0.28
                and worst_forward >= -0.22
                and bilateral >= -0.22
                and min_conf >= 0.50
            )

            if not strict_direction_ok and not extension_pregate_ok:
                intersection_for_diag = _test_extension_intersection_linear(
                    ep_a,
                    ep_b,
                    radius_t1 * 1.5 if tier == 1 else radius_t2,
                )
                if intersection_for_diag is None:
                    path_a_diag = result.paths[ep_a.path_index]
                    path_b_diag = result.paths[ep_b.path_index]
                    intersection_for_diag = _test_extension_intersection_curved(
                        ep_a,
                        ep_b,
                        path_a_diag,
                        path_b_diag,
                        convergence_threshold=convergence_threshold,
                    )

                if cross_diag is not None:
                    _diag_increment_reason(cross_diag, "cross_direction_gate")
                    reasons = ["cross_direction_gate"]
                    if intersection_for_diag is not None:
                        _diag_increment_reason(cross_diag, "cross_direction_gate_with_intersection")
                        reasons.append("cross_direction_gate_with_intersection")

                    _diag_record_rejection(
                        cross_diag,
                        _diag_pair_payload(
                            ep_a,
                            ep_b,
                            dist,
                            reasons=reasons,
                            tier=tier,
                            forward_a=fwd_a,
                            forward_b=fwd_b,
                            bilateral=bilateral,
                            misalignment_deg=misalignment_deg,
                            extension_pregate_pass=extension_pregate_ok,
                            intersection_available=intersection_for_diag is not None,
                        ),
                        max_records=max_diag_records,
                    )
                continue

            if cross_diag is not None:
                cross_diag["eligible_pairs"] = int(cross_diag.get("eligible_pairs", 0)) + 1

            if spur_involved:
                if dist > radius_t1 * 0.65:
                    if cross_diag is not None:
                        _diag_increment_reason(cross_diag, "cross_spur_distance_gate")
                        _diag_record_rejection(
                            cross_diag,
                            _diag_pair_payload(
                                ep_a,
                                ep_b,
                                dist,
                                reasons=["cross_spur_distance_gate"],
                                tier=tier,
                                forward_a=fwd_a,
                                forward_b=fwd_b,
                                bilateral=bilateral,
                                misalignment_deg=misalignment_deg,
                                extension_pregate_pass=extension_pregate_ok,
                            ),
                            max_records=max_diag_records,
                        )
                    continue
                if bilateral < 0.45 or misalignment_deg > 64.0:
                    if cross_diag is not None:
                        _diag_increment_reason(cross_diag, "cross_spur_geometry_gate")
                        _diag_record_rejection(
                            cross_diag,
                            _diag_pair_payload(
                                ep_a,
                                ep_b,
                                dist,
                                reasons=["cross_spur_geometry_gate"],
                                tier=tier,
                                forward_a=fwd_a,
                                forward_b=fwd_b,
                                bilateral=bilateral,
                                misalignment_deg=misalignment_deg,
                                extension_pregate_pass=extension_pregate_ok,
                            ),
                            max_records=max_diag_records,
                        )
                    continue

            path_a = result.paths[ep_a.path_index]
            path_b = result.paths[ep_b.path_index]

            # --- Scenario 1: Good Continuation ---
            tol = continuation_tolerance if tier == 1 else continuation_tolerance * 0.5
            if strict_direction_ok and _test_good_continuation(ep_a, ep_b, tol) and _test_good_continuation(ep_b, ep_a, tol):
                bridge_segs = _build_continuation_bridge(ep_a, ep_b)
                candidates.append(ConnectionCandidate(
                    id=cid,
                    ep_a=ep_a, ep_b=ep_b,
                    scenario="continuation",
                    bridge_points=_sample_bridge(bridge_segs),
                    bridge_bezier=bridge_segs,
                    distance=dist,
                    tier=tier,
                    bilateral_alignment=bilateral,
                    misalignment_deg=misalignment_deg,
                    spur_involved=spur_involved,
                    extension_quality=0.0,
                ))
                cid += 1
                ep_candidate_count[i] += 1
                ep_candidate_count[j] += 1
                seen_pairs.add(pair_key)
                if cross_diag is not None:
                    cross_diag["generated_pairs"] = int(cross_diag.get("generated_pairs", 0)) + 1
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
                        bilateral_alignment=bilateral,
                        misalignment_deg=misalignment_deg,
                        spur_involved=spur_involved,
                        intersection_point=intersection.copy(),
                        extension_quality=_extension_alignment_quality(ep_a, ep_b, intersection),
                    ))
                    cid += 1
                    ep_candidate_count[i] += 1
                    ep_candidate_count[j] += 1
                    seen_pairs.add(pair_key)
                    if cross_diag is not None:
                        cross_diag["generated_pairs"] = int(cross_diag.get("generated_pairs", 0)) + 1
                elif cross_diag is not None:
                    _diag_increment_reason(cross_diag, "cross_extension_bridge_empty")
                    _diag_record_rejection(
                        cross_diag,
                        _diag_pair_payload(
                            ep_a,
                            ep_b,
                            dist,
                            reasons=["cross_extension_bridge_empty"],
                            tier=tier,
                            forward_a=fwd_a,
                            forward_b=fwd_b,
                            bilateral=bilateral,
                            misalignment_deg=misalignment_deg,
                            extension_pregate_pass=extension_pregate_ok,
                            intersection_available=True,
                        ),
                        max_records=max_diag_records,
                    )
            elif cross_diag is not None:
                _diag_increment_reason(cross_diag, "cross_no_extension_intersection")
                _diag_record_rejection(
                    cross_diag,
                    _diag_pair_payload(
                        ep_a,
                        ep_b,
                        dist,
                        reasons=["cross_no_extension_intersection"],
                        tier=tier,
                        forward_a=fwd_a,
                        forward_b=fwd_b,
                        bilateral=bilateral,
                        misalignment_deg=misalignment_deg,
                        extension_pregate_pass=extension_pregate_ok,
                        intersection_available=False,
                    ),
                    max_records=max_diag_records,
                )

    # Keep geometry-consistent candidates first before scoring.
    candidates.sort(key=lambda c: (c.distance, c.misalignment_deg, -c.bilateral_alignment, c.id))
    return candidates
