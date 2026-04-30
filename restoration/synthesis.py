"""Phase 5 — Synthesis.

Construct final bridge Bezier curves from accepted candidates, merge
into the restored path set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.candidates import ConnectionCandidate


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _stabilize_bridge_control_points(cp: np.ndarray) -> np.ndarray:
    """Clamp handle geometry to prevent local hook/backtracking artifacts."""
    p0, p1, p2, p3 = cp.astype(np.float64)
    chord_vec = p3 - p0
    chord = float(np.linalg.norm(chord_vec))
    if chord < 1e-6:
        return cp

    chord_dir = chord_vec / chord
    min_forward = 0.02 * chord
    rescue_forward = 0.08 * chord
    max_side = 0.60 * chord

    v1 = p1 - p0
    proj1 = float(np.dot(v1, chord_dir))
    side1 = v1 - proj1 * chord_dir
    side1_norm = float(np.linalg.norm(side1))
    if proj1 < min_forward:
        proj1 = rescue_forward
    if side1_norm > max_side and side1_norm > 1e-12:
        side1 = side1 * (max_side / side1_norm)
    p1_new = p0 + proj1 * chord_dir + side1

    v2 = p3 - p2
    proj2 = float(np.dot(v2, chord_dir))
    side2 = v2 - proj2 * chord_dir
    side2_norm = float(np.linalg.norm(side2))
    if proj2 < min_forward:
        proj2 = rescue_forward
    if side2_norm > max_side and side2_norm > 1e-12:
        side2 = side2 * (max_side / side2_norm)
    p2_new = p3 - (proj2 * chord_dir + side2)

    return np.vstack([p0, p1_new, p2_new, p3])


def _is_straight(curvature_a: float, curvature_b: float,
                  tangent_a: np.ndarray, tangent_b: np.ndarray) -> bool:
    """Both endpoints are straight and tangents are nearly collinear."""
    both_flat = curvature_a < 0.005 and curvature_b < 0.005
    aligned = abs(float(np.dot(tangent_a, -tangent_b))) > 0.95
    return both_flat and aligned


def _is_strongly_straight(curvature_a: float, curvature_b: float,
                           tangent_a: np.ndarray, tangent_b: np.ndarray) -> bool:
    """Phase D: Stricter straight detection for bridge enforcement."""
    both_flat = curvature_a < 0.003 and curvature_b < 0.003
    aligned = abs(float(np.dot(tangent_a, -tangent_b))) > 0.92
    return both_flat and aligned


def _re_estimate_tangent_from_path(path: Optional['BezierPath'], end: str,
                                    fraction: float = 0.15) -> Optional[np.ndarray]:
    """Phase D: Re-estimate endpoint tangent from sampled path body using PCA.

    Uses the last `fraction` of the path's sampled polyline, weighted by
    distance from the endpoint, to get a cleaner tangent than the raw
    Voronoi-derived one.
    """
    if path is None or not path.segments:
        return None
    pts = path.sample(pts_per_segment=20)
    if len(pts) < 6:
        return None

    n = len(pts)
    window = max(3, int(n * fraction))

    if end == "end":
        local = pts[-window:]
        direction_sign = 1.0
    else:
        local = pts[:window][::-1]
        direction_sign = 1.0

    if len(local) < 3:
        return None

    # Distance weights: closer to endpoint = higher weight
    dists_from_end = np.linalg.norm(local - local[0], axis=1)
    max_d = max(float(np.max(dists_from_end)), 1e-6)
    weights = np.exp(-2.0 * dists_from_end / max_d)

    # Weighted PCA for dominant direction
    centroid = np.average(local, axis=0, weights=weights)
    centered = local - centroid
    weighted = centered * weights[:, np.newaxis]
    cov = weighted.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # largest eigenvalue

    # Orient along the path direction (endpoint toward interior)
    chord = local[-1] - local[0]
    if np.dot(principal, chord) < 0:
        principal = -principal

    n = np.linalg.norm(principal)
    if n < 1e-12:
        return None
    return (principal / n).astype(np.float64) * direction_sign


def _build_arc_fitted_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo,
                              path_a: Optional['BezierPath'] = None,
                              path_b: Optional['BezierPath'] = None) -> BezierSegment:
    """Phase D: Arc-fitted bridge using osculating circle interpolation.

    For curved endpoints, estimates the osculating circle from endpoint
    position, tangent and curvature, then samples points along the arc
    and fits a Bezier through them. This produces bridges that naturally
    follow the shape's curvature (e.g., oval gaps).
    """
    p0 = ep_a.position.copy()
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        chord = 1.0

    # Use re-estimated tangents if available (Phase D.3)
    tan_a = ep_a.tangent.copy()
    tan_b = ep_b.tangent.copy()
    re_tan_a = _re_estimate_tangent_from_path(path_a, ep_a.end)
    re_tan_b = _re_estimate_tangent_from_path(path_b, ep_b.end)
    if re_tan_a is not None:
        # Blend: 60% re-estimated, 40% original
        blended = 0.6 * re_tan_a + 0.4 * tan_a
        n = np.linalg.norm(blended)
        if n > 1e-12:
            tan_a = blended / n
    if re_tan_b is not None:
        blended = 0.6 * re_tan_b + 0.4 * tan_b
        n = np.linalg.norm(blended)
        if n > 1e-12:
            tan_b = blended / n

    curv_a = max(ep_a.curvature, 1e-8)
    curv_b = max(ep_b.curvature, 1e-8)
    avg_curv = (curv_a + curv_b) / 2.0
    radius = 1.0 / avg_curv if avg_curv > 1e-6 else chord * 5.0

    # Estimate center of osculating circle from endpoint A
    # Normal is perpendicular to tangent, pointing toward center
    normal_a = np.array([-tan_a[1], tan_a[0]], dtype=np.float64)
    # Decide which side the center is on
    to_b = _safe_normalize(p3 - p0)
    if np.dot(normal_a, to_b) < 0:
        normal_a = -normal_a

    center = p0 + radius * normal_a

    # Sample arc points from center
    angle_a = float(np.arctan2(p0[1] - center[1], p0[0] - center[0]))
    angle_b = float(np.arctan2(p3[1] - center[1], p3[0] - center[0]))

    # Ensure we go the short way around
    diff = angle_b - angle_a
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi

    # If the arc subtends too large an angle, fall back to G1
    if abs(diff) > np.pi * 0.9:
        return _build_g1_bridge(ep_a, ep_b)

    n_samples = max(5, int(abs(diff) * radius / 3.0))
    n_samples = min(n_samples, 20)
    arc_pts = []
    for i in range(n_samples + 1):
        t = i / float(n_samples)
        angle = angle_a + t * diff
        arc_pts.append(center + radius * np.array([np.cos(angle), np.sin(angle)]))
    arc_pts = np.array(arc_pts, dtype=np.float64)

    # Fit a G1-continuous Bezier using the arc tangents at endpoints
    # Handle lengths based on arc length
    arc_len = float(np.sum(np.linalg.norm(np.diff(arc_pts, axis=0), axis=1)))
    alpha = arc_len / 3.0

    p1 = p0 + alpha * tan_a
    p2 = p3 + alpha * tan_b  # tan_b is outward, so this goes backward

    cp = np.vstack([p0, p1, p2, p3])
    cp = _stabilize_bridge_control_points(cp)

    return BezierSegment(control_points=cp, source_type="bridge")


# ═══════════════════════════════════════════════════════════════════════════
# Bridge construction — production quality
# ═══════════════════════════════════════════════════════════════════════════

def _build_straight_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo) -> BezierSegment:
    """Exactly straight cubic Bezier between two endpoints."""
    p0 = ep_a.position.copy()
    p3 = ep_b.position.copy()
    d = p3 - p0
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        d = np.array([1.0, 0.0])
        norm = 1.0
    unit = d / norm
    step = norm / 3.0
    p1 = p0 + unit * step
    p2 = p3 - unit * step
    cp = np.vstack([p0, p1, p2, p3])
    cp = _stabilize_bridge_control_points(cp)
    return BezierSegment(
        control_points=cp,
        source_type="bridge",
    )


def _build_g1_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo) -> BezierSegment:
    """Single cubic Bezier with G1-continuous handles driven by curvature."""
    p0 = ep_a.position.copy()
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        chord = 1.0

    # Handle lengths — curvature-aware scaling
    alpha_a = chord / 3.0
    alpha_b = chord / 3.0
    if ep_a.curvature > 1e-6:
        alpha_a = min(alpha_a, 1.0 / (3.0 * ep_a.curvature))
    if ep_b.curvature > 1e-6:
        alpha_b = min(alpha_b, 1.0 / (3.0 * ep_b.curvature))

    p1 = p0 + alpha_a * ep_a.tangent
    # ep_b.tangent is outward; incoming derivative at p3 should align with -ep_b.tangent.
    p2 = p3 + alpha_b * ep_b.tangent

    cp = np.vstack([p0, p1, p2, p3])
    cp = _stabilize_bridge_control_points(cp)

    return BezierSegment(
        control_points=cp,
        source_type="bridge",
    )


def _build_intersection_bridge_linear(
    ep_a: EndpointInfo, ep_b: EndpointInfo, intersection: np.ndarray,
) -> List[BezierSegment]:
    """Two straight-ish segments through an intersection point (corner)."""
    I = intersection.copy()
    segments: List[BezierSegment] = []

    # Segment A → I
    p0 = ep_a.position.copy()
    p3 = I
    chord = float(np.linalg.norm(p3 - p0))
    if chord > 1e-6:
        alpha = chord / 3.0
        p1 = p0 + alpha * ep_a.tangent
        p2 = p3 - alpha * _safe_normalize(p3 - p0)
        cp = np.vstack([p0, p1, p2, p3])
        cp = _stabilize_bridge_control_points(cp)
        segments.append(BezierSegment(
            control_points=cp,
            source_type="bridge",
        ))

    # Segment I → B
    p0 = I
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord > 1e-6:
        alpha = chord / 3.0
        p1 = p0 + alpha * _safe_normalize(p3 - p0)
        # ep_b.tangent is outward; incoming derivative at p3 should align with -ep_b.tangent.
        p2 = p3 + alpha * ep_b.tangent
        cp = np.vstack([p0, p1, p2, p3])
        cp = _stabilize_bridge_control_points(cp)
        segments.append(BezierSegment(
            control_points=cp,
            source_type="bridge",
        ))

    return segments


def _build_intersection_bridge_curved(
    ep_a: EndpointInfo, ep_b: EndpointInfo, intersection: np.ndarray,
) -> List[BezierSegment]:
    """Two curved segments through a convergence point, preserving curvature."""
    I = intersection.copy()
    segments: List[BezierSegment] = []

    # Segment A → I
    p0 = ep_a.position.copy()
    p3 = I
    chord = float(np.linalg.norm(p3 - p0))
    if chord > 1e-6:
        alpha_a = chord / 3.0
        if ep_a.curvature > 1e-6:
            alpha_a = min(alpha_a, 1.0 / (3.0 * ep_a.curvature))
        p1 = p0 + alpha_a * ep_a.tangent
        # The approach handle toward I follows the chord direction
        # but respects curvature continuity
        approach_dir = _safe_normalize(I - p0)
        p2 = p3 - (chord / 3.0) * approach_dir
        cp = np.vstack([p0, p1, p2, p3])
        cp = _stabilize_bridge_control_points(cp)
        segments.append(BezierSegment(
            control_points=cp,
            source_type="bridge",
        ))

    # Segment I → B
    p0 = I
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord > 1e-6:
        departure_dir = _safe_normalize(p3 - I)
        p1 = p0 + (chord / 3.0) * departure_dir
        alpha_b = chord / 3.0
        if ep_b.curvature > 1e-6:
            alpha_b = min(alpha_b, 1.0 / (3.0 * ep_b.curvature))
        # ep_b.tangent is outward; incoming derivative at p3 should align with -ep_b.tangent.
        p2 = p3 + alpha_b * ep_b.tangent
        cp = np.vstack([p0, p1, p2, p3])
        cp = _stabilize_bridge_control_points(cp)
        segments.append(BezierSegment(
            control_points=cp,
            source_type="bridge",
        ))

    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def _synthesize_single(candidate: ConnectionCandidate,
                       paths: Optional[List['BezierPath']] = None) -> List[BezierSegment]:
    """Build the final bridge segment(s) for one accepted candidate.
    
    Phase D enhancements:
    - Arc-fitted bridges for curved endpoints (osculating circle interpolation)
    - Straight bridge enforcement for angular shapes
    - Tangent re-estimation from path body via PCA
    """
    ep_a = candidate.ep_a
    ep_b = candidate.ep_b
    
    # Resolve source paths for tangent re-estimation
    path_a = paths[ep_a.path_index] if paths and ep_a.path_index < len(paths) else None
    path_b = paths[ep_b.path_index] if paths and ep_b.path_index < len(paths) else None

    # Phase D: Straight bridge enforcement for angular shapes
    if _is_strongly_straight(ep_a.curvature, ep_b.curvature,
                              ep_a.tangent, ep_b.tangent):
        return [_build_straight_bridge(ep_a, ep_b)]

    # Phase D: Arc-fitted bridge for curved endpoints
    both_curved = ep_a.curvature > 0.005 and ep_b.curvature > 0.005
    curvature_consistent = (
        both_curved and
        min(ep_a.curvature, ep_b.curvature) / max(ep_a.curvature, ep_b.curvature) > 0.3
    )

    if candidate.scenario == "continuation":
        if _is_straight(ep_a.curvature, ep_b.curvature,
                        ep_a.tangent, ep_b.tangent):
            return [_build_straight_bridge(ep_a, ep_b)]
        if curvature_consistent:
            return [_build_arc_fitted_bridge(ep_a, ep_b, path_a, path_b)]
        return [_build_g1_bridge(ep_a, ep_b)]

    if candidate.scenario == "extension_intersection":
        intersection_obj = getattr(candidate, "intersection_point", None)
        if intersection_obj is None:
            bridge_pts = candidate.bridge_points
            if len(bridge_pts) < 2:
                if curvature_consistent:
                    return [_build_arc_fitted_bridge(ep_a, ep_b, path_a, path_b)]
                return [_build_g1_bridge(ep_a, ep_b)]
            mid = len(bridge_pts) // 2
            intersection = bridge_pts[mid].copy()
        else:
            intersection = np.asarray(intersection_obj, dtype=np.float64).reshape(2)

        if (
            getattr(candidate, "same_path_closure", False)
            and ep_a.curvature < 0.008
            and ep_b.curvature < 0.008
        ):
            return _build_intersection_bridge_linear(ep_a, ep_b, intersection)

        # Same-path closure with curvature: use arc-fitted bridge
        if getattr(candidate, "same_path_closure", False) and curvature_consistent:
            return [_build_arc_fitted_bridge(ep_a, ep_b, path_a, path_b)]

        # If the intersection point is very close to the chord midpoint,
        # a 2-segment bridge through it would create an unnecessary kink.
        chord = float(np.linalg.norm(ep_b.position - ep_a.position))
        if chord > 1e-6:
            midpoint = (ep_a.position + ep_b.position) / 2.0
            dist_to_mid = float(np.linalg.norm(intersection - midpoint))
            if dist_to_mid < 0.15 * chord:
                if curvature_consistent:
                    return [_build_arc_fitted_bridge(ep_a, ep_b, path_a, path_b)]
                return [_build_g1_bridge(ep_a, ep_b)]

        if _is_straight(ep_a.curvature, ep_b.curvature,
                        ep_a.tangent, ep_b.tangent):
            return _build_intersection_bridge_linear(ep_a, ep_b, intersection)
        return _build_intersection_bridge_curved(ep_a, ep_b, intersection)

    # Fallback
    if curvature_consistent:
        return [_build_arc_fitted_bridge(ep_a, ep_b, path_a, path_b)]
    return [_build_g1_bridge(ep_a, ep_b)]


def synthesize_bridges(
    accepted: List[ConnectionCandidate],
    original_paths: Optional[List[BezierPath]] = None,
) -> List[BezierSegment]:
    """Build all bridge segments for the accepted candidates."""
    all_bridges: List[BezierSegment] = []
    for c in accepted:
        c.bridge_bezier = _synthesize_single(c, paths=original_paths)
        all_bridges.extend(c.bridge_bezier)
    return all_bridges


# ═══════════════════════════════════════════════════════════════════════════
# Path merging
# ═══════════════════════════════════════════════════════════════════════════

def merge_restored_paths(
    original_paths: List[BezierPath],
    bridges: List[BezierSegment],
    accepted: List[ConnectionCandidate],
) -> List[BezierPath]:
    """Merge bridge segments into the original path set.

    For each accepted connection:
      - If the bridge connects path A's end to path B's start,
        merge them into one path: A.segments + bridge + B.segments
      - Detect if the merged path forms a closed loop

    Returns the new set of paths (some merged, some untouched).
    """
    def endpoint_token_from_info(ep: EndpointInfo) -> Tuple[str, int]:
        if getattr(ep, "endpoint_id", -1) >= 0:
            return ("id", int(ep.endpoint_id))
        return ("path_end", int(ep.path_index) * 2 + (1 if ep.end == "end" else 0))

    def reverse_component(comp: Dict[str, object]) -> None:
        comp["segments"] = _reverse_segments(comp["segments"])
        comp["start_token"], comp["end_token"] = comp["end_token"], comp["start_token"]

    def orient_component_to_end(comp: Dict[str, object], token: Tuple[str, int]) -> bool:
        if comp["end_token"] == token:
            return True
        if comp["start_token"] == token:
            reverse_component(comp)
            return True
        return False

    def orient_component_to_start(comp: Dict[str, object], token: Tuple[str, int]) -> bool:
        if comp["start_token"] == token:
            return True
        if comp["end_token"] == token:
            reverse_component(comp)
            return True
        return False

    def rebuild_endpoint_to_component(active: Dict[int, Dict[str, object]]) -> Dict[Tuple[str, int], int]:
        mapping: Dict[Tuple[str, int], int] = {}
        for cid, comp in active.items():
            start_token = comp.get("start_token")
            end_token = comp.get("end_token")
            if start_token is not None:
                mapping[start_token] = cid
            if end_token is not None:
                mapping[end_token] = cid
        return mapping

    def estimate_path_length(segments: List[BezierSegment]) -> float:
        if not segments:
            return 0.0
        pts_parts = [seg.sample(12) for seg in segments]
        pts = np.vstack(pts_parts)
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    endpoint_by_path_end: Dict[Tuple[int, str], Tuple[str, int]] = {}
    for c in accepted:
        endpoint_by_path_end[(c.ep_a.path_index, c.ep_a.end)] = endpoint_token_from_info(c.ep_a)
        endpoint_by_path_end[(c.ep_b.path_index, c.ep_b.end)] = endpoint_token_from_info(c.ep_b)

    active: Dict[int, Dict[str, object]] = {}
    for i, path in enumerate(original_paths):
        if path.is_closed or not path.segments:
            start_token = None
            end_token = None
        else:
            start_token = endpoint_by_path_end.get((i, "start"), ("path_end", i * 2))
            end_token = endpoint_by_path_end.get((i, "end"), ("path_end", i * 2 + 1))

        active[i] = {
            "segments": list(path.segments),
            "start_token": start_token,
            "end_token": end_token,
        }

    endpoint_to_component = rebuild_endpoint_to_component(active)

    def grid_sort_key(c: ConnectionCandidate) -> Tuple[float, float]:
        y = min(c.ep_a.position[1], c.ep_b.position[1])
        x = min(c.ep_a.position[0], c.ep_b.position[0])
        return (round(y / 15.0), x)

    ordered_connections = sorted(accepted, key=grid_sort_key)

    for c in ordered_connections:
        token_a = endpoint_token_from_info(c.ep_a)
        token_b = endpoint_token_from_info(c.ep_b)

        comp_a_id = endpoint_to_component.get(token_a)
        comp_b_id = endpoint_to_component.get(token_b)
        if comp_a_id is None or comp_b_id is None:
            continue

        comp_a = active.get(comp_a_id)
        comp_b = active.get(comp_b_id)
        if comp_a is None or comp_b is None:
            continue

        if comp_a_id == comp_b_id:
            if token_a == token_b:
                continue
            if not orient_component_to_end(comp_a, token_a):
                continue
            if comp_a.get("start_token") != token_b:
                continue

            comp_a["segments"] = comp_a["segments"] + list(c.bridge_bezier)
            comp_a["start_token"] = None
            comp_a["end_token"] = None
            endpoint_to_component = rebuild_endpoint_to_component(active)
            continue

        if not orient_component_to_end(comp_a, token_a):
            continue
        if not orient_component_to_start(comp_b, token_b):
            continue

        merged_segments = comp_a["segments"] + list(c.bridge_bezier) + comp_b["segments"]
        merged_start = comp_a.get("start_token")
        merged_end = comp_b.get("end_token")

        active[comp_a_id] = {
            "segments": merged_segments,
            "start_token": merged_start,
            "end_token": merged_end,
        }
        del active[comp_b_id]
        endpoint_to_component = rebuild_endpoint_to_component(active)

    result: List[BezierPath] = []
    for comp in active.values():
        segs: List[BezierSegment] = comp["segments"]
        if not segs:
            continue

        start_pt = segs[0].control_points[0]
        end_pt = segs[-1].control_points[3]
        path_len = estimate_path_length(segs)
        closure_tolerance = max(5.0, 0.01 * path_len)
        is_closed = float(np.linalg.norm(start_pt - end_pt)) < closure_tolerance

        result.append(BezierPath(
            segments=segs,
            is_closed=is_closed,
            source_type="restored",
        ))

    return result


def _reverse_segments(segments: List[BezierSegment]) -> List[BezierSegment]:
    """Reverse segment order and flip each segment's control points."""
    reversed_segs: List[BezierSegment] = []
    for seg in reversed(segments):
        cp = seg.control_points[::-1].copy()
        reversed_segs.append(BezierSegment(
            control_points=cp, source_type=seg.source_type,
        ))
    return reversed_segs
