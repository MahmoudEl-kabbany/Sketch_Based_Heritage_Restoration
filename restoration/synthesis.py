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


def _is_straight(curvature_a: float, curvature_b: float,
                  tangent_a: np.ndarray, tangent_b: np.ndarray) -> bool:
    """Both endpoints are straight and tangents are nearly collinear."""
    both_flat = curvature_a < 0.005 and curvature_b < 0.005
    aligned = abs(float(np.dot(tangent_a, -tangent_b))) > 0.95
    return both_flat and aligned


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
    return BezierSegment(
        control_points=np.vstack([p0, p1, p2, p3]),
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
    p2 = p3 + alpha_b * (-ep_b.tangent)

    return BezierSegment(
        control_points=np.vstack([p0, p1, p2, p3]),
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
        segments.append(BezierSegment(
            control_points=np.vstack([p0, p1, p2, p3]),
            source_type="bridge",
        ))

    # Segment I → B
    p0 = I
    p3 = ep_b.position.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord > 1e-6:
        alpha = chord / 3.0
        p1 = p0 + alpha * _safe_normalize(p3 - p0)
        p2 = p3 + alpha * (-ep_b.tangent)
        segments.append(BezierSegment(
            control_points=np.vstack([p0, p1, p2, p3]),
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
        segments.append(BezierSegment(
            control_points=np.vstack([p0, p1, p2, p3]),
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
        p2 = p3 + alpha_b * (-ep_b.tangent)
        segments.append(BezierSegment(
            control_points=np.vstack([p0, p1, p2, p3]),
            source_type="bridge",
        ))

    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def _synthesize_single(candidate: ConnectionCandidate) -> List[BezierSegment]:
    """Build the final bridge segment(s) for one accepted candidate."""
    ep_a = candidate.ep_a
    ep_b = candidate.ep_b

    if candidate.scenario == "continuation":
        if _is_straight(ep_a.curvature, ep_b.curvature,
                        ep_a.tangent, ep_b.tangent):
            return [_build_straight_bridge(ep_a, ep_b)]
        return [_build_g1_bridge(ep_a, ep_b)]

    if candidate.scenario == "extension_intersection":
        # Recover intersection point from the bridge geometry
        bridge_pts = candidate.bridge_points
        if len(bridge_pts) < 2:
            return [_build_g1_bridge(ep_a, ep_b)]

        # The intersection point is approximately where the two bridge
        # sub-segments meet — use the midpoint index of the bridge samples
        mid = len(bridge_pts) // 2
        intersection = bridge_pts[mid].copy()

        if _is_straight(ep_a.curvature, ep_b.curvature,
                        ep_a.tangent, ep_b.tangent):
            return _build_intersection_bridge_linear(ep_a, ep_b, intersection)
        return _build_intersection_bridge_curved(ep_a, ep_b, intersection)

    # Fallback
    return [_build_g1_bridge(ep_a, ep_b)]


def synthesize_bridges(
    accepted: List[ConnectionCandidate],
) -> List[BezierSegment]:
    """Build all bridge segments for the accepted candidates."""
    all_bridges: List[BezierSegment] = []
    for c in accepted:
        c.bridge_bezier = _synthesize_single(c)
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
    # Build a union-find over path indices
    n = len(original_paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Map: path_index → accumulated segments (in order)
    path_segments: Dict[int, List[BezierSegment]] = {}
    for i, p in enumerate(original_paths):
        path_segments[i] = list(p.segments)

    # Process accepted connections
    connection_order: List[Tuple[int, int, List[BezierSegment], str, str]] = []
    for c in accepted:
        connection_order.append((
            c.ep_a.path_index, c.ep_b.path_index,
            c.bridge_bezier,
            c.ep_a.end, c.ep_b.end,
        ))

    # Build merged paths
    merged_into: Set[int] = set()

    for pa_idx, pb_idx, bridge_segs, end_a, end_b in connection_order:
        if pa_idx in merged_into or pb_idx in merged_into:
            # Already consumed by a prior merge — add bridge as standalone
            path_segments.setdefault(n, []).extend(bridge_segs)
            n += 1
            continue

        # Orient paths so bridge connects end of A to start of B
        segs_a = path_segments[pa_idx]
        segs_b = path_segments[pb_idx]

        if end_a == "start":
            segs_a = _reverse_segments(segs_a)
        if end_b == "end":
            segs_b = _reverse_segments(segs_b)

        new_segs = segs_a + bridge_segs + segs_b
        path_segments[pa_idx] = new_segs
        merged_into.add(pb_idx)
        union(pa_idx, pb_idx)

    # Collect final paths
    result: List[BezierPath] = []
    seen_roots: Set[int] = set()
    for i in range(max(n, len(path_segments))):
        if i in merged_into or i not in path_segments:
            continue
        segs = path_segments[i]
        if not segs:
            continue

        # Detect closure
        start_pt = segs[0].control_points[0]
        end_pt = segs[-1].control_points[3]
        is_closed = float(np.linalg.norm(start_pt - end_pt)) < 5.0

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
