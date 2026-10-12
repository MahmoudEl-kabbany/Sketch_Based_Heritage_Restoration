"""Phase 5 — Synthesis.

Construct final bridge Bezier curves from accepted candidates, merge
into the restored path set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment, fit_straight_cubic
from restoration.extraction import EndpointInfo
from restoration.candidates import ConnectionCandidate
from restoration.geometry import g1_control_points, polyline_length, safe_normalize


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

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

    def clamp(handle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split a handle into (along-chord, sideways) parts, each kept in range."""
        proj = float(np.dot(handle, chord_dir))
        side = handle - proj * chord_dir
        side_norm = float(np.linalg.norm(side))
        if proj < min_forward:
            proj = rescue_forward
        if side_norm > max_side and side_norm > 1e-12:
            side = side * (max_side / side_norm)
        return proj * chord_dir, side

    along1, side1 = clamp(p1 - p0)
    along2, side2 = clamp(p3 - p2)
    return np.vstack([p0, p0 + along1 + side1, p3 - (along2 + side2), p3])


def _bridge(cp: np.ndarray) -> BezierSegment:
    """Bridge segment from raw control points, with stabilized handles."""
    return BezierSegment(
        control_points=_stabilize_bridge_control_points(cp),
        source_type="bridge",
    )


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
    return _bridge(fit_straight_cubic(ep_a.position, ep_b.position))


def _build_g1_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo) -> BezierSegment:
    """Single cubic Bezier with G1-continuous handles driven by curvature."""
    return _bridge(g1_control_points(
        ep_a.position, ep_a.tangent, ep_a.curvature,
        ep_b.position, ep_b.tangent, ep_b.curvature,
    ))


def _build_intersection_bridge_linear(
    ep_a: EndpointInfo, ep_b: EndpointInfo, intersection: np.ndarray,
) -> List[BezierSegment]:
    """Two straight-ish segments through an intersection point (corner)."""
    segments: List[BezierSegment] = []
    # (start, end, tangent leaving the start endpoint, outward tangent of the end endpoint)
    legs: Tuple[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], ...] = (
        (ep_a.position, intersection, ep_a.tangent, None),
        (intersection, ep_b.position, None, ep_b.tangent),
    )
    for p0, p3, start_tangent, end_tangent in legs:
        chord = float(np.linalg.norm(p3 - p0))
        if chord <= 1e-6:
            continue
        alpha = chord / 3.0
        toward_end = safe_normalize(p3 - p0)
        p1 = p0 + alpha * (toward_end if start_tangent is None else start_tangent)
        if end_tangent is None:
            p2 = p3 - alpha * toward_end
        else:
            # The tangent is outward; the incoming derivative at p3 should align with -tangent.
            p2 = p3 + alpha * end_tangent
        segments.append(_bridge(np.vstack([p0, p1, p2, p3])))

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

    # Two straight strokes that meet at an extension intersection form a sharp
    # corner through that point; everything else gets a smooth G1 bridge.
    if (
        candidate.scenario == "extension_intersection"
        and ep_a.curvature < 0.008
        and ep_b.curvature < 0.008
    ):
        intersection = np.asarray(candidate.intersection_point, dtype=np.float64).reshape(2)
        return _build_intersection_bridge_linear(ep_a, ep_b, intersection)

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
    accepted: List[ConnectionCandidate],
) -> List[BezierPath]:
    """Merge the accepted candidates' bridge segments into the original path set.

    For each accepted connection:
      - If the bridge connects path A's end to path B's start,
        merge them into one path: A.segments + bridge + B.segments
      - Detect if the merged path forms a closed loop

    Returns the new set of paths (some merged, some untouched).
    """
    # A component is a chain of segments plus the tokens of its two free ends.
    # A token is the (path_index, end) of an original open path end; it is None
    # for closed paths and once a bridge has closed the component.
    active: Dict[int, Dict[str, Any]] = {}
    for i, path in enumerate(original_paths):
        is_open = not path.is_closed and bool(path.segments)
        active[i] = {
            "segments": list(path.segments),
            "start_token": (i, "start") if is_open else None,
            "end_token": (i, "end") if is_open else None,
        }

    def component_with(token: Tuple[int, str]) -> Optional[int]:
        return next(
            (cid for cid, comp in active.items()
             if token in (comp["start_token"], comp["end_token"])),
            None,
        )

    def orient(comp: Dict[str, Any], side: str, token: Tuple[int, str]) -> None:
        """Reverse *comp* if needed so that *token* is its *side* end."""
        if comp[side] != token:
            comp["segments"] = _reverse_segments(comp["segments"])
            comp["start_token"], comp["end_token"] = comp["end_token"], comp["start_token"]

    for c in sorted(accepted, key=lambda c: (-c.score, c.distance, c.id)):
        token_a = (c.ep_a.path_index, c.ep_a.end)
        token_b = (c.ep_b.path_index, c.ep_b.end)
        id_a, id_b = component_with(token_a), component_with(token_b)
        if id_a is None or id_b is None or token_a == token_b:
            continue

        comp_a = active[id_a]
        orient(comp_a, "end_token", token_a)
        if id_a == id_b:
            # The bridge closes this component onto itself.
            comp_a["segments"] = comp_a["segments"] + list(c.bridge_bezier)
            comp_a["start_token"] = comp_a["end_token"] = None
            continue

        comp_b = active.pop(id_b)
        orient(comp_b, "start_token", token_b)
        active[id_a] = {
            "segments": comp_a["segments"] + list(c.bridge_bezier) + comp_b["segments"],
            "start_token": comp_a["start_token"],
            "end_token": comp_b["end_token"],
        }

    result: List[BezierPath] = []
    for comp in active.values():
        segs: List[BezierSegment] = comp["segments"]
        if not segs:
            continue

        start_pt = segs[0].control_points[0]
        end_pt = segs[-1].control_points[3]
        path_len = polyline_length(np.vstack([seg.sample(12) for seg in segs]))
        closure_tolerance = max(5.0, 0.01 * path_len)
        is_closed = float(np.linalg.norm(start_pt - end_pt)) < closure_tolerance

        result.append(BezierPath(
            segments=segs,
            is_closed=is_closed,
            source_type="restored",
        ))

    return result


def _reverse_segments(segments: List[BezierSegment]) -> List[BezierSegment]:
    """Reversed copy: segment order and each segment's control points flipped."""
    return [
        BezierSegment(control_points=seg.control_points[::-1].copy(), source_type=seg.source_type)
        for seg in reversed(segments)
    ]
