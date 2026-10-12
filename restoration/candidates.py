"""Phase 2 — Candidate connections.

Every pair of free stroke ends that lie within reach of each other is a
candidate, in up to two shapes: a smooth bridge that leaves both ends along
their tangents, and a corner through the point where the two tangent rays
meet. Nothing is rejected here on geometric grounds; judging a candidate is
the scorer's job (scoring.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from bezier_curves.bezier import BezierSegment, fit_straight_cubic
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.geometry import g1_control_points

# A free end is paired with free ends no farther away than this share of the
# image diagonal, and with at most this many of the nearest ones.
REACH_DIAGONAL_FRACTION = 0.30
PARTNERS_PER_ENDPOINT = 6
# A corner farther from either end than this many gap lengths is not proposed.
CORNER_MAX_LEG_GAPS = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectionCandidate:
    """One possible bridge between two free stroke ends."""

    id: int
    ep_a: EndpointInfo
    ep_b: EndpointInfo
    shape: str                 # "smooth" | "corner"
    bridge_points: np.ndarray  # (N, 2) sampled bridge, from ep_a to ep_b
    bridge_bezier: List[BezierSegment]
    distance: float            # straight distance between the two ends, in pixels
    same_path_closure: bool = False             # the two ends belong to one stroke
    corner_point: Optional[np.ndarray] = None   # where a corner bridge turns
    # Filled in by scoring.score_candidates:
    features: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, float] = field(default_factory=dict)   # log-odds each feature adds
    probability: float = 0.0   # estimated chance that the bridge redraws a missing stroke
    score: float = 0.0         # log-odds above the acceptance threshold; positive = worth drawing


# ═══════════════════════════════════════════════════════════════════════════
# Bridge shapes
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


def smooth_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo) -> List[BezierSegment]:
    """One cubic that leaves ep_a along its tangent and enters ep_b against its tangent."""
    cp = g1_control_points(
        ep_a.position, ep_a.tangent, ep_a.curvature,
        ep_b.position, ep_b.tangent, ep_b.curvature,
    )
    return [BezierSegment(control_points=_stabilize_bridge_control_points(cp), source_type="bridge")]


def corner_bridge(ep_a: EndpointInfo, ep_b: EndpointInfo, corner: np.ndarray) -> List[BezierSegment]:
    """Two straight segments: ep_a to the corner, the corner to ep_b."""
    legs = ((ep_a.position, corner), (corner, ep_b.position))
    return [
        BezierSegment(control_points=fit_straight_cubic(p, q), source_type="bridge")
        for p, q in legs
        if float(np.linalg.norm(q - p)) > 1e-6
    ]


def tangent_ray_intersection(
    ep_a: EndpointInfo, ep_b: EndpointInfo, max_leg: float,
) -> Optional[np.ndarray]:
    """Point where the two outward tangent rays meet, ahead of both ends.

    None if the rays are parallel, meet behind an end, or meet farther than
    *max_leg* from either end.
    """
    d1, d2 = ep_a.tangent, ep_b.tangent
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-9:
        return None
    diff = ep_b.position - ep_a.position
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / det
    s = (diff[0] * d1[1] - diff[1] * d1[0]) / det
    if not (0.0 < t <= max_leg and 0.0 < s <= max_leg):
        return None
    return ep_a.position + t * d1


def sample_bridge(segments: List[BezierSegment], n: int = 50) -> np.ndarray:
    """Sample a bridge composed of one or more BezierSegments."""
    if not segments:
        return np.empty((0, 2))
    return np.vstack([seg.sample(n) for seg in segments])


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def endpoint_pairs(
    result: ExtractionResult,
    reach_fraction: float = REACH_DIAGONAL_FRACTION,
    partners_per_endpoint: int = PARTNERS_PER_ENDPOINT,
) -> List[Tuple[int, int]]:
    """Index pairs (i < j) of free ends that could be joined.

    Two ends are a pair when one is among the other's nearest partners within
    reach. The two ends of one stroke are a pair whenever they are within reach.
    """
    endpoints = result.endpoints
    if len(endpoints) < 2:
        return []

    positions = np.array([ep.position for ep in endpoints])
    reach = result.diagonal * reach_fraction
    k = min(partners_per_endpoint + 1, len(endpoints))
    distances, neighbours = cKDTree(positions).query(positions, k=k, distance_upper_bound=reach)

    pairs: Set[Tuple[int, int]] = set()
    for i in range(len(endpoints)):
        for distance, j in zip(distances[i], neighbours[i]):
            if j != i and np.isfinite(distance):
                pairs.add((min(i, int(j)), max(i, int(j))))

    ends_of_path: Dict[int, List[int]] = {}
    for i, ep in enumerate(endpoints):
        ends_of_path.setdefault(ep.path_index, []).append(i)
    for members in ends_of_path.values():
        if len(members) == 2 and np.linalg.norm(positions[members[0]] - positions[members[1]]) <= reach:
            pairs.add((members[0], members[1]))

    return sorted(pairs)


def generate_candidates(
    result: ExtractionResult,
    reach_fraction: float = REACH_DIAGONAL_FRACTION,
    partners_per_endpoint: int = PARTNERS_PER_ENDPOINT,
) -> List[ConnectionCandidate]:
    """All candidate bridges between the free ends of an extraction, unscored."""
    endpoints = result.endpoints
    candidates: List[ConnectionCandidate] = []

    for i, j in endpoint_pairs(result, reach_fraction, partners_per_endpoint):
        ep_a, ep_b = endpoints[i], endpoints[j]
        distance = float(np.linalg.norm(ep_b.position - ep_a.position))
        if distance < 1e-6:
            continue
        same_path = ep_a.path_index == ep_b.path_index

        shapes: List[Tuple[str, List[BezierSegment], Optional[np.ndarray]]] = [
            ("smooth", smooth_bridge(ep_a, ep_b), None),
        ]
        corner = tangent_ray_intersection(ep_a, ep_b, max_leg=CORNER_MAX_LEG_GAPS * distance)
        if corner is not None:
            shapes.append(("corner", corner_bridge(ep_a, ep_b, corner), corner))

        for shape, segments, corner_point in shapes:
            if not segments:
                continue
            candidates.append(ConnectionCandidate(
                id=len(candidates),
                ep_a=ep_a,
                ep_b=ep_b,
                shape=shape,
                bridge_points=sample_bridge(segments),
                bridge_bezier=segments,
                distance=distance,
                same_path_closure=same_path,
                corner_point=corner_point,
            ))

    return candidates
