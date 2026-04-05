from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from bezier_curves.bezier import BezierPath


@dataclass
class FeatureBridgeConfig:
    """Configuration for endpoint extraction and gap candidate scoring."""

    max_gap_distance_ratio: float = 0.08
    max_gap_distance_px: float = 140.0
    max_direction_misalignment_deg: float = 72.0
    max_corner_curvature_deg: float = 145.0
    allow_same_endpoint_type_pairing: bool = False
    enforce_endpoint_uniqueness: bool = True
    force_connect_unmatched: bool = True
    min_confidence: float = 0.20
    efd_confidence_threshold: float = 0.68
    symmetry_confidence_threshold: float = 0.60

    # Score weights (must sum approximately to 1.0).
    proximity_weight: float = 0.34
    continuation_weight: float = 0.38
    curvature_weight: float = 0.18
    symmetry_weight: float = 0.10


@dataclass
class EndpointRecord:
    """Terminal point descriptor for an open BezierPath."""

    endpoint_id: int
    path_index: int
    endpoint_type: str  # "start" or "end"
    point: np.ndarray
    outward_tangent: np.ndarray
    curvature_deg: float
    source_type: str = "unknown"


@dataclass
class GapRecord:
    """Candidate link between two endpoints."""

    gap_id: int
    endpoint_a_id: int
    endpoint_b_id: int
    path_a: int
    path_b: int
    distance: float
    angle_a_deg: float
    angle_b_deg: float
    curvature_delta_deg: float
    continuation_score: float
    proximity_score: float
    symmetry_score: float
    geometric_score: float
    confidence: float
    gap_kind: str = "bridge"  # "bridge" | "single_contour_gap"
    suggested_method: str = "bezier"  # "bezier" | "efd" | "symmetry"
    is_forced: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class FeatureBundle:
    """Structured geometric features passed to ASP and restoration stages."""

    endpoints: List[EndpointRecord] = field(default_factory=list)
    gaps: List[GapRecord] = field(default_factory=list)
    symmetry_axis: Optional[Tuple[np.ndarray, np.ndarray]] = None
    global_symmetry_confidence: float = 0.0


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (vec / n).astype(np.float64)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = _safe_normalize(u)
    vv = _safe_normalize(v)
    dot = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _estimate_endpoint_curvature(control_points: np.ndarray, endpoint_type: str) -> float:
    """Approximate local curvature from control polygon turning angle."""
    cp = control_points.astype(np.float64)
    if endpoint_type == "start":
        v1 = cp[1] - cp[0]
        v2 = cp[2] - cp[1]
    else:
        v1 = cp[2] - cp[3]
        v2 = cp[1] - cp[2]

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return _angle_deg(v1, v2)


def _extract_endpoints(paths: List[BezierPath]) -> List[EndpointRecord]:
    endpoints: List[EndpointRecord] = []
    endpoint_id = 0

    for path_index, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue

        first_cp = path.segments[0].control_points
        last_cp = path.segments[-1].control_points

        p_start = first_cp[0].astype(np.float64)
        t_start_in = _safe_normalize(first_cp[1] - first_cp[0])
        start_endpoint = EndpointRecord(
            endpoint_id=endpoint_id,
            path_index=path_index,
            endpoint_type="start",
            point=p_start,
            outward_tangent=-t_start_in,
            curvature_deg=_estimate_endpoint_curvature(first_cp, "start"),
            source_type=path.source_type,
        )
        endpoints.append(start_endpoint)
        endpoint_id += 1

        p_end = last_cp[3].astype(np.float64)
        t_end_out = _safe_normalize(last_cp[3] - last_cp[2])
        end_endpoint = EndpointRecord(
            endpoint_id=endpoint_id,
            path_index=path_index,
            endpoint_type="end",
            point=p_end,
            outward_tangent=t_end_out,
            curvature_deg=_estimate_endpoint_curvature(last_cp, "end"),
            source_type=path.source_type,
        )
        endpoints.append(end_endpoint)
        endpoint_id += 1

    return endpoints


def _estimate_global_symmetry(paths: List[BezierPath]) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], float]:
    sampled_parts: List[np.ndarray] = []
    for path in paths:
        if not path.segments:
            continue
        sampled_parts.append(path.sample(40))

    if not sampled_parts:
        return None, 0.0

    points = np.vstack(sampled_parts)
    if len(points) < 3:
        return None, 0.0

    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = _safe_normalize(eig_vecs[:, int(np.argmax(eig_vals))])

    # Symmetry confidence from mirror reconstruction error.
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    signed = np.dot(centered, normal)
    reflected = points - 2.0 * signed[:, None] * normal[None, :]

    dmat = cdist(reflected, points)
    nearest = np.min(dmat, axis=1)
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) + 1e-6
    err = float(np.mean(nearest))
    conf = float(np.exp(-err / (0.06 * diag)))

    return (centroid, axis), conf


def _pair_allowed(a: EndpointRecord, b: EndpointRecord, cfg: FeatureBridgeConfig) -> bool:
    if not cfg.allow_same_endpoint_type_pairing and a.endpoint_type == b.endpoint_type:
        return False
    if a.endpoint_type == b.endpoint_type and cfg.allow_same_endpoint_type_pairing:
        return True

    # Cross-endpoint pairing only by default.
    return {a.endpoint_type, b.endpoint_type} == {"start", "end"}


def _reflect_point_on_axis(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    rel = point - origin
    signed = float(np.dot(rel, normal))
    return point - 2.0 * signed * normal


def _build_initial_candidates(
    endpoints: List[EndpointRecord],
    adjacency: Dict[int, set],
    cfg: FeatureBridgeConfig,
    symmetry_axis: Optional[Tuple[np.ndarray, np.ndarray]],
    efd_data: Dict[int, np.ndarray],
    image_diag: float,
) -> List[GapRecord]:
    candidates: List[GapRecord] = []
    max_dist = min(cfg.max_gap_distance_px, max(4.0, cfg.max_gap_distance_ratio * image_diag))
    dir_cos_threshold = float(np.cos(np.radians(cfg.max_direction_misalignment_deg)))

    gap_id = 0
    for i in range(len(endpoints)):
        a = endpoints[i]
        if a.curvature_deg > cfg.max_corner_curvature_deg:
            continue

        for j in range(i + 1, len(endpoints)):
            b = endpoints[j]
            if b.curvature_deg > cfg.max_corner_curvature_deg:
                continue

            if not _pair_allowed(a, b, cfg):
                continue

            if b.path_index in adjacency.get(a.path_index, set()):
                continue

            vec = b.point - a.point
            dist = float(np.linalg.norm(vec))
            if dist < 1e-9 or dist > max_dist:
                continue

            unit = vec / dist
            align_a = float(np.dot(a.outward_tangent, unit))
            align_b = float(np.dot(b.outward_tangent, -unit))
            if align_a < dir_cos_threshold or align_b < dir_cos_threshold:
                continue

            angle_a = _angle_deg(a.outward_tangent, unit)
            angle_b = _angle_deg(b.outward_tangent, -unit)
            continuation = float(np.clip((align_a + align_b) * 0.5, 0.0, 1.0))
            proximity = float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

            curvature_delta = abs(a.curvature_deg - b.curvature_deg)
            curvature_score = float(np.clip(1.0 - curvature_delta / 180.0, 0.0, 1.0))

            symmetry_score = 0.0
            if symmetry_axis is not None:
                center, axis = symmetry_axis
                reflected_a = _reflect_point_on_axis(a.point, center, axis)
                sym_err = float(np.linalg.norm(reflected_a - b.point))
                symmetry_score = float(np.exp(-sym_err / (0.5 * max_dist + 1e-6)))

            penalty = (
                cfg.proximity_weight * (1.0 - proximity)
                + cfg.continuation_weight * (1.0 - continuation)
                + cfg.curvature_weight * (1.0 - curvature_score)
                + cfg.symmetry_weight * (1.0 - symmetry_score)
            )
            confidence = float(np.clip(1.0 - penalty, 0.0, 1.0))
            if confidence < cfg.min_confidence:
                continue

            same_path = a.path_index == b.path_index
            gap_kind = "single_contour_gap" if same_path else "bridge"

            efd_ready = same_path and (a.path_index in efd_data)
            if gap_kind == "single_contour_gap" and efd_ready and confidence >= cfg.efd_confidence_threshold:
                method = "efd"
            elif gap_kind == "single_contour_gap" and symmetry_score >= cfg.symmetry_confidence_threshold:
                method = "symmetry"
            else:
                method = "bezier"

            reasons = [
                f"distance={dist:.2f}",
                f"align=({align_a:.2f},{align_b:.2f})",
                f"curv_delta={curvature_delta:.1f}",
            ]
            if method == "efd":
                reasons.append("single_gap_with_efd_support")
            elif method == "symmetry":
                reasons.append("single_gap_symmetry_fallback")

            candidates.append(
                GapRecord(
                    gap_id=gap_id,
                    endpoint_a_id=a.endpoint_id,
                    endpoint_b_id=b.endpoint_id,
                    path_a=a.path_index,
                    path_b=b.path_index,
                    distance=dist,
                    angle_a_deg=angle_a,
                    angle_b_deg=angle_b,
                    curvature_delta_deg=curvature_delta,
                    continuation_score=continuation,
                    proximity_score=proximity,
                    symmetry_score=symmetry_score,
                    geometric_score=penalty,
                    confidence=confidence,
                    gap_kind=gap_kind,
                    suggested_method=method,
                    is_forced=False,
                    reasons=reasons,
                )
            )
            gap_id += 1

    return candidates


def _select_non_conflicting(candidates: List[GapRecord], cfg: FeatureBridgeConfig) -> List[GapRecord]:
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda c: (
            -c.confidence,
            c.geometric_score,
            c.distance,
            c.angle_a_deg + c.angle_b_deg,
        ),
    )

    if not cfg.enforce_endpoint_uniqueness:
        return ranked

    selected: List[GapRecord] = []
    used_endpoints: set = set()
    for cand in ranked:
        if cand.endpoint_a_id in used_endpoints or cand.endpoint_b_id in used_endpoints:
            continue
        selected.append(cand)
        used_endpoints.add(cand.endpoint_a_id)
        used_endpoints.add(cand.endpoint_b_id)

    return selected


def _build_forced_candidates(
    endpoints: List[EndpointRecord],
    selected: List[GapRecord],
    adjacency: Dict[int, set],
    image_diag: float,
) -> List[GapRecord]:
    used: set = set()
    for cand in selected:
        used.add(cand.endpoint_a_id)
        used.add(cand.endpoint_b_id)

    id_to_endpoint = {ep.endpoint_id: ep for ep in endpoints}
    unmatched = [ep for ep in endpoints if ep.endpoint_id not in used]
    if len(unmatched) < 2:
        return []

    max_dist = max(6.0, 0.16 * image_diag)
    forced: List[GapRecord] = []
    next_id = max((c.gap_id for c in selected), default=-1) + 1

    # Greedy relaxed pairing among remaining endpoints.
    while len(unmatched) >= 2:
        best = None
        best_score = float("inf")

        for i in range(len(unmatched)):
            a = unmatched[i]
            for j in range(i + 1, len(unmatched)):
                b = unmatched[j]
                if b.path_index in adjacency.get(a.path_index, set()):
                    continue

                vec = b.point - a.point
                dist = float(np.linalg.norm(vec))
                if dist < 1e-9 or dist > max_dist:
                    continue

                unit = vec / dist
                angle_a = _angle_deg(a.outward_tangent, unit)
                angle_b = _angle_deg(b.outward_tangent, -unit)
                score = dist + 0.35 * (angle_a + angle_b)
                if score < best_score:
                    best_score = score
                    best = (i, j, dist, angle_a, angle_b)

        if best is None:
            break

        i, j, dist, angle_a, angle_b = best
        a = unmatched[i]
        b = unmatched[j]

        # Remove larger index first to avoid index shift.
        unmatched.pop(j)
        unmatched.pop(i)

        curvature_delta = abs(a.curvature_deg - b.curvature_deg)
        confidence = float(np.clip(0.25 - dist / (2.0 * max_dist), 0.05, 0.25))
        forced.append(
            GapRecord(
                gap_id=next_id,
                endpoint_a_id=a.endpoint_id,
                endpoint_b_id=b.endpoint_id,
                path_a=a.path_index,
                path_b=b.path_index,
                distance=dist,
                angle_a_deg=angle_a,
                angle_b_deg=angle_b,
                curvature_delta_deg=curvature_delta,
                continuation_score=0.0,
                proximity_score=float(np.clip(1.0 - dist / max_dist, 0.0, 1.0)),
                symmetry_score=0.0,
                geometric_score=1.0 - confidence,
                confidence=confidence,
                gap_kind="bridge" if a.path_index != b.path_index else "single_contour_gap",
                suggested_method="bezier",
                is_forced=True,
                reasons=["forced_best_effort", f"distance={dist:.2f}"],
            )
        )
        next_id += 1

    # Keep deterministic ordering.
    return sorted(forced, key=lambda g: g.gap_id)


def extract_all_features(
    paths: List[BezierPath],
    efd_data: Dict[int, np.ndarray],
    config: Optional[FeatureBridgeConfig] = None,
    adjacency: Optional[Dict[int, set]] = None,
) -> FeatureBundle:
    """Extract endpoints and scored gap candidates.

    This stage enforces strict directional validity for primary candidates,
    then optionally adds low-confidence forced connections so no endpoint is
    left unconsidered.
    """
    cfg = config or FeatureBridgeConfig()
    adj = adjacency or {}

    endpoints = _extract_endpoints(paths)
    if not endpoints:
        return FeatureBundle(endpoints=[], gaps=[], symmetry_axis=None, global_symmetry_confidence=0.0)

    all_pts = np.vstack([ep.point for ep in endpoints])
    image_diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))) + 1e-6

    symmetry_axis, symmetry_conf = _estimate_global_symmetry(paths)
    candidates = _build_initial_candidates(
        endpoints=endpoints,
        adjacency=adj,
        cfg=cfg,
        symmetry_axis=symmetry_axis,
        efd_data=efd_data,
        image_diag=image_diag,
    )

    selected = _select_non_conflicting(candidates, cfg)

    if cfg.force_connect_unmatched:
        forced = _build_forced_candidates(endpoints, selected, adj, image_diag)
        selected.extend(forced)

    selected = sorted(selected, key=lambda g: (-g.confidence, g.gap_id))
    for idx, gap in enumerate(selected):
        gap.gap_id = idx

    return FeatureBundle(
        endpoints=endpoints,
        gaps=selected,
        symmetry_axis=symmetry_axis,
        global_symmetry_confidence=symmetry_conf,
    )


def serialize_features_to_asp(features: FeatureBundle, paths: List[BezierPath]) -> str:
    """Serialize extracted geometry features as ASP facts."""
    lines: List[str] = []

    lines.append(f"num_paths({len(paths)}).")
    lines.append(f"num_endpoints({len(features.endpoints)}).")

    for ep in features.endpoints:
        eid = f"e{ep.endpoint_id}"
        etype = "start" if ep.endpoint_type == "start" else "end"
        x_i = int(round(float(ep.point[0])))
        y_i = int(round(float(ep.point[1])))
        tx_i = int(round(float(ep.outward_tangent[0]) * 1000.0))
        ty_i = int(round(float(ep.outward_tangent[1]) * 1000.0))
        curv_i = int(round(float(ep.curvature_deg) * 1000.0))
        lines.append(
            f"endpoint({eid},{ep.path_index},{etype},{x_i},{y_i})."
        )
        lines.append(f"endpoint_tangent({eid},{tx_i},{ty_i}).")
        lines.append(f"endpoint_curvature({eid},{curv_i}).")

    if features.symmetry_axis is not None:
        center, axis = features.symmetry_axis
        cx_i = int(round(float(center[0]) * 1000.0))
        cy_i = int(round(float(center[1]) * 1000.0))
        ax_i = int(round(float(axis[0]) * 1000000.0))
        ay_i = int(round(float(axis[1]) * 1000000.0))
        sym_i = int(round(float(features.global_symmetry_confidence) * 1000.0))
        lines.append(
            f"symmetry_axis({cx_i},{cy_i},{ax_i},{ay_i})."
        )
        lines.append(f"symmetry_confidence({sym_i}).")

    for gap in features.gaps:
        gid = f"g{gap.gap_id}"
        e1 = f"e{gap.endpoint_a_id}"
        e2 = f"e{gap.endpoint_b_id}"
        kind_atom = "single_gap" if gap.gap_kind == "single_contour_gap" else "bridge"
        method_atom = gap.suggested_method
        conf1000 = int(round(gap.confidence * 1000.0))
        dist1000 = int(round(gap.distance * 1000.0))
        angle_a1000 = int(round(gap.angle_a_deg * 1000.0))
        angle_b1000 = int(round(gap.angle_b_deg * 1000.0))

        lines.append(f"candidate({gid},{e1},{e2}).")
        lines.append(f"candidate_paths({gid},{gap.path_a},{gap.path_b}).")
        lines.append(f"candidate_kind({gid},{kind_atom}).")
        lines.append(f"candidate_method_hint({gid},{method_atom}).")
        lines.append(f"candidate_confidence({gid},{conf1000}).")
        lines.append(f"candidate_weight({gid},{conf1000}).")
        lines.append(f"candidate_distance({gid},{dist1000}).")
        lines.append(f"candidate_angle({gid},{angle_a1000},{angle_b1000}).")
        if gap.is_forced:
            lines.append(f"candidate_forced({gid}).")

    return "\n".join(lines) + "\n"
