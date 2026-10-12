"""Phase 3 — Scoring.

MDTW (via dtaidistance), MJerk, and Gestalt heuristic scoring for
connection candidates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from dtaidistance import dtw
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from bezier_curves.bezier import BezierPath
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.candidates import ConnectionCandidate

# Arc-length spacing (pixels) of the samples MJerk is computed on.
JERK_STEP_PX = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Default scoring weights
# ═══════════════════════════════════════════════════════════════════════════

WEIGHTS_TIER1 = {
    "proximity": 0.20,
    "continuation": 0.30,
    "closure": 0.15,
    "dtw": 0.20,
    "jerk": 0.15,
}

WEIGHTS_TIER2 = {
    "proximity": 0.05,
    "continuation": 0.40,
    "closure": 0.15,
    "dtw": 0.25,
    "jerk": 0.15,
}


# ═══════════════════════════════════════════════════════════════════════════
# Metric computations
# ═══════════════════════════════════════════════════════════════════════════

def context_curve(
    candidate: ConnectionCandidate,
    dense_samples: List[np.ndarray],
    tail_len: int = 20,
) -> np.ndarray:
    """The candidate's bridge with its local context, as one continuous curve.

    *dense_samples[i]* are the samples of path i from its start to its end.
    The curve is the last *tail_len* samples of the source path running into
    ep_a, the bridge from ep_a to ep_b, then the first *tail_len* samples of
    the target path running away from ep_b.
    """
    source = dense_samples[candidate.ep_a.path_index]
    target = dense_samples[candidate.ep_b.path_index]
    into_a = source[-tail_len:] if candidate.ep_a.end == "end" else source[:tail_len][::-1]
    away_from_b = target[:tail_len] if candidate.ep_b.end == "start" else target[-tail_len:][::-1]
    return np.vstack([into_a, candidate.bridge_points, away_from_b])


def compute_mdtw(combined: np.ndarray) -> float:
    """DTW distance between a curve and the straight line joining its ends.

    Lower = smoother, more natural connection.
    """
    if len(combined) < 4:
        return 0.0

    # Baseline: straight line from start to end of combined
    baseline = np.linspace(combined[0], combined[-1], len(combined))

    # Compute DTW on x and y independently, then combine
    dtw_x = dtw.distance(combined[:, 0].astype(np.double),
                         baseline[:, 0].astype(np.double))
    dtw_y = dtw.distance(combined[:, 1].astype(np.double),
                         baseline[:, 1].astype(np.double))
    return float(np.hypot(dtw_x, dtw_y))


def compute_mjerk(combined: np.ndarray) -> float:
    """Mean squared jerk (3rd derivative with respect to arc length) along a curve.

    Lower = smoother. The samples are first replaced by points every
    JERK_STEP_PX along a cubic spline through them. Differentiating the raw
    samples measures the sampling instead of the shape: Bezier samples repeat
    the point at each segment joint (a zero-length step) and are unevenly
    spaced, and a polyline has a small kink at every sample.
    """
    steps = np.linalg.norm(np.diff(combined, axis=0), axis=1)
    points = combined[np.concatenate(([True], steps > 1e-9))]
    if len(points) < 4:
        return 0.0

    # Chord length along the samples approximates arc length.
    chord = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    length = float(chord[-1])
    if length < 4.0 * JERK_STEP_PX:
        return 0.0

    num_points = int(np.ceil(length / JERK_STEP_PX)) + 1
    step = length / (num_points - 1)
    resampled = CubicSpline(chord, points)(np.linspace(0.0, length, num_points))
    jerk = np.diff(resampled, n=3, axis=0) / step ** 3
    return float(np.mean(np.sum(jerk ** 2, axis=1)))


# ═══════════════════════════════════════════════════════════════════════════
# Gestalt heuristics
# ═══════════════════════════════════════════════════════════════════════════

def _gestalt_proximity(distance: float, radius: float) -> float:
    """Proximity score: 1.0 when touching, 0.0 at radius edge."""
    if radius < 1e-6:
        return 1.0
    return max(0.0, 1.0 - distance / radius)


def _gestalt_continuation(
    tangent_a: np.ndarray, direction_ab: np.ndarray,
) -> float:
    """Good-continuation score: cos(angle between tangent and bridge direction).

    Returns value in [-1, 1]; 1.0 = perfect alignment.
    """
    n = np.linalg.norm(direction_ab)
    if n < 1e-12:
        return 0.0
    return float(np.dot(tangent_a, direction_ab / n))


def _gestalt_closure(
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    all_endpoints: List[EndpointInfo],
) -> float:
    """1.0 if connecting ep_a and ep_b would close a loop, else 0.0.

    A loop is closed when every endpoint of the paths involved in the
    chain between ep_a and ep_b becomes connected.
    """
    # Simple heuristic: if after connecting these two, both paths would
    # have all their endpoints connected to something, it's closure.
    # For now, check if connecting the *other* ends of these two paths
    # would complete a loop (both other ends already share a path).
    idx_a = ep_a.path_index
    idx_b = ep_b.path_index

    # Find the other endpoints of those same paths
    other_a = [e for e in all_endpoints
               if e.path_index == idx_a and e.end != ep_a.end]
    other_b = [e for e in all_endpoints
               if e.path_index == idx_b and e.end != ep_b.end]

    if not other_a or not other_b:
        return 0.0

    # If the other ends are close to each other, connecting a↔b creates a loop
    dist = float(np.linalg.norm(other_a[0].position - other_b[0].position))
    if dist < 15.0:
        return 1.0
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_metric(values: List[float]) -> List[float]:
    """Min-max normalize a list of values to [0, 1]."""
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    rng = mx - mn
    if rng < 1e-12:
        return [0.0] * len(values)
    return [(v - mn) / rng for v in values]


def _path_bbox_stats(paths: List[BezierPath]) -> Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """Return per-path bbox stats: (min_xy, max_xy, area, centroid)."""
    stats: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = {}
    for idx, path in enumerate(paths):
        pts = path.sample(pts_per_segment=20)
        if len(pts) == 0:
            min_xy = np.array([0.0, 0.0], dtype=np.float64)
            max_xy = np.array([0.0, 0.0], dtype=np.float64)
            area = 0.0
            centroid = np.array([0.0, 0.0], dtype=np.float64)
        else:
            min_xy = np.min(pts, axis=0)
            max_xy = np.max(pts, axis=0)
            wh = np.maximum(0.0, max_xy - min_xy)
            area = float(wh[0] * wh[1])
            centroid = 0.5 * (min_xy + max_xy)
        stats[idx] = (min_xy, max_xy, area, centroid)
    return stats


def _same_shape_affinity(
    path_a_idx: int,
    path_b_idx: int,
    bbox_stats: Dict[int, Tuple[np.ndarray, np.ndarray, float, np.ndarray]],
) -> float:
    """Estimate whether two paths are likely fragments of the same shape."""
    min_a, max_a, area_a, cent_a = bbox_stats[path_a_idx]
    min_b, max_b, area_b, cent_b = bbox_stats[path_b_idx]

    if area_a < 1e-6 or area_b < 1e-6:
        return 0.0

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_wh = np.maximum(0.0, inter_max - inter_min)
    inter_area = float(inter_wh[0] * inter_wh[1])
    overlap_ratio = inter_area / max(min(area_a, area_b), 1e-6)

    centroid_dist = float(np.linalg.norm(cent_a - cent_b))
    scale = np.sqrt(max(min(area_a, area_b), 1e-6))
    centroid_factor = max(0.0, 1.0 - centroid_dist / (3.0 * scale))

    affinity = overlap_ratio * centroid_factor
    return float(np.clip(affinity, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def score_candidates(
    candidates: List[ConnectionCandidate],
    result: ExtractionResult,
) -> List[ConnectionCandidate]:
    """Score all candidates in-place and return sorted (best first)."""
    if not candidates:
        return []

    radius_t1 = result.diagonal * 0.15
    radius_t2 = radius_t1 * 2.0
    paths = result.paths
    endpoints = result.endpoints
    bbox_stats = _path_bbox_stats(paths)

    all_path_pts = []
    path_indices = []
    for idx, p in enumerate(paths):
        pts = p.sample(pts_per_segment=20)
        if len(pts) > 0:
            all_path_pts.append(pts)
            path_indices.extend([idx] * len(pts))
    
    kdtree = None
    if all_path_pts:
        all_path_pts = np.vstack(all_path_pts)
        kdtree = cKDTree(all_path_pts)
        path_indices = np.array(path_indices)

    # Compute raw metrics on each bridge together with its local path context
    dense_samples = [p.sample(pts_per_segment=50) for p in paths]
    raw_dtw: List[float] = []
    raw_jerk: List[float] = []
    for c in candidates:
        combined = context_curve(c, dense_samples)
        raw_dtw.append(compute_mdtw(combined))
        raw_jerk.append(compute_mjerk(combined))

    # Normalize costs
    norm_dtw = _normalize_metric(raw_dtw)
    norm_jerk = _normalize_metric(raw_jerk)

    # Score each candidate
    for idx, c in enumerate(candidates):
        radius = radius_t1 if c.tier == 1 else radius_t2
        weights = WEIGHTS_TIER1 if c.tier == 1 else WEIGHTS_TIER2

        prox = _gestalt_proximity(c.distance, radius)
        direction_ab = c.ep_b.position - c.ep_a.position
        cont = _gestalt_continuation(c.ep_a.tangent, direction_ab)
        clos = _gestalt_closure(c.ep_a, c.ep_b, endpoints)

        c.score = (
            weights["proximity"] * prox
            + weights["continuation"] * max(0.0, cont)
            + weights["closure"] * clos
            - weights["dtw"] * norm_dtw[idx]
            - weights["jerk"] * norm_jerk[idx]
        )

        # PR2 tie-break refinements: prioritize robust local closure, suppress spur links.
        c.score += 0.16 * max(0.0, float(c.bilateral_alignment))
        if c.same_path_closure:
            c.score += 0.20
        if c.spur_involved:
            c.score -= 0.22

        misalignment_deg = float(c.misalignment_deg)
        if misalignment_deg > 70.0:
            c.score -= 0.08 * min(1.0, (misalignment_deg - 70.0) / 60.0)

        if c.tier == 2 and float(c.bilateral_alignment) < 0.25:
            c.score -= 0.06

        ext_quality = float(np.clip(c.extension_quality, 0.0, 1.0))
        if c.scenario == "extension_intersection":
            if c.same_path_closure:
                # Favor extension for sharp same-path closures only when alignment evidence is strong.
                corner_factor = float(np.clip((misalignment_deg - 55.0) / 60.0, 0.0, 1.0))
                continuation_strength = float(np.clip(c.bilateral_alignment, 0.0, 1.0))
                extension_need = 1.0 - continuation_strength
                context_conf = min(
                    float(np.clip(c.ep_a.tangent_confidence, 0.0, 1.0)),
                    float(np.clip(c.ep_b.tangent_confidence, 0.0, 1.0)),
                )
                c.score += (0.26 + 0.10 * corner_factor) * ext_quality * (extension_need ** 2) * context_conf
                if continuation_strength > 0.60:
                    c.score -= 0.16 * ((continuation_strength - 0.60) / 0.40)
                if 70.0 <= misalignment_deg <= 100.0 and continuation_strength > 0.60:
                    c.score -= 0.08 * ((continuation_strength - 0.60) / 0.40)
                if ext_quality < 0.40:
                    c.score -= 0.06 * ((0.40 - ext_quality) / 0.40)
            else:
                c.score += 0.04 * ext_quality
                # Recover short corner-like cross-path closures that are geometrically
                # plausible but under-scored by continuation-based penalties.
                bilateral = float(np.clip(c.bilateral_alignment, -1.0, 1.0))
                if (
                    c.tier == 1
                    and c.distance <= min(60.0, radius_t1 * 0.42)
                    and ext_quality >= 0.90
                    and 125.0 <= misalignment_deg <= 165.0
                    and 0.0 <= bilateral <= 0.18
                ):
                    c.score += 0.20 * ext_quality

                # Sparse-scene Tier-2 fallback removed to prevent massive false connections.
                
                # Penalize extension intersections that cross other paths.
                if c.tier == 2 and kdtree is not None:
                    dists, idxs = kdtree.query(c.bridge_points)
                    closest_paths = path_indices[idxs]
                    mask = (closest_paths != int(c.ep_a.path_index)) & (closest_paths != int(c.ep_b.path_index))
                    if np.any(dists[mask] < 3.0):
                        c.score -= 0.60

    # PR4b: suppress cross-shape links when a path has strong self-closure support.
    best_self_closure_by_path: Dict[int, Tuple[float, float]] = {}
    affinity_by_candidate: Dict[int, float] = {}

    for c in candidates:
        if not c.same_path_closure:
            continue
        path_idx = int(c.ep_a.path_index)
        current = best_self_closure_by_path.get(path_idx)
        candidate_data = (float(c.score), float(c.distance))
        if current is None or candidate_data[0] > current[0]:
            best_self_closure_by_path[path_idx] = candidate_data

    for c in candidates:
        if c.same_path_closure:
            continue
        if c.ep_a.path_index == c.ep_b.path_index:
            continue

        suppression = 0.0
        touched_paths = (int(c.ep_a.path_index), int(c.ep_b.path_index))
        for path_idx in touched_paths:
            self_info = best_self_closure_by_path.get(path_idx)
            if self_info is None:
                continue
            self_score, self_dist = self_info
            distance_ratio = float(c.distance / max(self_dist, 1e-6))

            if self_score >= 0.60 and distance_ratio >= 0.95:
                suppression = max(suppression, 0.18)
            elif self_score >= 0.45 and distance_ratio >= 1.02:
                suppression = max(suppression, 0.12)

        if suppression > 0.0:
            c.score -= suppression

        # PR4b: boost likely same-shape fragment links (e.g. split bubble parts).
        affinity = _same_shape_affinity(
            int(c.ep_a.path_index),
            int(c.ep_b.path_index),
            bbox_stats,
        )
        affinity_by_candidate[int(c.id)] = affinity
        if affinity >= 0.18 and c.distance <= (result.diagonal * 0.22):
            c.score += 0.16 * affinity

    # PR4b: if a path has a strong high-affinity option, suppress low-affinity competitors.
    best_high_affinity_score_by_path: Dict[int, float] = {}
    for c in candidates:
        if c.same_path_closure:
            continue
        affinity = affinity_by_candidate.get(int(c.id), 0.0)
        if affinity < 0.22:
            continue
        for pidx in (int(c.ep_a.path_index), int(c.ep_b.path_index)):
            best_high_affinity_score_by_path[pidx] = max(
                best_high_affinity_score_by_path.get(pidx, -1e9),
                float(c.score),
            )

    for c in candidates:
        if c.same_path_closure:
            continue
        affinity = affinity_by_candidate.get(int(c.id), 0.0)
        if affinity >= 0.10:
            continue

        suppression = 0.0
        for pidx in (int(c.ep_a.path_index), int(c.ep_b.path_index)):
            support_score = best_high_affinity_score_by_path.get(pidx)
            if support_score is None:
                continue
            if float(c.score) <= support_score + 0.18:
                suppression = max(suppression, 0.10)

        if suppression > 0.0:
            c.score -= suppression

    # Sort best-first
    candidates.sort(
        key=lambda c: (
            -c.score,
            int(not c.same_path_closure),
            -float(c.extension_quality),
            c.distance,
            c.misalignment_deg,
            -c.bilateral_alignment,
            c.id,
        )
    )
    return candidates
