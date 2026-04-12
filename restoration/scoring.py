"""Phase 3 — Scoring.

MDTW (via dtaidistance), MJerk, and Gestalt heuristic scoring for
connection candidates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from dtaidistance import dtw

from bezier_curves.bezier import BezierPath
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.candidates import ConnectionCandidate


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

def _compute_mdtw(
    bridge_pts: np.ndarray,
    source_path: BezierPath,
    target_path: BezierPath,
    tail_len: int = 20,
) -> float:
    """DTW distance between bridge + local path context and a straight baseline.

    Lower = smoother, more natural connection.
    """
    # Sample tails from source and target paths
    source_samples = source_path.sample(pts_per_segment=50)
    target_samples = target_path.sample(pts_per_segment=50)

    source_tail = source_samples[-tail_len:] if len(source_samples) >= tail_len else source_samples
    target_head = target_samples[:tail_len] if len(target_samples) >= tail_len else target_samples

    # Combined curve: source_tail + bridge + target_head
    combined = np.vstack([source_tail, bridge_pts, target_head])
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


def _compute_mjerk(
    bridge_pts: np.ndarray,
    source_path: BezierPath,
    target_path: BezierPath,
    tail_len: int = 20,
) -> float:
    """Mean squared jerk (3rd derivative) of source_tail + bridge + target_head.

    Lower = smoother.
    """
    source_samples = source_path.sample(pts_per_segment=50)
    target_samples = target_path.sample(pts_per_segment=50)

    source_tail = source_samples[-tail_len:] if len(source_samples) >= tail_len else source_samples
    target_head = target_samples[:tail_len] if len(target_samples) >= tail_len else target_samples

    combined = np.vstack([source_tail, bridge_pts, target_head])
    if len(combined) < 5:
        return 0.0

    # Arc-length parameterization
    diffs = np.diff(combined, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    ds[ds < 1e-12] = 1e-12

    # Velocity (1st derivative)
    vel = diffs / ds[:, np.newaxis]

    # Acceleration (2nd derivative)
    if len(vel) < 2:
        return 0.0
    d_vel = np.diff(vel, axis=0)
    ds2 = ds[:-1]
    ds2[ds2 < 1e-12] = 1e-12
    acc = d_vel / ds2[:, np.newaxis]

    # Jerk (3rd derivative)
    if len(acc) < 2:
        return 0.0
    d_acc = np.diff(acc, axis=0)
    ds3 = ds2[:-1]
    ds3[ds3 < 1e-12] = 1e-12
    jerk = d_acc / ds3[:, np.newaxis]

    # Mean squared jerk
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
    paths: List[BezierPath],
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

    # Compute raw metrics
    raw_dtw: List[float] = []
    raw_jerk: List[float] = []

    for c in candidates:
        src_path = paths[c.ep_a.path_index]
        tgt_path = paths[c.ep_b.path_index]

        mdtw_val = _compute_mdtw(c.bridge_points, src_path, tgt_path)
        mjerk_val = _compute_mjerk(c.bridge_points, src_path, tgt_path)
        raw_dtw.append(mdtw_val)
        raw_jerk.append(mjerk_val)

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
        clos = _gestalt_closure(c.ep_a, c.ep_b, paths, endpoints)

        c.score = (
            weights["proximity"] * prox
            + weights["continuation"] * max(0.0, cont)
            + weights["closure"] * clos
            - weights["dtw"] * norm_dtw[idx]
            - weights["jerk"] * norm_jerk[idx]
        )

        # PR2 tie-break refinements: prioritize robust local closure, suppress spur links.
        c.score += 0.16 * max(0.0, float(getattr(c, "bilateral_alignment", 0.0)))
        if getattr(c, "same_path_closure", False):
            c.score += 0.20
        if getattr(c, "spur_involved", False):
            c.score -= 0.22

        misalignment_deg = float(getattr(c, "misalignment_deg", 180.0))
        if misalignment_deg > 70.0:
            c.score -= 0.08 * min(1.0, (misalignment_deg - 70.0) / 60.0)

        if c.tier == 2 and float(getattr(c, "bilateral_alignment", 0.0)) < 0.25:
            c.score -= 0.06

    # Sort best-first
    candidates.sort(
        key=lambda c: (
            -c.score,
            c.distance,
            c.misalignment_deg,
            -c.bilateral_alignment,
            c.id,
        )
    )
    return candidates
