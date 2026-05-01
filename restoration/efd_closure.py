"""Phase 6 — EFD Single-Gap Closure.

Detects open paths that form nearly-closed shapes (single gap),
uses symmetry detection or curvature-aware arc interpolation to close them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from bezier_curves.bezier import BezierPath, BezierSegment


# ═══════════════════════════════════════════════════════════════════════════
# Symmetry detection
# ═══════════════════════════════════════════════════════════════════════════

def _reflect_points(points: np.ndarray, axis_angle: float) -> np.ndarray:
    """Reflect points across a line through the origin at *axis_angle* radians."""
    c = math.cos(2.0 * axis_angle)
    s = math.sin(2.0 * axis_angle)
    R = np.array([[c, s], [s, -c]], dtype=np.float64)
    return points @ R.T


def _detect_symmetry(
    points: np.ndarray,
    centroid: np.ndarray,
    num_axes: int = 24,
    hausdorff_threshold: float = 8.0,
    max_points: int = 500,
) -> Optional[float]:
    """Test reflection symmetry across evenly-spaced axes through the centroid.

    Returns the axis angle (radians) with best symmetry, or None.
    Subsamples to *max_points* for performance on dense contours.
    """
    centered = points - centroid

    # Subsample to cap Hausdorff O(n²) cost
    if len(centered) > max_points:
        indices = np.linspace(0, len(centered) - 1, max_points, dtype=int)
        centered = centered[indices]

    best_angle: Optional[float] = None
    best_dist = hausdorff_threshold

    for k in range(num_axes):
        angle = k * math.pi / num_axes
        reflected = _reflect_points(centered, angle)
        # Hausdorff distance between original and reflected
        d_fwd = directed_hausdorff(centered, reflected)[0]
        d_bwd = directed_hausdorff(reflected, centered)[0]
        d = max(d_fwd, d_bwd)
        if d < best_dist:
            best_dist = d
            best_angle = angle

    return best_angle


def _resample_polyline(points: np.ndarray, max_points: int) -> np.ndarray:
    """Downsample or resample a polyline to at most *max_points* samples."""
    if len(points) <= max_points:
        return points.copy()
    if max_points < 2:
        return points[[0, -1]].copy()

    diffs = np.diff(points, axis=0)
    d = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(d)))
    total = float(cumulative[-1])
    if total < 1e-9:
        idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[idx].copy()

    targets = np.linspace(0.0, total, max_points)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.column_stack((x, y)).astype(np.float64)


def _polyline_length(points: np.ndarray) -> float:
    """Arc length of a polyline."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _bridge_overlaps_gap(
    path: BezierPath,
    gap_start: np.ndarray,
    gap_end: np.ndarray,
    proximity: float,
) -> bool:
    """True when a bridge segment appears to already close this specific gap."""
    for seg in path.segments:
        if seg.source_type != "bridge":
            continue
        b0 = seg.control_points[0]
        b1 = seg.control_points[3]
        direct = float(np.linalg.norm(b0 - gap_start) + np.linalg.norm(b1 - gap_end))
        reverse = float(np.linalg.norm(b0 - gap_end) + np.linalg.norm(b1 - gap_start))
        if min(direct, reverse) <= 2.0 * proximity:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Mirroring
# ═══════════════════════════════════════════════════════════════════════════

def _mirror_gap_from_opposite(
    points: np.ndarray,
    centroid: np.ndarray,
    axis_angle: float,
    gap_start: np.ndarray,
    gap_end: np.ndarray,
) -> np.ndarray:
    """Mirror the portion of the contour opposite to the gap across the symmetry axis.

    Returns (N, 2) points that fill the gap.
    """
    centered = points - centroid
    gap_start_c = gap_start - centroid
    gap_end_c = gap_end - centroid

    # Reflect the gap endpoints to find the "opposite" region
    opp_start = _reflect_points(gap_start_c.reshape(1, 2), axis_angle)[0]
    opp_end = _reflect_points(gap_end_c.reshape(1, 2), axis_angle)[0]

    # Find the portion of the contour near the opposite region
    dists_to_opp_start = np.linalg.norm(centered - opp_start, axis=1)
    dists_to_opp_end = np.linalg.norm(centered - opp_end, axis=1)

    idx_start = int(np.argmin(dists_to_opp_start))
    idx_end = int(np.argmin(dists_to_opp_end))

    # Extract the opposite arc
    if idx_start <= idx_end:
        opposite_arc = centered[idx_start:idx_end + 1]
    else:
        opposite_arc = np.vstack([centered[idx_start:], centered[:idx_end + 1]])

    if len(opposite_arc) < 2:
        return np.empty((0, 2))

    # Reflect the opposite arc — this fills the gap
    mirrored = _reflect_points(opposite_arc, axis_angle) + centroid

    # Trim/adjust endpoints to match gap_start and gap_end exactly
    if len(mirrored) >= 2:
        # Ensure correct direction: mirrored should go from gap_start to gap_end
        d_fwd = (np.linalg.norm(mirrored[0] - gap_start)
                 + np.linalg.norm(mirrored[-1] - gap_end))
        d_rev = (np.linalg.norm(mirrored[-1] - gap_start)
                 + np.linalg.norm(mirrored[0] - gap_end))
        if d_rev < d_fwd:
            mirrored = mirrored[::-1]

        # Snap endpoints
        mirrored[0] = gap_start
        mirrored[-1] = gap_end

    return mirrored


# ═══════════════════════════════════════════════════════════════════════════
# Curvature-aware arc closure (no symmetry)
# ═══════════════════════════════════════════════════════════════════════════

def _curvature_aware_arc(
    gap_start: np.ndarray,
    gap_end: np.ndarray,
    tangent_start: np.ndarray,
    tangent_end: np.ndarray,
    curvature_start: float,
    curvature_end: float,
) -> List[BezierSegment]:
    """Build one or more cubic Bezier segments to close a gap respecting curvature.

    Tangent_start points INTO the gap (outward from path end).
    Tangent_end points INTO the gap (outward from path start, negated).
    """
    p0 = gap_start.copy()
    p3 = gap_end.copy()
    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        return []

    avg_k = (curvature_start + curvature_end) / 2.0

    # Handle lengths
    alpha_0 = chord / 3.0
    alpha_3 = chord / 3.0
    if curvature_start > 1e-6:
        alpha_0 = min(alpha_0, 1.0 / (3.0 * curvature_start))
    if curvature_end > 1e-6:
        alpha_3 = min(alpha_3, 1.0 / (3.0 * curvature_end))

    p1 = p0 + alpha_0 * tangent_start
    p2 = p3 + alpha_3 * tangent_end

    cp = np.vstack([p0, p1, p2, p3])
    return [BezierSegment(control_points=cp, source_type="efd_closure")]


def _points_to_bezier_segments(
    points: np.ndarray,
    max_segments: int = 16,
) -> List[BezierSegment]:
    """Convert a polyline to a sequence of cubic Bezier segments.

    Uses simple chord-based fitting: every ~4 points → 1 cubic segment.
    """
    if len(points) < 2:
        return []

    segments: List[BezierSegment] = []
    n = len(points)
    max_segments = max(1, int(max_segments))
    step = max(2, int(math.ceil((n - 1) / max_segments)))

    i = 0
    while i < n - 1:
        end = min(i + step, n - 1)
        p0 = points[i]
        p3 = points[end]
        chord = float(np.linalg.norm(p3 - p0))
        if chord < 1e-6:
            i = end
            continue

        # Estimate tangent direction from local points
        num = min(3, end - i)
        t_start = points[i + num] - points[i]
        n_s = np.linalg.norm(t_start)
        if n_s > 1e-12:
            t_start = t_start / n_s
        else:
            t_start = (p3 - p0) / chord

        t_end = points[end] - points[max(i, end - num)]
        n_e = np.linalg.norm(t_end)
        if n_e > 1e-12:
            t_end = t_end / n_e
        else:
            t_end = (p3 - p0) / chord

        alpha = chord / 3.0
        p1 = p0 + alpha * t_start
        p2 = p3 - alpha * t_end

        segments.append(BezierSegment(
            control_points=np.vstack([p0, p1, p2, p3]),
            source_type="efd_closure",
        ))
        i = end

    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint tangent extraction
# ═══════════════════════════════════════════════════════════════════════════

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _context_endpoint_tangent(path: BezierPath, end: str) -> Tuple[np.ndarray, float]:
    """Estimate outward tangent from local sampled path context."""
    sampled = path.sample(pts_per_segment=24)
    if len(sampled) < 3:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    diffs = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    keep = np.ones(len(sampled), dtype=bool)
    keep[1:] = diffs > 1e-6
    sampled = sampled[keep]
    if len(sampled) < 3:
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    max_window = max(2, len(sampled) - 1)
    window = int(round(len(sampled) * 0.12))
    window = max(6, min(24, max_window, window))

    if end == "end":
        local = sampled[-(window + 1):]
        vectors = np.diff(local, axis=0)
    else:
        local = sampled[: window + 1]
        vectors = -np.diff(local, axis=0)

    valid = np.linalg.norm(vectors, axis=1) > 1e-9
    if not np.any(valid):
        return np.array([1.0, 0.0], dtype=np.float64), 0.0

    unit = vectors[valid] / np.linalg.norm(vectors[valid], axis=1, keepdims=True)
    mean_vec = _safe_normalize(np.mean(unit, axis=0))
    consistency = float(np.mean(np.clip(unit @ mean_vec, 0.0, 1.0)))
    return mean_vec, float(np.clip(consistency, 0.0, 1.0))

def _path_tangent_at(path: BezierPath, end: str) -> np.ndarray:
    """Unit tangent pointing into the gap (outward from path boundary)."""
    if end == "end":
        cp = path.segments[-1].control_points
        derivative = 3.0 * (cp[3] - cp[2])
    else:
        cp = path.segments[0].control_points
        derivative = 3.0 * (cp[0] - cp[1])

    derivative_tangent = _safe_normalize(derivative)
    context_tangent, confidence = _context_endpoint_tangent(path, end)
    blend = float(np.clip(confidence, 0.15, 0.90))
    return _safe_normalize((1.0 - blend) * derivative_tangent + blend * context_tangent)


def _path_curvature_at(path: BezierPath, end: str) -> float:
    """Unsigned curvature at a path boundary."""
    if end == "end":
        cp = path.segments[-1].control_points
        t = 1.0
    else:
        cp = path.segments[0].control_points
        t = 0.0
    u = 1.0 - t
    d1 = (3.0 * u * u * (cp[1] - cp[0])
          + 6.0 * u * t * (cp[2] - cp[1])
          + 3.0 * t * t * (cp[3] - cp[2]))
    d2 = (6.0 * u * (cp[2] - 2 * cp[1] + cp[0])
          + 6.0 * t * (cp[3] - 2 * cp[2] + cp[1]))
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    speed_sq = float(d1[0] ** 2 + d1[1] ** 2)
    if speed_sq < 1e-24:
        return 0.0
    return abs(cross) / (speed_sq ** 1.5)


def _closure_plausibility_metrics(
    path: BezierPath,
    gap_start: np.ndarray,
    gap_end: np.ndarray,
    gap_dist: float,
    gap_ratio: float,
) -> Dict[str, float]:
    """Compute semantic closure plausibility metrics for one open path."""
    t_start = _path_tangent_at(path, "end")
    t_end = _path_tangent_at(path, "start")
    _, conf_start = _context_endpoint_tangent(path, "end")
    _, conf_end = _context_endpoint_tangent(path, "start")

    gap_dir = _safe_normalize(gap_end - gap_start)
    forward_start = float(np.dot(t_start, gap_dir))
    forward_end = float(np.dot(t_end, -gap_dir))
    bilateral = float(min(forward_start, forward_end))
    continuation = float(np.clip(0.5 * (max(0.0, forward_start) + max(0.0, forward_end)), 0.0, 1.0))

    anti_parallel = float(np.clip(np.dot(t_start, -t_end), -1.0, 1.0))
    misalignment_deg = float(np.degrees(np.arccos(anti_parallel)))

    k_start = _path_curvature_at(path, "end")
    k_end = _path_curvature_at(path, "start")
    k_sum = float(k_start + k_end)
    if k_sum < 1e-9:
        curvature_coherence = 1.0
    else:
        curvature_coherence = float(np.clip(1.0 - abs(k_start - k_end) / k_sum, 0.0, 1.0))

    context_conf = float(np.clip(min(conf_start, conf_end), 0.0, 1.0))
    confidence_factor = float(np.clip(0.60 + 0.40 * context_conf, 0.0, 1.0))
    score_raw = (
        0.45 * continuation
        + 0.35 * float(np.clip(bilateral, 0.0, 1.0))
        + 0.20 * curvature_coherence
    )
    plausibility_score = float(np.clip(score_raw * confidence_factor, 0.0, 1.0))

    long_gap_semantic_risk = bool(
        gap_dist > 90.0
        and gap_ratio > 0.16
        and plausibility_score < 0.62
        and (bilateral < 0.12 or misalignment_deg > 95.0)
    )

    return {
        "forward_start": forward_start,
        "forward_end": forward_end,
        "bilateral_alignment": bilateral,
        "continuation_score": continuation,
        "misalignment_deg": misalignment_deg,
        "curvature_start": float(k_start),
        "curvature_end": float(k_end),
        "curvature_coherence": curvature_coherence,
        "context_confidence": context_conf,
        "plausibility_score": plausibility_score,
        "long_gap_semantic_risk": float(1.0 if long_gap_semantic_risk else 0.0),
    }


def _evaluate_closure_validity(
    metrics: Dict[str, float],
    gap_dist: float,
    gap_ratio: float,
    has_symmetry: bool,
    plausibility_threshold: float,
    min_gap_for_check: float,
) -> Tuple[bool, str]:
    """Return (accepted, reason) for semantic EFD closure validation."""
    if gap_dist <= min_gap_for_check:
        return True, "tiny_gap_bypass"

    if bool(metrics.get("long_gap_semantic_risk", 0.0) >= 0.5):
        return False, "long_gap_semantic_risk"

    bilateral = float(metrics.get("bilateral_alignment", 0.0))
    continuation = float(metrics.get("continuation_score", 0.0))
    misalignment = float(metrics.get("misalignment_deg", 180.0))
    score = float(metrics.get("plausibility_score", 0.0))
    effective_threshold = float(plausibility_threshold)

    # Be slightly more permissive for tiny smooth gaps where tangent evidence is
    # often unstable but geometric closure is still valid.
    if (
        gap_ratio <= 0.035
        and gap_dist <= 45.0
        and continuation >= 0.10
        and misalignment <= 35.0
    ):
        effective_threshold = min(effective_threshold, 0.24)

    # Symmetry can rescue a borderline case, but not highly implausible tangent geometry.
    if has_symmetry and bilateral >= 0.20 and misalignment <= 110.0:
        if score >= max(0.45, effective_threshold * 0.80):
            return True, "symmetry_supported"

    if score >= effective_threshold:
        return True, "plausibility_pass"

    # Borderline pass for moderately long but coherent closures.
    if (
        gap_ratio <= 0.16
        and score >= max(0.42, plausibility_threshold * 0.85)
        and bilateral >= 0.16
        and continuation >= 0.35
        and misalignment <= 132.0
    ):
        return True, "borderline_continuation_pass"

    if bilateral < 0.05 and misalignment > 120.0:
        return False, "weak_continuation_support"
    return False, "low_plausibility_score"


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def close_single_gaps(
    paths: List[BezierPath],
    efd_contours: List[dict],
    gap_threshold: float = 0.30,
    symmetry_hausdorff: float = 8.0,
    skip_if_bridge_present: bool = True,
    symmetry_min_gap_ratio: float = 0.08,
    symmetry_max_gap_ratio: float = 0.30,
    max_mirrored_points: int = 80,
    max_closure_length_ratio: float = 0.45,
    conservative_small_gap_px: float = 20.0,
    conservative_gap_ratio: float = 0.45,
    conservative_min_perimeter: float = 80.0,
    validity_check_enabled: bool = True,
    plausibility_threshold: float = 0.50,
    min_gap_for_validity_check: float = 3.0,
    return_diagnostics: bool = False,
) -> Union[List[BezierPath], Tuple[List[BezierPath], List[Dict[str, Any]]]]:
    """Detect and close single-gap contours.

    A path qualifies if:
      - It is open (not already closed)
      - Its start and end points are within gap_threshold * perimeter of each other
        (default 0.30 = Tier 3 large-gap support)
      - It has enough segments to be a meaningful shape

    Multi-gap shapes are left untouched.
    """
    result: List[BezierPath] = []
    diagnostics: List[Dict[str, Any]] = []

    for path_index, path in enumerate(paths):
        if path.is_closed or len(path.segments) < 3:
            result.append(path)
            continue

        start_pt = path.segments[0].control_points[0]
        end_pt = path.segments[-1].control_points[3]
        gap_dist = float(np.linalg.norm(end_pt - start_pt))

        # Estimate perimeter
        samples = path.sample(pts_per_segment=30)
        diffs = np.diff(samples, axis=0)
        perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))

        if perimeter < 1e-6:
            result.append(path)
            continue

        if skip_if_bridge_present:
            # PR4 tuning: skip only if a bridge overlaps this same open gap.
            gap_proximity = max(8.0, perimeter * 0.02)
            if _bridge_overlaps_gap(path, end_pt, start_pt, proximity=gap_proximity):
                result.append(path)
                continue

        gap_ratio = gap_dist / perimeter

        force_small_gap_fallback = False
        if gap_ratio > gap_threshold:
            force_small_gap_fallback = (
                gap_dist <= conservative_small_gap_px
                and gap_ratio <= conservative_gap_ratio
                and perimeter >= conservative_min_perimeter
            )
            if not force_small_gap_fallback:
                result.append(path)
                continue

        if gap_ratio < 1e-6:
            result.append(path)
            continue

        # This is a single-gap contour — attempt closure
        centroid = np.mean(samples, axis=0)

        # Scale Hausdorff threshold for larger shapes
        scaled_hausdorff = max(symmetry_hausdorff, perimeter * 0.015)

        # Try symmetry-based mirroring
        axis_angle = None
        use_symmetry = (
            not force_small_gap_fallback
            and symmetry_min_gap_ratio <= gap_ratio <= symmetry_max_gap_ratio
        )
        if use_symmetry:
            axis_angle = _detect_symmetry(
                samples, centroid,
                hausdorff_threshold=scaled_hausdorff,
            )

        plausibility_metrics = _closure_plausibility_metrics(
            path=path,
            gap_start=end_pt,
            gap_end=start_pt,
            gap_dist=gap_dist,
            gap_ratio=gap_ratio,
        )

        validity_enabled = bool(validity_check_enabled) and not force_small_gap_fallback
        if validity_enabled:
            valid, reject_reason = _evaluate_closure_validity(
                plausibility_metrics,
                gap_dist=gap_dist,
                gap_ratio=gap_ratio,
                has_symmetry=(axis_angle is not None),
                plausibility_threshold=plausibility_threshold,
                min_gap_for_check=min_gap_for_validity_check,
            )
            if not valid:
                diagnostics.append({
                    "path_index": int(path_index),
                    "accepted": False,
                    "closure_method": "none",
                    "reason": str(reject_reason),
                    "gap_distance_px": round(float(gap_dist), 3),
                    "gap_ratio": round(float(gap_ratio), 6),
                    "plausibility_threshold": round(float(plausibility_threshold), 4),
                    "metrics": {
                        k: round(float(v), 6) for k, v in plausibility_metrics.items()
                    },
                })
                result.append(path)
                continue

        closure_segments: List[BezierSegment] = []
        closure_method = "curvature_arc"

        if axis_angle is not None:
            # Mirror the opposite portion
            mirrored = _mirror_gap_from_opposite(
                samples, centroid, axis_angle,
                gap_start=end_pt, gap_end=start_pt,
            )
            if len(mirrored) >= 2:
                mirrored = _resample_polyline(mirrored, max_points=max_mirrored_points)
                mirrored_len = _polyline_length(mirrored)
                # PR3: reject mirrored closures that are much too complex for this gap.
                if mirrored_len <= max(gap_dist * 3.0, perimeter * max_closure_length_ratio):
                    closure_segments = _points_to_bezier_segments(
                        mirrored,
                        max_segments=max(6, int(max_mirrored_points // 8)),
                    )
                    if closure_segments:
                        closure_method = "symmetry_mirroring"

        if not closure_segments:
            # Fallback: curvature-aware arc
            t_start = _path_tangent_at(path, "end")
            t_end = _path_tangent_at(path, "start")
            k_start = _path_curvature_at(path, "end")
            k_end = _path_curvature_at(path, "start")
            closure_segments = _curvature_aware_arc(
                end_pt, start_pt,
                t_start, t_end,
                k_start, k_end,
            )
            closure_method = "curvature_arc"

        if closure_segments:
            new_segments = list(path.segments) + closure_segments
            # Snap closure endpoints
            new_segments[-1].control_points[3] = new_segments[0].control_points[0].copy()
            result.append(BezierPath(
                segments=new_segments,
                is_closed=True,
                source_type="restored",
            ))
            diagnostics.append({
                "path_index": int(path_index),
                "accepted": True,
                "closure_method": str(closure_method),
                "reason": "accepted",
                "gap_distance_px": round(float(gap_dist), 3),
                "gap_ratio": round(float(gap_ratio), 6),
                "plausibility_threshold": round(float(plausibility_threshold), 4),
                "metrics": {
                    k: round(float(v), 6) for k, v in plausibility_metrics.items()
                },
            })
        else:
            result.append(path)
            diagnostics.append({
                "path_index": int(path_index),
                "accepted": False,
                "closure_method": "none",
                "reason": "no_geometry_constructed",
                "gap_distance_px": round(float(gap_dist), 3),
                "gap_ratio": round(float(gap_ratio), 6),
                "plausibility_threshold": round(float(plausibility_threshold), 4),
                "metrics": {
                    k: round(float(v), 6) for k, v in plausibility_metrics.items()
                },
            })

    if return_diagnostics:
        return result, diagnostics
    return result
