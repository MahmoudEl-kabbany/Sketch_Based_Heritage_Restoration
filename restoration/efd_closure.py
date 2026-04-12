"""Phase 6 — EFD Single-Gap Closure.

Detects open paths that form nearly-closed shapes (single gap),
uses symmetry detection or curvature-aware arc interpolation to close them.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

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
    num_axes: int = 18,
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
) -> List[BezierSegment]:
    """Convert a polyline to a sequence of cubic Bezier segments.

    Uses simple chord-based fitting: every ~4 points → 1 cubic segment.
    """
    if len(points) < 2:
        return []

    segments: List[BezierSegment] = []
    n = len(points)
    step = max(3, n // max(1, n // 4))

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

def _path_tangent_at(path: BezierPath, end: str) -> np.ndarray:
    """Unit tangent pointing into the gap (outward from path boundary)."""
    if end == "end":
        cp = path.segments[-1].control_points
        d = 3.0 * (cp[3] - cp[2])
    else:
        cp = path.segments[0].control_points
        d = 3.0 * (cp[0] - cp[1])
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return d / n


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


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def close_single_gaps(
    paths: List[BezierPath],
    efd_contours: List[dict],
    gap_threshold: float = 0.30,
    symmetry_hausdorff: float = 8.0,
) -> List[BezierPath]:
    """Detect and close single-gap contours.

    A path qualifies if:
      - It is open (not already closed)
      - Its start and end points are within gap_threshold * perimeter of each other
        (default 0.30 = Tier 3 large-gap support)
      - It has enough segments to be a meaningful shape

    Multi-gap shapes are left untouched.
    """
    result: List[BezierPath] = []

    for path in paths:
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

        gap_ratio = gap_dist / perimeter
        if gap_ratio > gap_threshold or gap_ratio < 1e-6:
            result.append(path)
            continue

        # This is a single-gap contour — attempt closure
        centroid = np.mean(samples, axis=0)

        # Scale Hausdorff threshold for larger shapes
        scaled_hausdorff = max(symmetry_hausdorff, perimeter * 0.01)

        # Try symmetry-based mirroring
        axis_angle = _detect_symmetry(
            samples, centroid,
            hausdorff_threshold=scaled_hausdorff,
        )

        closure_segments: List[BezierSegment] = []

        if axis_angle is not None:
            # Mirror the opposite portion
            mirrored = _mirror_gap_from_opposite(
                samples, centroid, axis_angle,
                gap_start=end_pt, gap_end=start_pt,
            )
            if len(mirrored) >= 2:
                closure_segments = _points_to_bezier_segments(mirrored)

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

        if closure_segments:
            new_segments = list(path.segments) + closure_segments
            # Snap closure endpoints
            new_segments[-1].control_points[3] = new_segments[0].control_points[0].copy()
            result.append(BezierPath(
                segments=new_segments,
                is_closed=True,
                source_type="restored",
            ))
        else:
            result.append(path)

    return result
