"""
R-4: Geometric Synthesis (Restoration)
========================================
Constructs the actual restoration geometry from ASP decisions.

Key operations
--------------
* ``bridge_curves``       – Bézier bridge between two path endpoints (G1)
* ``close_contour_g1``    – G1-continuous closure of an open path
* ``efd_close_contour``   – EFD-based closure for single-gap contours
* ``mirror_bezier_path``  – Affine reflection
* ``replicate_motif``     – Translation-based replication
* ``blend_efd_completion``– Weighted blend of EFD coefficient matrices
* ``execute_restoration`` – Top-level dispatch from ranked hypotheses
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.asp.asp_inference import RankedHypothesis, RestorationAction


# ═══════════════════════════════════════════════════════════════════════
# Shape vocabulary (for template matching)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ShapeVocab:
    """Library of known shapes for template-driven completion."""
    entries: Dict[str, np.ndarray] = field(default_factory=dict)


def query_shape_vocabulary(
    feature_vec: np.ndarray, vocab: ShapeVocab, top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Find the closest shapes in the vocabulary."""
    if not vocab.entries:
        return []
    results = []
    for name, ref_vec in vocab.entries.items():
        dist = float(np.linalg.norm(feature_vec - ref_vec))
        results.append((name, dist))
    results.sort(key=lambda x: x[1])
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# Bridge curves (Bézier bridging for multi-gap shapes)
# ═══════════════════════════════════════════════════════════════════════

def _get_endpoint_and_tangent(
    path: BezierPath, which: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (point, inward_tangent) for an endpoint."""
    if which == "start":
        seg = path.segments[0]
        pt = seg.control_points[0].copy()
        handle = seg.control_points[1]
        # Inward tangent: from endpoint toward the interior
        tangent = handle - pt
    else:
        seg = path.segments[-1]
        pt = seg.control_points[3].copy()
        handle = seg.control_points[2]
        tangent = handle - pt
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / norm
    return pt, tangent


def bridge_curves(
    path_a: BezierPath,
    path_b: BezierPath,
    endpoint_a: str = "end",
    endpoint_b: str = "start",
) -> BezierSegment:
    """Create a G1-continuous cubic Bézier bridge between two path endpoints.

    The bridge handles are aligned with the existing endpoint tangents
    so the join is smooth.  Handle length is proportional to the gap
    distance (1/3 rule — the standard heuristic for Bézier fitting).

    Parameters
    ----------
    path_a, path_b : BezierPath
        The two paths to bridge.
    endpoint_a : str
        Which end of path_a to connect ("start" or "end").
    endpoint_b : str
        Which end of path_b to connect ("start" or "end").

    Returns
    -------
    BezierSegment
        A cubic Bézier segment bridging the gap.
    """
    pt_a, tan_a = _get_endpoint_and_tangent(path_a, endpoint_a)
    pt_b, tan_b = _get_endpoint_and_tangent(path_b, endpoint_b)

    gap = np.linalg.norm(pt_b - pt_a)
    handle_len = gap / 3.0  # Standard Bézier heuristic

    # For "end" endpoint, the tangent points inward → we need outward
    # direction for the bridge handle.  The bridge goes FROM pt_a TO pt_b,
    # so handle at pt_a should continue the tangent direction of path_a,
    # and handle at pt_b should continue the tangent direction of path_b.

    if endpoint_a == "end":
        # Tangent is inward (toward path interior).
        # Bridge handle should continue outward = opposite of inward tangent?
        # No — G1 means the bridge should smoothly continue the curve.
        # So the handle at pt_a is pt_a + (outward_extension) = pt_a - tan_a * len
        # Wait: tan_a = handle - pt, so tan_a points from pt toward handle (inward).
        # For the bridge, we want to continue in the direction the curve was going
        # at the end, which is the direction from the handle TO the endpoint,
        # i.e., -tan_a.  But the bridge CP1 should be on the "other side",
        # continuing the motion.  Actually for G1 at pt_a:
        #   last_seg.P2, last_seg.P3 = handle, pt_a
        #   bridge.P0, bridge.P1 = pt_a, pt_a + alpha * (pt_a - handle)
        # So bridge P1 = pt_a - alpha * tan_a  (where tan_a = handle - pt_a)
        cp1 = pt_a - handle_len * tan_a
    else:
        # endpoint_a == "start": tangent = P1 - P0 (inward).
        # Bridge goes FROM start, so handle continues backward:
        # bridge P1 = pt_a - handle_len * tan_a
        cp1 = pt_a - handle_len * tan_a

    if endpoint_b == "start":
        cp2 = pt_b - handle_len * tan_b
    else:
        cp2 = pt_b - handle_len * tan_b

    control_points = np.array([pt_a, cp1, cp2, pt_b], dtype=np.float64)
    return BezierSegment(control_points, source_type="bridge")


# ═══════════════════════════════════════════════════════════════════════
# Close contour (G1-continuous)
# ═══════════════════════════════════════════════════════════════════════

def close_contour_g1(path: BezierPath) -> BezierSegment:
    """Create a G1-continuous closing segment for an open path.

    Connects path.end → path.start with a cubic Bézier whose handles
    maintain tangent continuity at both joins.
    """
    return bridge_curves(path, path, endpoint_a="end", endpoint_b="start")


# ═══════════════════════════════════════════════════════════════════════
# EFD-based contour closure
# ═══════════════════════════════════════════════════════════════════════

def efd_close_contour(
    contour_points: np.ndarray,
    order: int = 20,
    num_recon_points: int = 500,
) -> Optional[np.ndarray]:
    """Close a single-gap open contour using EFD reconstruction.

    Steps
    -----
    1. Treat the open contour as if it were closed (append start to end).
    2. Compute EFD coefficients on this "pseudo-closed" contour.
    3. Reconstruct the full closed shape with high resolution.
    4. Extract only the gap-filling arc (the part not covered by the
       original contour).

    Parameters
    ----------
    contour_points : np.ndarray
        (N, 2) array of the existing contour points.
    order : int
        Number of EFD harmonics.
    num_recon_points : int
        Points in the reconstructed contour.

    Returns
    -------
    np.ndarray or None
        (M, 2) array of points forming the gap-filling arc, or None
        if reconstruction fails.
    """
    try:
        import pyefd
    except ImportError:
        return None

    pts = np.squeeze(contour_points)
    if pts.ndim != 2 or len(pts) < 5:
        return None

    # Close the contour by appending the start point
    closed_pts = np.vstack([pts, pts[0:1]])

    # Compute EFD coefficients
    try:
        coeffs = pyefd.elliptic_fourier_descriptors(
            closed_pts, order=order, normalize=False
        )
        a0, c0 = pyefd.calculate_dc_coefficients(closed_pts)
        reconstructed = pyefd.reconstruct_contour(
            coeffs, locus=(a0, c0), num_points=num_recon_points
        )
    except Exception:
        return None

    if reconstructed is None or len(reconstructed) < 3:
        return None

    # Find the portions of the reconstruction that fill the gap.
    # Strategy: find the reconstructed points nearest to the two
    # endpoints of the original contour, then extract the arc between
    # them that does NOT overlap the original contour.

    start_pt = pts[0]
    end_pt = pts[-1]

    # Find nearest reconstructed point to each endpoint
    dists_start = np.linalg.norm(reconstructed - start_pt, axis=1)
    dists_end = np.linalg.norm(reconstructed - end_pt, axis=1)

    idx_start = int(np.argmin(dists_start))
    idx_end = int(np.argmin(dists_end))

    if idx_start == idx_end:
        return None

    # Extract arc: from end_pt's nearest → start_pt's nearest
    # (going the "short way" around — the gap side)
    n = len(reconstructed)

    # Two possible arcs
    if idx_end < idx_start:
        arc1 = reconstructed[idx_end:idx_start + 1]
        arc2 = np.vstack([reconstructed[idx_start:], reconstructed[:idx_end + 1]])
    else:
        arc1 = reconstructed[idx_start:idx_end + 1]
        arc2 = np.vstack([reconstructed[idx_end:], reconstructed[:idx_start + 1]])

    # The gap-filling arc is the shorter one (it should be smaller than
    # max_closure_gap_fraction of the total)
    if len(arc1) <= len(arc2):
        gap_arc = arc1
    else:
        gap_arc = arc2

    # Ensure the arc goes from end_pt to start_pt
    if np.linalg.norm(gap_arc[0] - end_pt) > np.linalg.norm(gap_arc[-1] - end_pt):
        gap_arc = gap_arc[::-1]

    return gap_arc


# ═══════════════════════════════════════════════════════════════════════
# Mirror / Replicate
# ═══════════════════════════════════════════════════════════════════════

def mirror_bezier_path(
    path: BezierPath, axis: str = "vertical",
    center: Optional[np.ndarray] = None,
) -> BezierPath:
    """Reflect a Bézier path across a vertical or horizontal axis.

    Parameters
    ----------
    path : BezierPath
    axis : str
        "vertical" reflects across the Y axis (flips X).
        "horizontal" reflects across the X axis (flips Y).
    center : np.ndarray or None
        Centre of reflection.  Defaults to the path's centroid.
    """
    if center is None:
        all_pts = path.sample(pts_per_segment=20)
        center = all_pts.mean(axis=0) if len(all_pts) > 0 else np.zeros(2)

    new_segments = []
    for seg in path.segments:
        cp = seg.control_points.copy()
        centred = cp - center
        if axis == "vertical":
            centred[:, 0] = -centred[:, 0]
        else:
            centred[:, 1] = -centred[:, 1]
        new_cp = centred + center
        new_segments.append(BezierSegment(new_cp, source_type=seg.source_type))

    return BezierPath(new_segments, is_closed=path.is_closed,
                      source_type=path.source_type)


def replicate_motif(
    path: BezierPath,
    target_centroid: np.ndarray,
) -> BezierPath:
    """Translate a path so its centroid lands on ``target_centroid``."""
    all_pts = path.sample(pts_per_segment=20)
    if len(all_pts) == 0:
        return path
    current = all_pts.mean(axis=0)
    delta = target_centroid - current

    new_segments = []
    for seg in path.segments:
        new_cp = seg.control_points.copy() + delta
        new_segments.append(BezierSegment(new_cp, source_type=seg.source_type))

    return BezierPath(new_segments, is_closed=path.is_closed,
                      source_type=path.source_type)


# ═══════════════════════════════════════════════════════════════════════
# EFD coefficient blending
# ═══════════════════════════════════════════════════════════════════════

def blend_efd_completion(
    coeffs_partial: np.ndarray,
    coeffs_template: np.ndarray,
    overlap_fraction: float = 0.5,
) -> np.ndarray:
    """Weighted blend of two EFD coefficient matrices.

    Parameters
    ----------
    coeffs_partial  : (order, 4) — EFD of the partial contour
    coeffs_template : (order, 4) — EFD of a complete template
    overlap_fraction: 0→all template, 1→all partial

    Returns
    -------
    (order, 4) blended coefficients
    """
    alpha = np.clip(overlap_fraction, 0.0, 1.0)
    return alpha * coeffs_partial + (1.0 - alpha) * coeffs_template


# ═══════════════════════════════════════════════════════════════════════
# Restoration result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RestorationResult:
    """Output of the restoration execution."""
    original_paths: List[BezierPath]
    new_segments: List[BezierSegment] = field(default_factory=list)
    new_paths: List[BezierPath] = field(default_factory=list)
    efd_arcs: List[np.ndarray] = field(default_factory=list)
    actions_applied: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Execute restoration (top-level dispatch)
# ═══════════════════════════════════════════════════════════════════════

def execute_restoration(
    hypotheses: List[RankedHypothesis],
    paths: List[BezierPath],
    contour_points_map: Optional[Dict[int, np.ndarray]] = None,
) -> RestorationResult:
    """Apply the best-ranked hypothesis to produce new geometry.

    Parameters
    ----------
    hypotheses : List[RankedHypothesis]
        Ranked ASP solutions.
    paths : List[BezierPath]
        Original Bézier paths.
    contour_points_map : dict, optional
        Mapping path_id → sampled (N, 2) contour points (for EFD closure).

    Returns
    -------
    RestorationResult
    """
    result = RestorationResult(original_paths=paths)

    if not hypotheses:
        return result

    best = hypotheses[0]
    used_endpoints: set = set()

    for action in best.actions:
        try:
            if action.action_type == "extend_curve":
                _apply_extend(action, paths, result, used_endpoints)
            elif action.action_type == "complete_contour":
                _apply_complete(action, paths, result, used_endpoints,
                                contour_points_map)
            elif action.action_type == "mirror_element":
                _apply_mirror(action, paths, result)
        except Exception as exc:
            result.actions_applied.append({
                "type": "action_failed",
                "action_type": action.action_type,
                "error": str(exc),
            })

    return result


def _apply_extend(
    action: RestorationAction,
    paths: List[BezierPath],
    result: RestorationResult,
    used: set,
) -> None:
    """Apply an extend_curve action."""
    pa = action.arguments.get("path_a")
    pb = action.arguments.get("path_b")
    ea = action.arguments.get("endpoint_a", "end")
    eb = action.arguments.get("endpoint_b", "start")

    if pa is None or pb is None:
        return

    # Self-bridge check
    if pa == pb:
        result.actions_applied.append({
            "type": "action_skipped",
            "action_type": "extend_curve",
            "reason": "self_bridge_blocked",
        })
        return

    # Endpoint validity
    if ea not in ("start", "end") or eb not in ("start", "end"):
        result.actions_applied.append({
            "type": "action_failed",
            "action_type": "extend_curve",
            "error": f"invalid endpoint: {ea}, {eb}",
        })
        return

    # Out-of-range check
    if pa >= len(paths) or pb >= len(paths):
        return

    # One-use-per-endpoint
    key_a = (pa, ea)
    key_b = (pb, eb)
    if key_a in used or key_b in used:
        result.actions_applied.append({
            "type": "action_skipped",
            "action_type": "extend_curve",
            "reason": "endpoint_already_used",
        })
        return

    bridge = bridge_curves(paths[pa], paths[pb], ea, eb)
    result.new_segments.append(bridge)
    used.add(key_a)
    used.add(key_b)

    result.actions_applied.append({
        "type": "extend_curve",
        "path_a": pa,
        "path_b": pb,
        "endpoint_a": ea,
        "endpoint_b": eb,
        "confidence": action.confidence,
    })


def _apply_complete(
    action: RestorationAction,
    paths: List[BezierPath],
    result: RestorationResult,
    used: set,
    contour_points_map: Optional[Dict[int, np.ndarray]],
) -> None:
    """Apply a complete_contour action.

    First tries EFD closure (smoother for curved shapes).
    Falls back to G1 Bézier closure.
    """
    pid = action.arguments.get("contour_id")
    if pid is None or pid >= len(paths):
        return

    path = paths[pid]
    if path.is_closed:
        return

    # Check endpoints not already used
    key_start = (pid, "start")
    key_end = (pid, "end")
    if key_start in used or key_end in used:
        return

    # Try EFD closure first
    efd_arc = None
    if contour_points_map and pid in contour_points_map:
        efd_arc = efd_close_contour(contour_points_map[pid])

    if efd_arc is None:
        # EFD not available, sample the path for EFD
        sampled = path.sample(pts_per_segment=50)
        if len(sampled) >= 5:
            efd_arc = efd_close_contour(sampled)

    if efd_arc is not None and len(efd_arc) >= 2:
        result.efd_arcs.append(efd_arc)
        result.actions_applied.append({
            "type": "complete_contour",
            "path_id": pid,
            "method": "efd",
            "confidence": action.confidence,
        })
    else:
        # Fallback to G1 Bézier closure
        closure_seg = close_contour_g1(path)
        result.new_segments.append(closure_seg)
        result.actions_applied.append({
            "type": "complete_contour",
            "path_id": pid,
            "method": "bezier_g1",
            "confidence": action.confidence,
        })

    used.add(key_start)
    used.add(key_end)


def _apply_mirror(
    action: RestorationAction,
    paths: List[BezierPath],
    result: RestorationResult,
) -> None:
    """Apply a mirror_element action."""
    pid = action.arguments.get("element_id")
    axis = action.arguments.get("axis", "vertical")
    if pid is None or pid >= len(paths):
        return

    mirrored = mirror_bezier_path(paths[pid], axis=str(axis))
    result.new_paths.append(mirrored)
    result.actions_applied.append({
        "type": "mirror_element",
        "path_id": pid,
        "axis": axis,
    })
