"""
R-1: Feature Extraction / Feature Bridge
==========================================
Extracts geometric features from Bézier paths and produces ASP-ready
fact strings for the inference engine.

Features extracted
------------------
* Endpoint gaps (proximity + good-continuation scoring)
* Curvature profiles and segment classification
* Reflective symmetry detection
* EFD distance between contours
* Periodicity / motif repetition

All thresholds are collected in ``FeatureBridgeConfig`` so the
pipeline can scale them to image resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
import sknw

import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bezier_curves.bezier import BezierPath, BezierSegment


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FeatureBridgeConfig:
    """All tuneable thresholds for feature extraction."""
    max_gap_distance: float = 80.0       # px — max distance for a candidate gap
    min_gap_px: float = 2.0              # px — gaps smaller than this are already connected
    continuation_max_angle_deg: float = 45.0  # max deviation from tangent continuation
    min_continues_conf: float = 0.30     # min confidence for a continues() fact
    min_closure_conf: float = 0.35       # min confidence for a closure() fact
    min_closure_gap: float = 3.0         # px — gap must be at least this to attempt closure
    max_closure_gap_fraction: float = 0.35  # max fraction of contour that can be a gap
    symmetry_tolerance: float = 15.0     # px — reflective symmetry tolerance
    efd_similarity_threshold: float = 2.0  # max L2 for "similar_shape" fact
    pragnanz_circularity_threshold: float = 0.75  # min circularity for pragnanz fact


# ═══════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GapCandidate:
    """A candidate gap between two path endpoints."""
    path_id_a: int
    path_id_b: int
    endpoint_a: str            # "start" | "end"
    endpoint_b: str            # "start" | "end"
    point_a: np.ndarray        # (x, y) of the endpoint
    point_b: np.ndarray        # (x, y) of the endpoint
    tangent_a: np.ndarray      # unit tangent pointing outward from the contour
    tangent_b: np.ndarray      # unit tangent pointing outward from the contour
    gap_dist: float            # Euclidean distance
    continuation_angle: float  # degrees between tangent extension and gap vector
    confidence: float = 0.0    # overall continuation confidence


@dataclass
class FeatureBundle:
    """Container for all extracted features."""
    gaps: List[GapCandidate] = field(default_factory=list)
    curvature_profiles: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    segment_types: Dict[int, List[str]] = field(default_factory=dict)
    symmetry_axis: Optional[Tuple[float, float, float]] = None  # (cx, cy, angle_deg)
    efd_data: Dict[int, np.ndarray] = field(default_factory=dict)
    path_lengths: Dict[int, float] = field(default_factory=dict)
    closure_candidates: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Endpoint gap detection
# ═══════════════════════════════════════════════════════════════════════

def _endpoint_tangent(path: BezierPath, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (point, outward_tangent) for 'start' or 'end' of a path."""
    if which == "start":
        seg = path.segments[0]
        pt = seg.control_points[0].copy()
        handle = seg.control_points[1]
        tangent = pt - handle  # outward = away from interior
    else:
        seg = path.segments[-1]
        pt = seg.control_points[3].copy()
        handle = seg.control_points[2]
        tangent = pt - handle  # outward = away from interior
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        tangent = np.array([1.0, 0.0])
    else:
        tangent = tangent / norm
    return pt, tangent


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors [0, 180]."""
    cos_val = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _score_continuation(
    pt_a: np.ndarray, tan_a: np.ndarray,
    pt_b: np.ndarray, tan_b: np.ndarray,
    gap_dist: float,
    config: FeatureBridgeConfig,
) -> Tuple[float, float]:
    """Score how well tangent at A continues toward B.

    Returns (angle_degrees, confidence).
    """
    gap_vec = pt_b - pt_a
    gap_norm = np.linalg.norm(gap_vec)
    if gap_norm < 1e-12:
        return 0.0, 1.0
    gap_dir = gap_vec / gap_norm

    # Angle between tangent_a (outward) and gap direction
    angle_a = _angle_between(tan_a, gap_dir)

    # Angle between tangent_b (outward) and reverse gap direction
    angle_b = _angle_between(tan_b, -gap_dir)

    # Combined angle: average of both endpoint angles
    combined_angle = (angle_a + angle_b) / 2.0

    if combined_angle > config.continuation_max_angle_deg:
        return combined_angle, 0.0

    # Confidence: 1.0 at 0°, linearly decaying to 0.0 at max_angle
    angle_score = 1.0 - (combined_angle / config.continuation_max_angle_deg)

    # Proximity score: 1.0 at 0 distance, decaying to 0.0 at max_gap_distance
    prox_score = max(0.0, 1.0 - (gap_dist / config.max_gap_distance))

    # Direction compatibility: both tangents should roughly face each other
    facing = np.dot(tan_a, tan_b)
    # Two tangents facing each other → dot product ≈ -1
    # Same direction → dot product ≈ +1 (bad for continuation)
    facing_score = max(0.0, (-facing + 1.0) / 2.0)  # maps -1→1, 0→0.5, 1→0

    confidence = angle_score * prox_score * facing_score
    return combined_angle, confidence


def extract_endpoint_gaps(
    paths: List[BezierPath],
    adjacency: Optional[Dict[int, set]] = None,
    config: Optional[FeatureBridgeConfig] = None,
) -> List[GapCandidate]:
    """Find candidate gaps between open path endpoints.

    For each pair of open paths, evaluates all 4 endpoint pairings
    and emits the best one.  Enforces each endpoint is used at most
    once across all gaps (greedy nearest-first assignment).
    """
    if config is None:
        config = FeatureBridgeConfig()
    if adjacency is None:
        adjacency = {}

    open_paths = [(i, p) for i, p in enumerate(paths)
                  if not p.is_closed and p.segments]

    if len(open_paths) < 2:
        # Check for self-closure candidates
        return []

    # Collect all candidates
    all_candidates: List[GapCandidate] = []

    for idx_a in range(len(open_paths)):
        pid_a, path_a = open_paths[idx_a]
        for idx_b in range(idx_a + 1, len(open_paths)):
            pid_b, path_b = open_paths[idx_b]

            # Skip if already connected via skeleton adjacency
            if pid_b in adjacency.get(pid_a, set()):
                continue

            best_candidate = None
            best_conf = -1.0

            for ep_a in ("start", "end"):
                pt_a, tan_a = _endpoint_tangent(path_a, ep_a)
                for ep_b in ("start", "end"):
                    pt_b, tan_b = _endpoint_tangent(path_b, ep_b)

                    gap_dist = float(np.linalg.norm(pt_a - pt_b))

                    if gap_dist > config.max_gap_distance:
                        continue
                    if gap_dist < config.min_gap_px:
                        continue

                    angle, conf = _score_continuation(
                        pt_a, tan_a, pt_b, tan_b, gap_dist, config
                    )

                    if conf < config.min_continues_conf:
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_candidate = GapCandidate(
                            path_id_a=pid_a,
                            path_id_b=pid_b,
                            endpoint_a=ep_a,
                            endpoint_b=ep_b,
                            point_a=pt_a,
                            point_b=pt_b,
                            tangent_a=tan_a,
                            tangent_b=tan_b,
                            gap_dist=gap_dist,
                            continuation_angle=angle,
                            confidence=conf,
                        )

            if best_candidate is not None:
                all_candidates.append(best_candidate)

    # Greedy assignment: sort by confidence descending, each endpoint used once
    all_candidates.sort(key=lambda c: c.confidence, reverse=True)
    used_endpoints: set = set()
    result: List[GapCandidate] = []

    for cand in all_candidates:
        key_a = (cand.path_id_a, cand.endpoint_a)
        key_b = (cand.path_id_b, cand.endpoint_b)
        if key_a in used_endpoints or key_b in used_endpoints:
            continue
        used_endpoints.add(key_a)
        used_endpoints.add(key_b)
        result.append(cand)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Curvature and segment classification
# ═══════════════════════════════════════════════════════════════════════

def extract_curvature_profile(
    segment: BezierSegment, n: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute signed curvature κ(t) at *n* sample points.

    Uses the formula κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    for a parametric curve.
    """
    cp = segment.control_points  # (4, 2)

    # Derivative control points
    d1 = 3.0 * np.diff(cp, axis=0)   # (3, 2) — first derivative CP
    d2 = 2.0 * np.diff(d1, axis=0)   # (2, 2) — second derivative CP

    ts = np.linspace(0.0, 1.0, n)
    u = 1.0 - ts

    # First derivative (quadratic Bezier with d1 CPs)
    dx = (u**2)[:, None] * d1[0] + 2*u[:, None]*ts[:, None]*d1[1] + (ts**2)[:, None]*d1[2]

    # Second derivative (linear Bezier with d2 CPs)
    ddx = u[:, None] * d2[0] + ts[:, None] * d2[1]

    # Signed curvature
    cross = dx[:, 0] * ddx[:, 1] - dx[:, 1] * ddx[:, 0]
    speed_sq = dx[:, 0]**2 + dx[:, 1]**2
    speed_cubed = np.power(speed_sq, 1.5)

    kappa = np.zeros(n)
    valid = speed_cubed > 1e-12
    kappa[valid] = cross[valid] / speed_cubed[valid]

    return ts, kappa


def classify_segment_type(curvature: np.ndarray, threshold: float = 0.01) -> str:
    """Classify a segment from its curvature profile."""
    abs_k = np.abs(curvature)
    max_k = float(abs_k.max()) if len(abs_k) > 0 else 0.0
    mean_k = float(abs_k.mean()) if len(abs_k) > 0 else 0.0

    if max_k < threshold:
        return "straight"

    # Check for sign changes (S-curve)
    signs = np.sign(curvature[curvature != 0])
    if len(signs) > 1:
        sign_changes = int(np.sum(np.abs(np.diff(signs)) > 0))
        if sign_changes >= 2:
            return "s_curve"

    if mean_k > threshold:
        return "arch"

    return "complex"


# ═══════════════════════════════════════════════════════════════════════
# Symmetry detection
# ═══════════════════════════════════════════════════════════════════════

def detect_symmetry_axis(
    paths: List[BezierPath],
    config: Optional[FeatureBridgeConfig] = None,
) -> Optional[Tuple[float, float, float]]:
    """Detect a reflective symmetry axis from sampled path points.

    Returns (cx, cy, angle_degrees) or None.
    """
    if config is None:
        config = FeatureBridgeConfig()
    if not paths:
        return None

    # Sample all paths to get a point cloud
    all_pts = []
    for p in paths:
        pts = p.sample(pts_per_segment=30)
        if len(pts) > 0:
            all_pts.append(pts)
    if not all_pts:
        return None

    cloud = np.vstack(all_pts)
    if len(cloud) < 6:
        return None

    centroid = cloud.mean(axis=0)
    centered = cloud - centroid

    # PCA to find principal axes
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Test each principal axis as a candidate symmetry axis
    for axis_idx in range(2):
        axis = eigenvectors[:, axis_idx]
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))

        # Reflect all points across this axis
        reflected = _reflect_points(centered, axis)

        # Build KD-tree for nearest-neighbour matching
        tree = KDTree(centered)
        dists, _ = tree.query(reflected)
        median_err = float(np.median(dists))

        if median_err < config.symmetry_tolerance:
            return (float(centroid[0]), float(centroid[1]), angle)

    return None


def _reflect_points(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Reflect 2D points across an axis through the origin."""
    # Reflection matrix: R = 2 * (a⊗a) - I
    a = axis / np.linalg.norm(axis)
    R = 2.0 * np.outer(a, a) - np.eye(2)
    return (R @ points.T).T


# ═══════════════════════════════════════════════════════════════════════
# EFD distance
# ═══════════════════════════════════════════════════════════════════════

def compute_efd_distance(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """L2 distance between two normalised EFD feature vectors."""
    return float(np.linalg.norm(feat_a - feat_b))


# ═══════════════════════════════════════════════════════════════════════
# Periodicity detection
# ═══════════════════════════════════════════════════════════════════════

def detect_periodicity(
    paths: List[BezierPath],
    efd_data: Optional[Dict[int, np.ndarray]] = None,
    config: Optional[FeatureBridgeConfig] = None,
) -> List[Tuple[int, int]]:
    """Detect pairs of paths that look like repeated motifs (EFD similarity)."""
    if config is None:
        config = FeatureBridgeConfig()
    if efd_data is None or len(efd_data) < 2:
        return []

    ids = sorted(efd_data.keys())
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = compute_efd_distance(efd_data[ids[i]], efd_data[ids[j]])
            if d < config.efd_similarity_threshold:
                pairs.append((ids[i], ids[j]))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Skeleton graph
# ═══════════════════════════════════════════════════════════════════════

def build_skeleton_graph(binary_image: np.ndarray):
    """Build a networkx graph from a binary image via skeletonisation."""
    import networkx as nx
    if binary_image is None or binary_image.max() == 0:
        return nx.Graph()
    skel = skeletonize(binary_image > 0).astype(np.uint8)
    graph = sknw.build_sknw(skel)
    return graph


# ═══════════════════════════════════════════════════════════════════════
# Closure candidates (single-gap contours)
# ═══════════════════════════════════════════════════════════════════════

def _detect_closure_candidates(
    paths: List[BezierPath],
    config: FeatureBridgeConfig,
) -> List[dict]:
    """Find open paths whose endpoints are close enough for EFD closure."""
    candidates = []
    for pid, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue

        start_pt, start_tan = _endpoint_tangent(path, "start")
        end_pt, end_tan = _endpoint_tangent(path, "end")
        gap_dist = float(np.linalg.norm(start_pt - end_pt))

        if gap_dist < config.min_closure_gap:
            continue

        # Estimate path length
        sampled = path.sample(pts_per_segment=50)
        if len(sampled) < 5:
            continue
        diffs = np.diff(sampled, axis=0)
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

        if path_len < 1e-6:
            continue

        gap_fraction = gap_dist / path_len
        if gap_fraction > config.max_closure_gap_fraction:
            continue

        # Closure confidence based on gap fraction (smaller fraction = higher confidence)
        conf = max(0.0, 1.0 - (gap_fraction / config.max_closure_gap_fraction))

        # Boost confidence if endpoints' tangents point toward each other
        gap_vec = start_pt - end_pt
        gap_norm = np.linalg.norm(gap_vec)
        if gap_norm > 1e-12:
            gap_dir = gap_vec / gap_norm
            angle_end = _angle_between(end_tan, gap_dir)
            angle_start = _angle_between(start_tan, -gap_dir)
            avg_angle = (angle_end + angle_start) / 2.0
            if avg_angle < 90:
                angle_boost = 1.0 - (avg_angle / 90.0)
                conf = min(1.0, conf + 0.2 * angle_boost)

        if conf >= config.min_closure_conf:
            candidates.append({
                "path_id": pid,
                "gap_dist": gap_dist,
                "path_length": path_len,
                "gap_fraction": gap_fraction,
                "confidence": conf,
            })

    return candidates


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def extract_all_features(
    paths: List[BezierPath],
    adjacency: Optional[Dict[int, set]] = None,
    efd_data: Optional[Dict[int, np.ndarray]] = None,
    config: Optional[FeatureBridgeConfig] = None,
) -> FeatureBundle:
    """Extract all features for the ASP inference engine."""
    if config is None:
        config = FeatureBridgeConfig()

    bundle = FeatureBundle()

    # Endpoint gaps
    bundle.gaps = extract_endpoint_gaps(paths, adjacency, config)

    # Curvature profiles and segment types
    for pid, path in enumerate(paths):
        profiles = []
        types = []
        for seg in path.segments:
            ts, ks = extract_curvature_profile(seg)
            profiles.append((ts, ks))
            types.append(classify_segment_type(ks))
        bundle.curvature_profiles[pid] = profiles
        bundle.segment_types[pid] = types

    # Path lengths
    for pid, path in enumerate(paths):
        sampled = path.sample(pts_per_segment=50)
        if len(sampled) >= 2:
            diffs = np.diff(sampled, axis=0)
            bundle.path_lengths[pid] = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            bundle.path_lengths[pid] = 0.0

    # Symmetry
    bundle.symmetry_axis = detect_symmetry_axis(paths, config)

    # EFD data
    if efd_data:
        bundle.efd_data = efd_data

    # Closure candidates
    bundle.closure_candidates = _detect_closure_candidates(paths, config)

    return bundle


# ═══════════════════════════════════════════════════════════════════════
# ASP serialization
# ═══════════════════════════════════════════════════════════════════════

def serialize_features_to_asp(
    bundle: FeatureBundle,
    paths: Optional[List[BezierPath]] = None,
    config: Optional[FeatureBridgeConfig] = None,
) -> str:
    """Convert extracted features to an ASP fact string for clingo."""
    if config is None:
        config = FeatureBridgeConfig()

    lines = ["% Auto-generated ASP facts from feature extraction", ""]

    # Dynamic threshold facts (scaled to integers for ASP)
    lines.append(f"max_gap_limit({int(config.max_gap_distance * 100)}).")
    lines.append(f"max_angle_limit({int(config.continuation_max_angle_deg * 100)}).")
    lines.append(f"min_continue_conf({int(config.min_continues_conf * 100)}).")
    lines.append(f"min_closure_conf({int(config.min_closure_conf * 100)}).")
    lines.append("")

    # Path facts
    if paths:
        for pid, path in enumerate(paths):
            lines.append(f"path({pid}).")
            if path.is_closed:
                lines.append(f"observed_closed({pid}).")
            else:
                lines.append(f"open_path({pid}).")
        lines.append("")

    # Gap / continuation facts
    for gap in bundle.gaps:
        ea = 1 if gap.endpoint_a == "end" else 0
        eb = 1 if gap.endpoint_b == "end" else 0
        conf_int = int(gap.confidence * 100)
        dist_int = int(gap.gap_dist * 100)
        angle_int = int(gap.continuation_angle * 100)
        lines.append(
            f"continues({gap.path_id_a},{gap.path_id_b},{ea},{eb},{conf_int})."
        )
        lines.append(
            f"gap_distance({gap.path_id_a},{gap.path_id_b},{dist_int})."
        )
        lines.append(
            f"continuation_angle({gap.path_id_a},{gap.path_id_b},{angle_int})."
        )
    if bundle.gaps:
        lines.append("")

    # Proximity groups
    if paths and len(paths) >= 2:
        for gap in bundle.gaps:
            lines.append(f"proximity_group({gap.path_id_a},{gap.path_id_b}).")
        lines.append("")

    # Closure candidates
    for cc in bundle.closure_candidates:
        conf_int = int(cc["confidence"] * 100)
        lines.append(f"closure({cc['path_id']},{conf_int}).")
    if bundle.closure_candidates:
        lines.append("")

    # Symmetry
    if bundle.symmetry_axis is not None:
        cx, cy, angle = bundle.symmetry_axis
        lines.append(f"symmetry_axis({int(cx)},{int(cy)},{int(angle)}).")
        lines.append("")

    # EFD similarity pairs
    if bundle.efd_data and len(bundle.efd_data) >= 2:
        ids = sorted(bundle.efd_data.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = compute_efd_distance(bundle.efd_data[ids[i]],
                                         bundle.efd_data[ids[j]])
                if d < config.efd_similarity_threshold:
                    lines.append(f"similar_shape({ids[i]},{ids[j]}).")
        lines.append("")

    # Segment types
    for pid, types in bundle.segment_types.items():
        for seg_idx, stype in enumerate(types):
            lines.append(f"segment_type({pid},{seg_idx},{stype}).")
    if bundle.segment_types:
        lines.append("")

    # Pragnanz (circularity) — emitted for paths with high area/perimeter ratio
    if paths:
        for pid, path in enumerate(paths):
            sampled = path.sample(pts_per_segment=50)
            if len(sampled) >= 5:
                area = float(cv2.contourArea(sampled.astype(np.float32).reshape(-1, 1, 2)))
                perim = float(np.sum(np.linalg.norm(np.diff(sampled, axis=0), axis=1)))
                if perim > 1e-6:
                    circularity = 4 * math.pi * area / (perim * perim)
                    if circularity > config.pragnanz_circularity_threshold:
                        lines.append(f"pragnanz({pid},{int(circularity * 100)}).")
        lines.append("")

    return "\n".join(lines)
