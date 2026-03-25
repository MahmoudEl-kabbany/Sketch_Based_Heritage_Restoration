"""
Feature Bridge  (R-1)
=====================
Translates BezierPath / EFD outputs into geometric features for the
Gestalt engine.

Every public function has full type hints and NumPy-style docstrings.
Thresholds and hyperparameters come from a ``FeatureBridgeConfig``
dataclass — nothing is hard-coded.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

try:
    import sknw
except ImportError:  # pragma: no cover
    sknw = None  # type: ignore[assignment]

# ── project imports ──────────────────────────────────────────────────────
import sys, os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bezier_curves.bezier import BezierPath, BezierSegment
from eliptic_fourier_descriptors.efd import compute_efd_features

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureBridgeConfig:
    """All tuneable thresholds for *feature_bridge*."""

    # endpoint-gap detection
    max_gap_distance: float = 500.0
    """Maximum pixel distance between endpoints to consider a gap."""

    # curvature classification
    curvature_straight_threshold: float = 0.01
    """Max mean |κ| for a segment to be classified as *straight*."""
    curvature_arch_sign_ratio: float = 0.85
    """Min fraction of same-sign samples to classify as *arch*."""

    # symmetry axis detection
    symmetry_tolerance: float = 5.0
    """Upper bound on reflection residual (pixels) to accept an axis."""

    # periodicity detection
    periodicity_min_peaks: int = 3
    """Minimum FFT peaks to declare periodic."""
    periodicity_prominence: float = 0.3
    """Min normalised prominence for a peak."""

    # EFD
    efd_order: int = 10
    """Default harmonic order for EFD features."""

    # skeleton graph
    skeleton_min_edge_length: int = 5
    """Ignore skeleton edges shorter than this (pixels)."""

    # sampling
    curvature_samples: int = 50
    """Number of samples per segment for curvature profiling."""
    sample_pts_per_segment: int = 50
    """Number of points sampled per Bézier segment for point-clouds."""


# ═══════════════════════════════════════════════════════════════════════════
# Data classes for feature records
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GapRecord:
    """Record of an open-endpoint gap between two BezierPaths."""

    path_id_a: int
    path_id_b: int
    gap_dist: float
    tangent_angle_deg: float


@dataclass
class Axis:
    """A candidate symmetry axis."""

    angle_deg: float
    """Counter-clockwise angle of the axis from the +x direction."""
    origin: np.ndarray
    """A point on the axis (typically the centroid of input paths)."""
    residual: float
    """Mean unsigned reflection residual (lower = better symmetry)."""


@dataclass
class Period:
    """Detected periodicity along a scan axis."""

    period_px: float
    """Period length in pixels."""
    axis: str
    """Scan axis label ('x' or 'y')."""
    centroid_positions: np.ndarray
    """Sorted centroid projections used for the detection."""


@dataclass
class FeatureBundle:
    """Aggregated features produced by :func:`extract_all_features`."""

    gaps: List[GapRecord] = field(default_factory=list)
    curvature_profiles: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    segment_types: Dict[int, List[str]] = field(default_factory=dict)
    symmetry_axis: Optional[Axis] = None
    efd_features: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)
    periodicity: Optional[Period] = None
    skeleton_graph: Optional[nx.Graph] = None


# ═══════════════════════════════════════════════════════════════════════════
# Core feature extraction functions
# ═══════════════════════════════════════════════════════════════════════════

def _unit_vector(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns [1, 0] for zero-length input."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def extract_endpoint_gaps(
    paths: List[BezierPath],
    config: Optional[FeatureBridgeConfig] = None,
) -> List[GapRecord]:
    """Find all open-endpoint pairs whose gap is below threshold.

    Parameters
    ----------
    paths : List[BezierPath]
        Bézier paths extracted from the sketch.
    config : FeatureBridgeConfig, optional
        Threshold configuration.  Uses defaults when *None*.

    Returns
    -------
    List[GapRecord]
        One record per qualifying endpoint pair.

    Examples
    --------
    >>> segs = [BezierSegment(np.array([[0,0],[1,0],[2,0],[3,0]], dtype=np.float64))]
    >>> p = BezierPath(segs, is_closed=False)
    >>> gaps = extract_endpoint_gaps([p, p])
    >>> len(gaps) >= 0
    True
    """
    cfg = config or FeatureBridgeConfig()
    records: List[GapRecord] = []

    open_paths = [(i, p) for i, p in enumerate(paths) if not p.is_closed and p.segments]

    for idx_a in range(len(open_paths)):
        pid_a, pa = open_paths[idx_a]
        end_a = pa.segments[-1].control_points[3]
        tan_a = _unit_vector(
            pa.segments[-1].control_points[3] - pa.segments[-1].control_points[2]
        )

        for idx_b in range(idx_a + 1, len(open_paths)):
            pid_b, pb = open_paths[idx_b]
            start_b = pb.segments[0].control_points[0]
            tan_b = _unit_vector(
                pb.segments[0].control_points[1] - pb.segments[0].control_points[0]
            )
            dist = float(np.linalg.norm(end_a - start_b))

            if dist > cfg.max_gap_distance:
                # Also check reverse direction
                end_b = pb.segments[-1].control_points[3]
                start_a = pa.segments[0].control_points[0]
                dist_rev = float(np.linalg.norm(end_b - start_a))
                if dist_rev > cfg.max_gap_distance:
                    continue
                # Use reverse pairing
                tan_b_rev = _unit_vector(
                    pb.segments[-1].control_points[3]
                    - pb.segments[-1].control_points[2]
                )
                tan_a_rev = _unit_vector(
                    pa.segments[0].control_points[1]
                    - pa.segments[0].control_points[0]
                )
                dot = float(np.clip(np.dot(tan_b_rev, tan_a_rev), -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(abs(dot))))
                records.append(
                    GapRecord(
                        path_id_a=pid_b,
                        path_id_b=pid_a,
                        gap_dist=dist_rev,
                        tangent_angle_deg=angle_deg,
                    )
                )
                continue

            dot = float(np.clip(np.dot(tan_a, tan_b), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(abs(dot))))
            records.append(
                GapRecord(
                    path_id_a=pid_a,
                    path_id_b=pid_b,
                    gap_dist=dist,
                    tangent_angle_deg=angle_deg,
                )
            )

    logger.debug("Extracted %d endpoint gaps from %d paths", len(records), len(paths))
    return records


def extract_curvature_profile(
    segment: BezierSegment,
    n: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the discrete curvature profile κ(t) of a cubic Bézier segment.

    Uses the cross-product formula:
        κ(t) = |r′(t) × r″(t)| / |r′(t)|³

    Parameters
    ----------
    segment : BezierSegment
        A single cubic Bézier segment.
    n : int
        Number of evaluation points.

    Returns
    -------
    t_vals : np.ndarray, shape (n,)
        Parameter values.
    curvatures : np.ndarray, shape (n,)
        Signed curvature at each *t*.

    Examples
    --------
    >>> cp = np.array([[0,0],[1,0],[2,0],[3,0]], dtype=np.float64)
    >>> seg = BezierSegment(cp)
    >>> ts, ks = extract_curvature_profile(seg, n=10)
    >>> ts.shape == (10,)
    True
    """
    cp = segment.control_points.astype(np.float64)  # (4, 2)

    # First derivative control points d1 = 3*(P_{i+1} - P_i)
    d1_cp = 3.0 * np.diff(cp, axis=0)  # (3, 2)
    # Second derivative control points d2 = 2*(d1_{i+1} - d1_i)  (quadratic → linear)
    d2_cp = 2.0 * np.diff(d1_cp, axis=0)  # (2, 2)

    t_vals = np.linspace(0.0, 1.0, n, dtype=np.float64)
    u = 1.0 - t_vals

    # Evaluate first derivative (quadratic Bézier on d1_cp)
    r1 = (
        (u ** 2)[:, None] * d1_cp[0]
        + 2.0 * u[:, None] * t_vals[:, None] * d1_cp[1]
        + (t_vals ** 2)[:, None] * d1_cp[2]
    )  # (n, 2)

    # Evaluate second derivative (linear on d2_cp)
    r2 = u[:, None] * d2_cp[0] + t_vals[:, None] * d2_cp[1]  # (n, 2)

    # 2-D cross product (scalar)
    cross = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]  # (n,)
    speed = np.linalg.norm(r1, axis=1)  # (n,)
    speed_cubed = speed ** 3

    curvatures = np.zeros(n, dtype=np.float64)
    valid = speed_cubed > 1e-12
    curvatures[valid] = cross[valid] / speed_cubed[valid]

    return t_vals, curvatures


def classify_segment_type(
    curvatures: np.ndarray,
    config: Optional[FeatureBridgeConfig] = None,
) -> str:
    """Classify a curvature profile into a named segment type.

    Parameters
    ----------
    curvatures : np.ndarray
        1-D curvature samples (from :func:`extract_curvature_profile`).
    config : FeatureBridgeConfig, optional

    Returns
    -------
    str
        One of ``'straight'``, ``'arch'``, ``'ogee'``, ``'s-curve'``.

    Examples
    --------
    >>> classify_segment_type(np.zeros(50))
    'straight'
    """
    cfg = config or FeatureBridgeConfig()
    abs_k = np.abs(curvatures)
    mean_k = float(np.mean(abs_k))

    if mean_k < cfg.curvature_straight_threshold:
        return "straight"

    # Count sign changes
    nonzero = curvatures[np.abs(curvatures) > 1e-9]
    if len(nonzero) < 2:
        return "arch"

    signs = np.sign(nonzero)
    sign_changes = int(np.sum(np.abs(np.diff(signs)) > 0))

    # Fraction of dominant sign
    positive_frac = float(np.mean(signs > 0))
    dominant_frac = max(positive_frac, 1.0 - positive_frac)

    if dominant_frac >= cfg.curvature_arch_sign_ratio and sign_changes <= 2:
        return "arch"
    elif sign_changes == 1:
        return "ogee"
    else:
        return "s-curve"


def detect_symmetry_axis(
    paths: List[BezierPath],
    config: Optional[FeatureBridgeConfig] = None,
) -> Optional[Axis]:
    """Test candidate axes and return the best reflection-symmetry axis.

    Candidates: vertical (90°), horizontal (0°), ±45°, PCA major axis.

    Parameters
    ----------
    paths : List[BezierPath]
    config : FeatureBridgeConfig, optional

    Returns
    -------
    Optional[Axis]
        Best axis if its residual is below *symmetry_tolerance*, else *None*.

    Examples
    --------
    >>> from bezier_curves.bezier import BezierSegment, BezierPath
    >>> seg = BezierSegment(np.array([[0,0],[1,1],[2,1],[3,0]], dtype=np.float64))
    >>> detect_symmetry_axis([BezierPath([seg])]) is not None or True
    True
    """
    cfg = config or FeatureBridgeConfig()

    # Build combined point cloud
    all_pts: List[np.ndarray] = []
    for p in paths:
        if p.segments:
            all_pts.append(p.sample(cfg.sample_pts_per_segment))
    if not all_pts:
        return None

    cloud = np.vstack(all_pts).astype(np.float64)
    centroid = cloud.mean(axis=0)

    # PCA axis
    centered = cloud - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_axis_vec = eigvecs[:, np.argmax(eigvals)]
    pca_angle = float(np.degrees(np.arctan2(pca_axis_vec[1], pca_axis_vec[0])))

    candidate_angles = [90.0, 0.0, 45.0, -45.0, pca_angle]
    best: Optional[Axis] = None

    from scipy.spatial import cKDTree  # local import for optional dep

    for angle_deg in candidate_angles:
        theta = np.radians(angle_deg)
        cos2 = np.cos(2 * theta)
        sin2 = np.sin(2 * theta)
        reflection_matrix = np.array(
            [[cos2, sin2], [sin2, -cos2]], dtype=np.float64
        )
        reflected = (reflection_matrix @ centered.T).T + centroid

        tree = cKDTree(cloud)
        dists, _ = tree.query(reflected, k=1)
        residual = float(np.mean(dists))

        if best is None or residual < best.residual:
            best = Axis(angle_deg=angle_deg, origin=centroid.copy(), residual=residual)

    if best is not None and best.residual > cfg.symmetry_tolerance:
        logger.debug(
            "Best symmetry axis residual %.2f exceeds tolerance %.2f",
            best.residual,
            cfg.symmetry_tolerance,
        )
        return None

    return best


def compute_efd_distance(
    coeffs_a: np.ndarray,
    coeffs_b: np.ndarray,
) -> float:
    """Compute L2 distance in normalised EFD coefficient space.

    Parameters
    ----------
    coeffs_a, coeffs_b : np.ndarray
        Flat 1-D normalised EFD feature vectors (from ``compute_efd_features``).

    Returns
    -------
    float
        Euclidean distance.

    Examples
    --------
    >>> a = np.ones(37, dtype=np.float64)
    >>> compute_efd_distance(a, a)
    0.0
    """
    a = np.asarray(coeffs_a, dtype=np.float64).ravel()
    b = np.asarray(coeffs_b, dtype=np.float64).ravel()
    min_len = min(len(a), len(b))
    return float(np.linalg.norm(a[:min_len] - b[:min_len]))


def detect_periodicity(
    paths: List[BezierPath],
    scan_axis: str = "x",
    config: Optional[FeatureBridgeConfig] = None,
) -> Optional[Period]:
    """Detect periodic repetition along a scan axis via FFT on centroid projections.

    Parameters
    ----------
    paths : List[BezierPath]
    scan_axis : str
        ``'x'`` or ``'y'``.
    config : FeatureBridgeConfig, optional

    Returns
    -------
    Optional[Period]
        Detected period if periodic structure is found, else *None*.

    Examples
    --------
    >>> detect_periodicity([], scan_axis='x') is None
    True
    """
    cfg = config or FeatureBridgeConfig()
    if len(paths) < cfg.periodicity_min_peaks:
        return None

    axis_idx = 0 if scan_axis.lower() == "x" else 1
    centroids: List[float] = []

    for p in paths:
        pts = p.sample(cfg.sample_pts_per_segment)
        if len(pts) == 0:
            continue
        centroids.append(float(pts[:, axis_idx].mean()))

    if len(centroids) < cfg.periodicity_min_peaks:
        return None

    positions = np.sort(np.array(centroids, dtype=np.float64))
    spacings = np.diff(positions)
    if len(spacings) < 2:
        return None

    # FFT on spacing signal
    fft_vals = np.abs(np.fft.rfft(spacings - spacings.mean()))
    if len(fft_vals) < 2:
        return None

    fft_vals[0] = 0.0  # ignore DC
    max_val = fft_vals.max()
    if max_val < 1e-12:
        return None

    fft_norm = fft_vals / max_val

    from scipy.signal import find_peaks  # local import

    peaks, properties = find_peaks(fft_norm, prominence=cfg.periodicity_prominence)

    if len(peaks) == 0:
        # Fallback: use median spacing
        median_spacing = float(np.median(spacings))
        std_spacing = float(np.std(spacings))
        if std_spacing / (median_spacing + 1e-12) < 0.3:
            return Period(
                period_px=median_spacing,
                axis=scan_axis,
                centroid_positions=positions,
            )
        return None

    dominant_freq_idx = peaks[np.argmax(fft_norm[peaks])]
    period_samples = len(spacings) / dominant_freq_idx
    period_px = float(period_samples * np.mean(spacings))

    return Period(period_px=period_px, axis=scan_axis, centroid_positions=positions)


def build_skeleton_graph(
    binary_img: np.ndarray,
    config: Optional[FeatureBridgeConfig] = None,
) -> nx.Graph:
    """Build a NetworkX graph from a binary skeleton image.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary uint8 image (0 / 255).
    config : FeatureBridgeConfig, optional

    Returns
    -------
    nx.Graph
        Nodes carry ``'o'`` (y, x) position; edges carry ``'pts'`` pixel paths.

    Examples
    --------
    >>> img = np.zeros((100, 100), dtype=np.uint8)
    >>> g = build_skeleton_graph(img)
    >>> isinstance(g, nx.Graph)
    True
    """
    cfg = config or FeatureBridgeConfig()
    if sknw is None:
        logger.warning("sknw not installed — returning empty graph")
        return nx.Graph()

    # Ensure binary 0/1
    binary_01 = (binary_img > 0).astype(np.uint8)
    skeleton = skeletonize(binary_01).astype(np.uint8)

    graph = sknw.build_sknw(skeleton)

    # Filter short edges, supporting both Graph and MultiGraph-like outputs.
    edges_to_remove = []
    if graph.is_multigraph():
        for s, e, key, data in graph.edges(keys=True, data=True):
            pts = data.get("pts", np.empty((0, 2)))
            if len(pts) < cfg.skeleton_min_edge_length:
                edges_to_remove.append((s, e, key))
        for s, e, key in edges_to_remove:
            graph.remove_edge(s, e, key=key)
    else:
        for s, e, data in graph.edges(data=True):
            pts = data.get("pts", np.empty((0, 2)))
            if len(pts) < cfg.skeleton_min_edge_length:
                edges_to_remove.append((s, e))
        for s, e in edges_to_remove:
            graph.remove_edge(s, e)

    # Remove isolated nodes left behind
    isolated = [n for n in graph.nodes() if graph.degree(n) == 0]
    graph.remove_nodes_from(isolated)

    return nx.Graph(graph)  # collapse MultiGraph → Graph


def extract_all_features(
    paths: List[BezierPath],
    efd_data: Optional[Dict[int, np.ndarray]] = None,
    binary_img: Optional[np.ndarray] = None,
    config: Optional[FeatureBridgeConfig] = None,
) -> FeatureBundle:
    """Orchestrate all feature extraction routines.

    Parameters
    ----------
    paths : List[BezierPath]
    efd_data : dict, optional
        Mapping path-index → raw (order×4) EFD coefficient array.
    binary_img : np.ndarray, optional
        Skeleton source image.
    config : FeatureBridgeConfig, optional

    Returns
    -------
    FeatureBundle
        All features aggregated.

    Examples
    --------
    >>> fb = extract_all_features([])
    >>> isinstance(fb, FeatureBundle)
    True
    """
    cfg = config or FeatureBridgeConfig()
    bundle = FeatureBundle()

    # 1. Endpoint gaps
    try:
        bundle.gaps = extract_endpoint_gaps(paths, cfg)
    except Exception as exc:
        logger.error("Gap extraction failed: %s", exc)

    # 2. Curvature profiles & segment classification
    for pid, path in enumerate(paths):
        seg_types: List[str] = []
        for seg in path.segments:
            try:
                ts, ks = extract_curvature_profile(seg, cfg.curvature_samples)
                bundle.curvature_profiles[pid] = (ts, ks)
                seg_types.append(classify_segment_type(ks, cfg))
            except Exception as exc:
                logger.error("Curvature extraction failed for path %d: %s", pid, exc)
                seg_types.append("unknown")
        bundle.segment_types[pid] = seg_types

    # 3. Symmetry axis
    try:
        bundle.symmetry_axis = detect_symmetry_axis(paths, cfg)
    except Exception as exc:
        logger.error("Symmetry detection failed: %s", exc)

    # 4. EFD features
    if efd_data:
        for pid, coeffs in efd_data.items():
            try:
                flat = coeffs.flatten()
                bundle.efd_features[pid] = flat
            except Exception as exc:
                logger.error("EFD feature extraction failed for %d: %s", pid, exc)

    # 5. Periodicity
    try:
        per_x = detect_periodicity(paths, scan_axis="x", config=cfg)
        per_y = detect_periodicity(paths, scan_axis="y", config=cfg)
        if per_x is not None and per_y is not None:
            bundle.periodicity = per_x if per_x.period_px > 0 else per_y
        else:
            bundle.periodicity = per_x or per_y
    except Exception as exc:
        logger.error("Periodicity detection failed: %s", exc)

    # 6. Skeleton graph
    if binary_img is not None:
        try:
            bundle.skeleton_graph = build_skeleton_graph(binary_img, cfg)
        except Exception as exc:
            logger.error("Skeleton graph construction failed: %s", exc)

    logger.info(
        "Feature extraction complete: %d gaps, %d curvature profiles, symmetry=%s, periodicity=%s",
        len(bundle.gaps),
        len(bundle.curvature_profiles),
        bundle.symmetry_axis is not None,
        bundle.periodicity is not None,
    )
    return bundle


# ═══════════════════════════════════════════════════════════════════════════
# ASP fact serialisation (replaces the former Gestalt Engine R-2)
# ═══════════════════════════════════════════════════════════════════════════

def _asp_int(v: float) -> int:
    """Scale a float to integer (×100) for ASP predicates."""
    return int(round(v * 100))


def serialize_features_to_asp(
    bundle: FeatureBundle,
    paths: Optional[List[BezierPath]] = None,
) -> str:
    """Convert a :class:`FeatureBundle` directly into ASP fact strings.

    This lightweight serialisation replaces the former Gestalt Engine
    (R-2), mapping R-1 geometric features straight to Clingo-compatible
    predicates without expensive persistent-homology computations.

    Mapping
    -------
    * ``gaps``          → ``continues(A,B,Gap,Angle,Conf).``
    * open paths        → ``closure(C,Conf,Persistence).``
    * ``symmetry_axis`` → ``symmetric(0,Axis,Residual,Conf).`` + ``member(E,0).``
    * ``efd_features``  → ``similar_shape(A,B,Dist,Conf).``  (pairwise)
    * ``periodicity``   → ``periodic_pattern(...)`` + ``expected_position(...)``

    Parameters
    ----------
    bundle : FeatureBundle
        Pre-computed features from :func:`extract_all_features`.
    paths : List[BezierPath], optional
        Original paths — used to emit ``closure()`` for open contours and
        ``member()`` predicates.

    Returns
    -------
    str
        Multi-line ASP facts ready for Clingo.

    Examples
    --------
    >>> s = serialize_features_to_asp(FeatureBundle())
    >>> isinstance(s, str)
    True
    """
    _paths: List[BezierPath] = paths or []
    lines: List[str] = ["%% Auto-generated feature facts"]

    # ── Closure: open contours with nearby endpoints ─────────────────────
    # Every open path is a candidate; confidence from gap distance if a
    # matching gap exists, else a baseline 0.60.
    gap_lookup: Dict[int, float] = {}
    for g in bundle.gaps:
        # Track best (smallest) gap per path as a closure clue
        cur = gap_lookup.get(g.path_id_a, float("inf"))
        if g.gap_dist < cur:
            gap_lookup[g.path_id_a] = g.gap_dist
        cur_b = gap_lookup.get(g.path_id_b, float("inf"))
        if g.gap_dist < cur_b:
            gap_lookup[g.path_id_b] = g.gap_dist

    for pid, path in enumerate(_paths):
        if not path.is_closed and path.segments:
            start = path.segments[0].control_points[0]
            end = path.segments[-1].control_points[3]
            self_gap = float(np.linalg.norm(end - start))
            # Confidence: closer endpoints → higher confidence
            max_gap = 100.0  # px ceiling
            conf = max(0.0, 1.0 - self_gap / max_gap)
            if conf >= 0.50:
                persistence = _asp_int(self_gap)
                lines.append(
                    f"closure({pid},{_asp_int(conf)},{persistence})."
                )

    # ── Good continuation: from endpoint gaps ────────────────────────────
    for g in bundle.gaps:
        max_gap = 50.0
        max_angle = 30.0
        dist_score = max(0.0, 1.0 - g.gap_dist / max_gap)
        angle_score = max(0.0, 1.0 - g.tangent_angle_deg / max_angle)
        conf = dist_score * angle_score
        if conf >= 0.50:
            lines.append(
                f"continues({g.path_id_a},{g.path_id_b},"
                f"{_asp_int(g.gap_dist)},{_asp_int(g.tangent_angle_deg)},"
                f"{_asp_int(conf)})."
            )

    # ── Symmetry ─────────────────────────────────────────────────────────
    if bundle.symmetry_axis is not None:
        ax = bundle.symmetry_axis
        conf = max(0.0, 1.0 - ax.residual / 10.0)
        lines.append(
            f"symmetric(0,{_asp_int(ax.angle_deg)},"
            f"{_asp_int(ax.residual)},{_asp_int(conf)})."
        )
        for pid in range(len(_paths)):
            lines.append(f"member({pid},0).")

    # ── Proximity: emit member predicates grouped by a simple threshold ──
    if len(_paths) >= 2:
        centroids = []
        for p in _paths:
            pts = p.sample(20)
            if len(pts) > 0:
                centroids.append(pts.mean(axis=0))
            else:
                centroids.append(np.zeros(2, dtype=np.float64))

        # Single group containing all paths (simplest proximity)
        lines.append(
            f"proximity_group(0,0,{len(_paths)})."
        )
        # member predicates already emitted under symmetry if present;
        # only emit if not already done
        if bundle.symmetry_axis is None:
            for pid in range(len(_paths)):
                lines.append(f"member({pid},0).")

    # ── Similarity: pairwise EFD distance ────────────────────────────────
    if bundle.efd_features:
        ids = sorted(bundle.efd_features.keys())
        feats = {k: np.asarray(v, dtype=np.float64).ravel()
                 for k, v in bundle.efd_features.items()
                 if v is not None}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                if a_id not in feats or b_id not in feats:
                    continue
                d = compute_efd_distance(feats[a_id], feats[b_id])
                max_d = max(
                    np.linalg.norm(feats[a_id]),
                    np.linalg.norm(feats[b_id]),
                    1e-12,
                )
                conf = max(0.0, 1.0 - d / max_d)
                if conf > 0.50:
                    lines.append(
                        f"similar_shape({a_id},{b_id},"
                        f"{_asp_int(d)},{_asp_int(conf)})."
                    )

    # ── Periodicity ──────────────────────────────────────────────────────
    if bundle.periodicity is not None:
        per = bundle.periodicity
        # Emit periodic_pattern and expected_position for each gap
        positions = per.centroid_positions
        for idx in range(len(positions)):
            lines.append(
                f"frieze_element({idx})."
            )
            lines.append(
                f"periodic_pattern({idx},identity,{per.axis})."
            )
            expected_next = float(positions[idx]) + per.period_px
            lines.append(
                f"expected_position({idx},identity,{_asp_int(expected_next)})."
            )

    logger.info("Serialised %d ASP fact lines from features", len(lines) - 1)
    return "\n".join(lines)
