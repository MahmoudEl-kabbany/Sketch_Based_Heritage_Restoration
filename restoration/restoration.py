"""
Geometric Synthesis  (R-4)
==========================
Translates ASP restoration atoms into concrete BezierPath / EFD
completions:  G1 contour closure, curve bridging, affine reflection,
motif replication, and EFD coefficient blending with a shape vocabulary.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ──────────────────────────────────────────────────────
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bezier_curves.bezier import BezierPath, BezierSegment
from eliptic_fourier_descriptors.efd import compute_efd_features, reconstruct_contour_efd

logger = logging.getLogger(__name__)

# ── optional FAISS -------------------------------------------------------
try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None
    logger.info("faiss-cpu not installed; falling back to scipy KDTree for kNN")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RestorationConfig:
    """Tuneable parameters for the geometric-synthesis stage."""

    # G1 closure
    g1_alpha_initial_scale: float = 1.0 / 3.0
    """Initial α = scale * |Pn − P0|."""
    g1_newton_iterations: int = 10
    """Number of Newton-Raphson iterations for α refinement."""
    g1_newton_step: float = 0.01
    """Finite-difference step for numerical Jacobian."""

    # EFD blending
    efd_order: int = 10
    """Default EFD harmonic order."""

    # Shape vocabulary
    vocab_efd_order: int = 10
    vocab_k: int = 3
    """Number of nearest neighbours to retrieve."""

    # Visualisation
    vis_dpi: int = 150
    vis_figsize: Tuple[int, int] = (14, 6)


# ═══════════════════════════════════════════════════════════════════════════
# Result data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VocabMatch:
    """A matched template from the shape vocabulary."""

    label: str
    distance: float
    coeffs: np.ndarray


@dataclass
class ShapeVocab:
    """Pre-computed shape vocabulary with FAISS / KDTree index."""

    labels: List[str] = field(default_factory=list)
    features: Optional[np.ndarray] = None  # (N, D)
    coeffs_list: List[np.ndarray] = field(default_factory=list)
    _index: Any = None  # faiss.IndexFlatL2 or scipy KDTree


@dataclass
class RestorationResult:
    """Aggregated output of the synthesis stage."""

    new_segments: List[BezierSegment] = field(default_factory=list)
    new_paths: List[BezierPath] = field(default_factory=list)
    blended_coeffs: List[np.ndarray] = field(default_factory=list)
    actions_applied: List[Dict[str, Any]] = field(default_factory=list)
    original_paths: List[BezierPath] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 1. G1-continuous contour closure
# ═══════════════════════════════════════════════════════════════════════════

def close_contour_g1(
    path: BezierPath,
    config: Optional[RestorationConfig] = None,
) -> BezierSegment:
    """Synthesise a G1-continuous closing segment for an open path.

    Constructs a cubic Bézier from ``path.segments[-1].end`` to
    ``path.segments[0].start`` whose tangents match the path endpoints.
    Uses a chord-fraction heuristic (α = chord / 3) for robust results
    on heritage sketches, with optional curvature-continuity refinement.

    Parameters
    ----------
    path : BezierPath
        An open path (``is_closed=False``).
    config : RestorationConfig, optional

    Returns
    -------
    BezierSegment
        The closing segment.

    Examples
    --------
    >>> cp = np.array([[0,0],[1,1],[2,1],[3,0]], dtype=np.float64)
    >>> seg = BezierSegment(cp)
    >>> p = BezierPath([seg], is_closed=False)
    >>> closing = close_contour_g1(p)
    >>> closing.control_points.shape
    (4, 2)
    """
    cfg = config or RestorationConfig()

    last_seg = path.segments[-1]
    first_seg = path.segments[0]

    P0 = last_seg.control_points[3].copy()  # start of closing segment
    Pn = first_seg.control_points[0].copy()  # end of closing segment

    # Tangent at P0: direction of last segment's P2→P3
    T0 = P0 - last_seg.control_points[2]
    norm_t0 = np.linalg.norm(T0)
    if norm_t0 > 1e-12:
        T0 = T0 / norm_t0
    else:
        T0 = np.array([1.0, 0.0], dtype=np.float64)

    # Tangent at Pn: direction of first segment's P0→P1 (reversed)
    T1 = first_seg.control_points[0] - first_seg.control_points[1]
    norm_t1 = np.linalg.norm(T1)
    if norm_t1 > 1e-12:
        T1 = T1 / norm_t1
    else:
        T1 = np.array([-1.0, 0.0], dtype=np.float64)

    chord = float(np.linalg.norm(Pn - P0))
    # Chord-fraction heuristic: α = chord * scale (default 1/3)
    alpha = chord * cfg.g1_alpha_initial_scale
    # Floor alpha to avoid degenerate zero-length handles
    alpha = max(alpha, 1.0)

    P1 = P0 + alpha * T0
    P2 = Pn + alpha * T1
    cp = np.vstack([P0, P1, P2, Pn]).astype(np.float64)
    return BezierSegment(cp, source_type="closure")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Curve bridging
# ═══════════════════════════════════════════════════════════════════════════

def _get_endpoint_and_tangent(
    path: BezierPath,
    which: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (point, unit_tangent) for a path's 'start' or 'end'."""
    if which == "start":
        pt = path.segments[0].control_points[0].copy()
        # Tangent at start: reversed P0→P1 direction
        tan = path.segments[0].control_points[0] - path.segments[0].control_points[1]
    else:  # "end"
        pt = path.segments[-1].control_points[3].copy()
        # Tangent at end: P2→P3 direction
        tan = path.segments[-1].control_points[3] - path.segments[-1].control_points[2]
    n = np.linalg.norm(tan)
    if n > 1e-12:
        tan = tan / n
    else:
        tan = np.array([1.0, 0.0], dtype=np.float64)
    return pt, tan


def bridge_curves(
    path_a: BezierPath,
    path_b: BezierPath,
    config: Optional[RestorationConfig] = None,
    endpoint_a: str = "end",
    endpoint_b: str = "start",
) -> BezierSegment:
    """Fit a single cubic Bézier bridging two path endpoints.

    By default bridges ``end(A) → start(B)`` but can connect any
    pair of endpoints when *endpoint_a* / *endpoint_b* are specified.

    Parameters
    ----------
    path_a, path_b : BezierPath
    config : RestorationConfig, optional
    endpoint_a : str
        ``'start'`` or ``'end'`` of *path_a*.
    endpoint_b : str
        ``'start'`` or ``'end'`` of *path_b*.

    Returns
    -------
    BezierSegment
    """
    cfg = config or RestorationConfig()

    P0, T0 = _get_endpoint_and_tangent(path_a, endpoint_a)
    P3, T3 = _get_endpoint_and_tangent(path_b, endpoint_b)

    chord = float(np.linalg.norm(P3 - P0))
    alpha = max(chord * cfg.g1_alpha_initial_scale, 1.0)

    P1 = P0 + alpha * T0
    P2 = P3 + alpha * T3
    cp = np.vstack([P0, P1, P2, P3]).astype(np.float64)
    return BezierSegment(cp, source_type="bridge")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Mirror (affine reflection)
# ═══════════════════════════════════════════════════════════════════════════

def mirror_bezier_path(
    path: BezierPath,
    axis: str = "vertical",
    origin: Optional[np.ndarray] = None,
) -> BezierPath:
    """Reflect all control points across *axis* through *origin*.

    Parameters
    ----------
    path : BezierPath
    axis : str
        ``'vertical'`` (x-reflection), ``'horizontal'`` (y-reflection),
        or a float angle in degrees.
    origin : np.ndarray, optional
        Reflection centre.  Defaults to the path centroid.

    Returns
    -------
    BezierPath

    Examples
    --------
    >>> cp = np.array([[0,0],[1,1],[2,1],[3,0]], dtype=np.float64)
    >>> mirrored = mirror_bezier_path(BezierPath([BezierSegment(cp)]))
    >>> mirrored.segments[0].control_points.shape
    (4, 2)
    """
    pts = path.sample(10)
    if origin is None:
        origin = pts.mean(axis=0) if len(pts) > 0 else np.zeros(2, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    # Build 2×2 reflection matrix
    if axis == "vertical":
        theta = np.pi / 2.0
    elif axis == "horizontal":
        theta = 0.0
    else:
        try:
            theta = np.radians(float(axis))
        except (ValueError, TypeError):
            theta = np.pi / 2.0

    cos2 = np.cos(2 * theta)
    sin2 = np.sin(2 * theta)
    R = np.array([[cos2, sin2], [sin2, -cos2]], dtype=np.float64)

    new_segments: List[BezierSegment] = []
    for seg in path.segments:
        cp = seg.control_points.astype(np.float64)
        centered = cp - origin
        reflected = (R @ centered.T).T + origin
        new_segments.append(
            BezierSegment(
                reflected.astype(np.float64),
                source_type="mirror",
            )
        )

    return BezierPath(
        new_segments, is_closed=path.is_closed, source_type="mirror"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Motif replication
# ═══════════════════════════════════════════════════════════════════════════

def replicate_motif(
    source_path: BezierPath,
    target_pos: np.ndarray,
    scale: float = 1.0,
) -> BezierPath:
    """Clone a BezierPath, translate its centroid to *target_pos*, and scale.

    Parameters
    ----------
    source_path : BezierPath
    target_pos : np.ndarray, shape (2,)
    scale : float

    Returns
    -------
    BezierPath

    Examples
    --------
    >>> cp = np.array([[0,0],[1,0],[2,0],[3,0]], dtype=np.float64)
    >>> rep = replicate_motif(BezierPath([BezierSegment(cp)]), np.array([10,10]))
    >>> rep.segments[0].control_points[0, 0] > 5
    True
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    pts = source_path.sample(20)
    centroid = pts.mean(axis=0) if len(pts) > 0 else np.zeros(2, dtype=np.float64)

    new_segments: List[BezierSegment] = []
    for seg in source_path.segments:
        cp = seg.control_points.astype(np.float64)
        centered = cp - centroid
        scaled = centered * scale
        translated = scaled + target_pos
        new_segments.append(
            BezierSegment(translated.astype(np.float64), source_type="replicate")
        )

    return BezierPath(
        new_segments, is_closed=source_path.is_closed, source_type="replicate"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. EFD coefficient blending
# ═══════════════════════════════════════════════════════════════════════════

def blend_efd_completion(
    partial_coeffs: np.ndarray,
    vocab_coeffs: np.ndarray,
    overlap_fraction: float = 0.5,
) -> np.ndarray:
    """Lambda-blend partial and retrieved EFD coefficients.

    ``C_restored = λ·C_partial + (1-λ)·C_retrieved``
    where ``λ = overlap_fraction``.

    Parameters
    ----------
    partial_coeffs : np.ndarray, shape (order, 4)
    vocab_coeffs : np.ndarray, shape (order, 4)
    overlap_fraction : float
        0 = full retrieval, 1 = full partial.

    Returns
    -------
    np.ndarray, shape (order, 4)

    Examples
    --------
    >>> a = np.ones((10, 4), dtype=np.float64)
    >>> b = np.zeros((10, 4), dtype=np.float64)
    >>> blended = blend_efd_completion(a, b, 0.5)
    >>> np.allclose(blended, 0.5)
    True
    """
    lam = float(np.clip(overlap_fraction, 0.0, 1.0))
    p = np.asarray(partial_coeffs, dtype=np.float64)
    v = np.asarray(vocab_coeffs, dtype=np.float64)

    # Pad to same shape if needed
    max_order = max(p.shape[0], v.shape[0])
    if p.shape[0] < max_order:
        p = np.vstack([p, np.zeros((max_order - p.shape[0], p.shape[1]), dtype=np.float64)])
    if v.shape[0] < max_order:
        v = np.vstack([v, np.zeros((max_order - v.shape[0], v.shape[1]), dtype=np.float64)])

    return lam * p + (1.0 - lam) * v


# ═══════════════════════════════════════════════════════════════════════════
# 6. Shape vocabulary
# ═══════════════════════════════════════════════════════════════════════════

def build_shape_vocabulary(
    shapes_dir: str,
    config: Optional[RestorationConfig] = None,
) -> ShapeVocab:
    """Pre-compute normalised EFD features for all images in *shapes_dir*.

    Parameters
    ----------
    shapes_dir : str
        Directory containing ``.jpg``/``.png`` reference shapes.
    config : RestorationConfig, optional

    Returns
    -------
    ShapeVocab

    Examples
    --------
    >>> vocab = build_shape_vocabulary("/nonexistent")
    >>> len(vocab.labels) == 0
    True
    """
    cfg = config or RestorationConfig()
    vocab = ShapeVocab()

    if not os.path.isdir(shapes_dir):
        logger.warning("Shapes directory not found: %s", shapes_dir)
        return vocab

    img_paths = sorted(
        glob.glob(os.path.join(shapes_dir, "*.png"))
        + glob.glob(os.path.join(shapes_dir, "*.jpg"))
        + glob.glob(os.path.join(shapes_dir, "*.jpeg"))
    )

    feat_list: List[np.ndarray] = []
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # Use the largest contour
            largest = max(contours, key=cv2.contourArea)
            feat = compute_efd_features(largest, order=cfg.vocab_efd_order)
            if feat is None:
                continue

            import pyefd
            raw_coeffs = pyefd.elliptic_fourier_descriptors(
                np.squeeze(largest), order=cfg.vocab_efd_order, normalize=False
            )

            vocab.labels.append(os.path.splitext(os.path.basename(img_path))[0])
            feat_list.append(feat)
            vocab.coeffs_list.append(raw_coeffs)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", img_path, exc)

    if not feat_list:
        return vocab

    features = np.vstack(feat_list).astype(np.float32)
    vocab.features = features

    if faiss is not None:
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)
        vocab._index = index
    else:
        from scipy.spatial import cKDTree

        vocab._index = cKDTree(features)

    logger.info("Shape vocabulary: %d entries, %d-dim features", len(vocab.labels), features.shape[1])
    return vocab


def query_shape_vocabulary(
    partial_coeffs: np.ndarray,
    vocab: ShapeVocab,
    k: int = 3,
) -> List[VocabMatch]:
    """kNN query in EFD feature space.

    Parameters
    ----------
    partial_coeffs : np.ndarray
        Flat EFD feature vector.
    vocab : ShapeVocab
    k : int

    Returns
    -------
    List[VocabMatch]

    Examples
    --------
    >>> query_shape_vocabulary(np.zeros(37), ShapeVocab())
    []
    """
    if vocab.features is None or vocab._index is None:
        return []

    query = np.asarray(partial_coeffs, dtype=np.float32).ravel()
    if len(query) != vocab.features.shape[1]:
        # Pad or truncate
        target_dim = vocab.features.shape[1]
        if len(query) < target_dim:
            query = np.pad(query, (0, target_dim - len(query)))
        else:
            query = query[:target_dim]

    k = min(k, len(vocab.labels))
    if k == 0:
        return []

    matches: List[VocabMatch] = []

    if faiss is not None and hasattr(vocab._index, "search"):
        D, I = vocab._index.search(query.reshape(1, -1), k)
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            matches.append(
                VocabMatch(
                    label=vocab.labels[idx],
                    distance=float(dist),
                    coeffs=vocab.coeffs_list[idx],
                )
            )
    else:
        # scipy KDTree
        dists, idxs = vocab._index.query(query, k=k)
        if np.isscalar(dists):
            dists = [dists]
            idxs = [idxs]
        for dist, idx in zip(dists, idxs):
            if idx >= len(vocab.labels):
                continue
            matches.append(
                VocabMatch(
                    label=vocab.labels[idx],
                    distance=float(dist),
                    coeffs=vocab.coeffs_list[idx],
                )
            )

    return matches


# ═══════════════════════════════════════════════════════════════════════════
# 7. Execute restoration (dispatch)
# ═══════════════════════════════════════════════════════════════════════════

def execute_restoration(
    hypotheses: List[Any],
    paths: List[BezierPath],
    efd_data: Optional[Dict[int, np.ndarray]] = None,
    vocab: Optional[ShapeVocab] = None,
    config: Optional[RestorationConfig] = None,
) -> RestorationResult:
    """Dispatch each ASP restoration atom to the correct synthesis function.

    Parameters
    ----------
    hypotheses : List[RankedHypothesis]
        Ranked ASP solutions.
    paths : List[BezierPath]
    efd_data : dict, optional
    vocab : ShapeVocab, optional
    config : RestorationConfig, optional

    Returns
    -------
    RestorationResult

    Examples
    --------
    >>> execute_restoration([], []).new_segments
    []
    """
    cfg = config or RestorationConfig()
    result = RestorationResult(original_paths=list(paths))

    if not hypotheses:
        return result

    # Use the top-ranked hypothesis
    best = hypotheses[0]

    for action in best.actions:
        try:
            if action.action_type == "complete_contour":
                cid = action.arguments.get("contour_id", 0)
                if 0 <= cid < len(paths) and not paths[cid].is_closed:
                    seg = close_contour_g1(paths[cid], cfg)
                    result.new_segments.append(seg)
                    result.actions_applied.append(
                        {"type": "complete_contour", "contour_id": cid}
                    )
                    logger.info("Applied complete_contour for contour %d", cid)

            elif action.action_type == "extend_curve":
                pa = action.arguments.get("path_a", 0)
                pb = action.arguments.get("path_b", 0)
                ep_a = action.arguments.get("endpoint_a", "end")
                ep_b = action.arguments.get("endpoint_b", "start")
                if 0 <= pa < len(paths) and 0 <= pb < len(paths):
                    seg = bridge_curves(
                        paths[pa], paths[pb], cfg,
                        endpoint_a=ep_a, endpoint_b=ep_b,
                    )
                    result.new_segments.append(seg)
                    result.actions_applied.append(
                        {"type": "extend_curve", "path_a": pa, "path_b": pb,
                         "endpoint_a": ep_a, "endpoint_b": ep_b}
                    )
                    logger.info("Applied extend_curve %d(%s)→%d(%s)", pa, ep_a, pb, ep_b)

            elif action.action_type == "mirror_element":
                eid = action.arguments.get("element_id", 0)
                axis = action.arguments.get("axis", "vertical")
                if 0 <= eid < len(paths):
                    mirrored = mirror_bezier_path(paths[eid], axis=str(axis))
                    result.new_paths.append(mirrored)
                    result.actions_applied.append(
                        {"type": "mirror_element", "element_id": eid, "axis": axis}
                    )
                    logger.info("Applied mirror_element %d across %s", eid, axis)

            elif action.action_type == "replicate_motif":
                mid = action.arguments.get("motif_id", 0)
                pos = action.arguments.get("position", 0)
                if 0 <= mid < len(paths):
                    target = np.array([float(pos), 0.0], dtype=np.float64)
                    rep = replicate_motif(paths[mid], target)
                    result.new_paths.append(rep)
                    result.actions_applied.append(
                        {"type": "replicate_motif", "motif_id": mid, "position": pos}
                    )
                    logger.info("Applied replicate_motif %d → pos %s", mid, pos)

            elif action.action_type == "flag_similar_missing":
                pa = action.arguments.get("path_a", 0)
                pb = action.arguments.get("path_b", 0)
                if efd_data and vocab:
                    feat = efd_data.get(pa)
                    if feat is not None:
                        matches = query_shape_vocabulary(feat, vocab, k=1)
                        if matches:
                            blended = blend_efd_completion(
                                feat.reshape(-1, 4) if feat.ndim == 1 else feat,
                                matches[0].coeffs,
                                overlap_fraction=0.5,
                            )
                            result.blended_coeffs.append(blended)
                            result.actions_applied.append(
                                {"type": "efd_blend", "path_a": pa, "match": matches[0].label}
                            )

        except Exception as exc:
            logger.error("Action %s failed: %s", action.action_type, exc)

    logger.info(
        "Restoration complete: %d new segments, %d new paths, %d blended coeffs",
        len(result.new_segments), len(result.new_paths), len(result.blended_coeffs),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 8. Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def visualise(
    result: RestorationResult,
    image_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config: Optional[RestorationConfig] = None,
) -> None:
    """Save an exact copy of the image and one with new paths overlaid."""
    cfg = config or RestorationConfig()

    img_bgr = None
    if image_path and os.path.exists(image_path):
        img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        logger.warning("No valid image_path provided to visualise, creating blank canvas.")
        img_bgr = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # 1. Save original image unaltered
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        orig_path = output_path.replace("visualisation", "original") if "visualisation" in output_path else output_path.replace(".png", "_orig.png")
        cv2.imwrite(orig_path, img_bgr)
        logger.info("Saved original visualisation to %s", orig_path)

    # 2. Draw restored overlays
    overlaid = img_bgr.copy()

    # Vibrant native BGR colors for drawing new paths cleanly
    vibrant_colors = [
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 0),     # Lime Green
        (255, 128, 0),   # Cyan
        (0, 165, 255),   # Orange
        (200, 100, 255), # Pink/Purple
    ]

    # Draw new segments
    for idx, seg in enumerate(result.new_segments):
        pts = seg.sample(200) # Heavy sampling for smooth curve
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        c = vibrant_colors[idx % len(vibrant_colors)]
        cv2.polylines(overlaid, [pts_int], False, c, thickness=3, lineType=cv2.LINE_AA)

    # Draw new paths (merged components)
    for idx, path in enumerate(result.new_paths):
        pts = path.sample(200)
        if len(pts) == 0:
            continue
        c = vibrant_colors[(idx + len(result.new_segments)) % len(vibrant_colors)]
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlaid, [pts_int], False, c, thickness=3, lineType=cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, overlaid)
        logger.info("Saved restored visualisation to %s", output_path)
    else:
        logger.info("No output_path specified, visualisations not saved.")
