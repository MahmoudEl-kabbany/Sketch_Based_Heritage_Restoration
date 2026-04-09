from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyefd

from bezier_curves.bezier import BezierPath, BezierSegment, _fit_cubic_single
from restoration.asp.asp_inference import RankedHypothesis

logger = logging.getLogger(__name__)


@dataclass
class RestorationConfig:
    """Configuration for geometric restoration synthesis."""

    max_actions: int = 256
    bezier_handle_scale: float = 0.33
    bezier_max_error: float = 5.0
    efd_order: int = 24
    efd_reconstruction_points: int = 360
    efd_method_min_confidence: float = 0.62
    symmetry_method_min_confidence: float = 0.55
    allow_self_extend_curve: bool = False
    output_original_name: str = "restoration_original.png"


@dataclass
class ShapeVocabEntry:
    label: str
    feature: np.ndarray


@dataclass
class ShapeVocab:
    entries: List[ShapeVocabEntry] = field(default_factory=list)


@dataclass
class AddedSegmentRecord:
    segment_id: int
    method: str
    confidence: float
    path_a: int
    path_b: int
    endpoint_a_id: int
    endpoint_b_id: int
    is_forced: bool
    reason: str
    control_points: np.ndarray


@dataclass
class RestorationResult:
    """Outputs from restoration stage."""

    original_paths: List[BezierPath]
    restored_paths: List[BezierPath]
    new_segments: List[BezierSegment] = field(default_factory=list)
    new_paths: List[BezierPath] = field(default_factory=list)
    actions_applied: List[str] = field(default_factory=list)
    additions: List[AddedSegmentRecord] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (vec / n).astype(np.float64)


def _estimate_endpoints(paths: List[BezierPath]) -> Dict[int, Dict[str, np.ndarray]]:
    endpoint_map: Dict[int, Dict[str, np.ndarray]] = {}
    endpoint_id = 0

    for path_idx, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue

        first_cp = path.segments[0].control_points.astype(np.float64)
        last_cp = path.segments[-1].control_points.astype(np.float64)

        start_point = first_cp[0]
        start_outward = -_safe_normalize(first_cp[1] - first_cp[0])
        endpoint_map[endpoint_id] = {
            "point": start_point,
            "outward": start_outward,
            "path_idx": np.array([path_idx], dtype=np.int32),
        }
        endpoint_id += 1

        end_point = last_cp[3]
        end_outward = _safe_normalize(last_cp[3] - last_cp[2])
        endpoint_map[endpoint_id] = {
            "point": end_point,
            "outward": end_outward,
            "path_idx": np.array([path_idx], dtype=np.int32),
        }
        endpoint_id += 1

    return endpoint_map


def _clone_paths(paths: List[BezierPath]) -> List[BezierPath]:
    cloned: List[BezierPath] = []
    for path in paths:
        segs = [
            BezierSegment(seg.control_points.copy(), source_type=seg.source_type)
            for seg in path.segments
        ]
        cloned.append(
            BezierPath(
                segments=segs,
                is_closed=path.is_closed,
                source_type=path.source_type,
            )
        )
    return cloned


def _estimate_global_symmetry_axis(paths: List[BezierPath]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    sampled: List[np.ndarray] = []
    for path in paths:
        if not path.segments:
            continue
        sampled.append(path.sample(35))

    if not sampled:
        return None

    points = np.vstack(sampled)
    if len(points) < 3:
        return None

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    axis = _safe_normalize(eig_vecs[:, int(np.argmax(eig_vals))])
    return centroid, axis


def _reflect_point(point: np.ndarray, axis_origin: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    normal = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    rel = point - axis_origin
    signed = float(np.dot(rel, normal))
    return point - 2.0 * signed * normal


def _project_on_axis(point: np.ndarray, axis_origin: np.ndarray, axis_unit: np.ndarray) -> np.ndarray:
    rel = point - axis_origin
    dist = float(np.dot(rel, axis_unit))
    return axis_origin + dist * axis_unit


def _segment_is_line_like(a_out: np.ndarray, b_out: np.ndarray, gap_vec: np.ndarray) -> bool:
    u = _safe_normalize(gap_vec)
    a_align = float(abs(np.dot(_safe_normalize(a_out), u)))
    b_align = float(abs(np.dot(_safe_normalize(b_out), -u)))
    return a_align > 0.95 and b_align > 0.95


def _build_bezier_bridge(
    p0: np.ndarray,
    p3: np.ndarray,
    t0_out: np.ndarray,
    t3_out: np.ndarray,
    cfg: RestorationConfig,
) -> List[BezierSegment]:
    d = float(np.linalg.norm(p3 - p0))
    if d < 1e-9:
        return []

    gap_vec = p3 - p0
    if _segment_is_line_like(t0_out, t3_out, gap_vec):
        direction = _safe_normalize(gap_vec)
        h = 0.30 * d
        p1 = p0 + h * direction
        p2 = p3 - h * direction
    else:
        h = cfg.bezier_handle_scale * d
        p1 = p0 + h * _safe_normalize(t0_out)
        p2 = p3 + h * _safe_normalize(t3_out)

    cp = np.vstack([p0, p1, p2, p3]).astype(np.float64)
    return [BezierSegment(cp, source_type="restored_bezier")]


def _extract_arc_between_indices(points: np.ndarray, i: int, j: int) -> np.ndarray:
    if i <= j:
        arc1 = points[i:j + 1]
        arc2 = np.vstack([points[j:], points[:i + 1]])
    else:
        arc1 = points[j:i + 1][::-1]
        arc2 = np.vstack([points[i:], points[:j + 1]])

    def arc_len(arc: np.ndarray) -> float:
        if len(arc) < 2:
            return float("inf")
        return float(np.sum(np.linalg.norm(np.diff(arc, axis=0), axis=1)))

    l1 = arc_len(arc1)
    l2 = arc_len(arc2)
    return arc1 if l1 <= l2 else arc2


def _build_efd_bridge(
    path: BezierPath,
    p0: np.ndarray,
    p3: np.ndarray,
    cfg: RestorationConfig,
) -> List[BezierSegment]:
    pts = path.sample(120).astype(np.float64)
    if len(pts) < 8:
        return []

    # Close for Fourier representation and reconstruct a smooth contour.
    contour = pts
    if np.linalg.norm(contour[0] - contour[-1]) > 1e-6:
        contour = np.vstack([contour, contour[0]])

    coeffs = pyefd.elliptic_fourier_descriptors(contour, order=cfg.efd_order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(contour)
    recon = pyefd.reconstruct_contour(
        coeffs,
        locus=(a0, c0),
        num_points=max(cfg.efd_reconstruction_points, 120),
    ).astype(np.float64)

    i0 = int(np.argmin(np.linalg.norm(recon - p0[None, :], axis=1)))
    i3 = int(np.argmin(np.linalg.norm(recon - p3[None, :], axis=1)))
    arc = _extract_arc_between_indices(recon, i0, i3)
    if len(arc) < 6:
        return []

    # Enforce exact anchor endpoints.
    if np.linalg.norm(arc[0] - p0) > np.linalg.norm(arc[0] - p3):
        arc = arc[::-1]
    arc = arc.copy()
    arc[0] = p0
    arc[-1] = p3

    left_tangent = _safe_normalize(arc[1] - arc[0])
    right_tangent = _safe_normalize(arc[-2] - arc[-1])

    cps_list = _fit_cubic_single(
        arc,
        left_tangent,
        right_tangent,
        max_error=cfg.bezier_max_error ** 2,
    )
    return [BezierSegment(cp.astype(np.float64), source_type="restored_efd") for cp in cps_list]


def _build_symmetry_bridge(
    p0: np.ndarray,
    p3: np.ndarray,
    t0_out: np.ndarray,
    t3_out: np.ndarray,
    axis: Optional[Tuple[np.ndarray, np.ndarray]],
    cfg: RestorationConfig,
) -> List[BezierSegment]:
    if axis is None:
        return _build_bezier_bridge(p0, p3, t0_out, t3_out, cfg)

    center, axis_u = axis
    d = float(np.linalg.norm(p3 - p0))
    if d < 1e-9:
        return []

    mid = 0.5 * (p0 + p3)
    mid_on_axis = _project_on_axis(mid, center, axis_u)
    reflected_start = _reflect_point(p0, center, axis_u)

    h = cfg.bezier_handle_scale * d
    p1 = p0 + h * _safe_normalize(t0_out)
    p2 = p3 + h * _safe_normalize(t3_out)

    # Pull handles toward a symmetry-consistent middle point.
    p1 = 0.7 * p1 + 0.3 * mid_on_axis
    p2 = 0.7 * p2 + 0.3 * mid_on_axis

    # If the reflected start already aligns with end, emphasize symmetry.
    if np.linalg.norm(reflected_start - p3) < 0.35 * d:
        p1 = 0.5 * p1 + 0.5 * mid_on_axis
        p2 = 0.5 * p2 + 0.5 * mid_on_axis

    cp = np.vstack([p0, p1, p2, p3]).astype(np.float64)
    return [BezierSegment(cp, source_type="restored_symmetry")]


def _choose_method(
    hyp: RankedHypothesis,
    same_path: bool,
    has_efd_feature: bool,
    shape_hint: str,
    cfg: RestorationConfig,
) -> str:
    hint = hyp.method.lower().strip()

    if not same_path:
        return "bezier"

    # Adaptive policy for one-gap contours: only one method is selected.
    if has_efd_feature and hyp.confidence >= cfg.efd_method_min_confidence:
        return "efd"

    if shape_hint == "symmetric" and hyp.confidence >= cfg.symmetry_method_min_confidence:
        return "symmetry"

    if hint in {"efd", "symmetry", "bezier"}:
        return hint
    return "bezier"


def _shape_hint_for_path(path: BezierPath) -> str:
    pts = path.sample(120)
    if len(pts) < 10:
        return "generic"

    # Estimate circularity from sampled polygon.
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perimeter = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) + 1e-9
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)

    if circularity > 0.72:
        return "symmetric"
    return "generic"


def build_shape_vocabulary(vocab_dir: str, config: Optional[RestorationConfig] = None) -> Optional[ShapeVocab]:
    """Load optional vocabulary descriptors from directory.

    Supported files:
      - *.json containing {'label': str, 'feature': [float, ...]}
      - *.npy containing 1-D vectors (label inferred from filename)
    """
    if not vocab_dir or not os.path.isdir(vocab_dir):
        return None

    entries: List[ShapeVocabEntry] = []
    for name in sorted(os.listdir(vocab_dir)):
        path = os.path.join(vocab_dir, name)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(name)
        try:
            if ext.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                label = str(payload.get("label", stem))
                feat = np.asarray(payload.get("feature", []), dtype=np.float64)
                if feat.ndim == 1 and feat.size > 0:
                    entries.append(ShapeVocabEntry(label=label, feature=feat))
            elif ext.lower() == ".npy":
                feat = np.load(path)
                feat = np.asarray(feat, dtype=np.float64).reshape(-1)
                if feat.size > 0:
                    entries.append(ShapeVocabEntry(label=stem, feature=feat))
        except Exception:
            continue

    if not entries:
        return None
    return ShapeVocab(entries=entries)


def execute_restoration(
    hypotheses: Sequence[RankedHypothesis],
    paths: List[BezierPath],
    efd_data: Dict[int, np.ndarray],
    vocab: Optional[ShapeVocab],
    config: Optional[RestorationConfig] = None,
) -> RestorationResult:
    """Apply ranked hypotheses to synthesize restoration geometry."""
    cfg = config or RestorationConfig()

    base_paths = _clone_paths(paths)
    endpoint_map = _estimate_endpoints(base_paths)
    symmetry_axis = _estimate_global_symmetry_axis(base_paths)

    result = RestorationResult(
        original_paths=_clone_paths(paths),
        restored_paths=_clone_paths(paths),
    )

    occupied_endpoints: set = set()
    segment_counter = 1

    ranked = sorted(hypotheses, key=lambda h: (-h.confidence, h.score, h.rank))
    for hyp in ranked[: cfg.max_actions]:
        if hyp.endpoint_a_id not in endpoint_map or hyp.endpoint_b_id not in endpoint_map:
            continue

        if hyp.endpoint_a_id in occupied_endpoints or hyp.endpoint_b_id in occupied_endpoints:
            continue

        if (not cfg.allow_self_extend_curve) and hyp.path_a == hyp.path_b and hyp.action == "connect_endpoints":
            # Self-connection is only accepted when it represents one-gap closure.
            same_path_ok = bool(hyp.metadata.get("is_single_gap", 0.0) > 0.5)
            if not same_path_ok:
                continue

        ep_a = endpoint_map[hyp.endpoint_a_id]
        ep_b = endpoint_map[hyp.endpoint_b_id]

        p0 = ep_a["point"].astype(np.float64)
        p3 = ep_b["point"].astype(np.float64)
        t0 = ep_a["outward"].astype(np.float64)
        t3 = ep_b["outward"].astype(np.float64)

        same_path = hyp.path_a == hyp.path_b
        shape_hint = "generic"
        if same_path and 0 <= hyp.path_a < len(base_paths):
            shape_hint = _shape_hint_for_path(base_paths[hyp.path_a])

        chosen_method = _choose_method(
            hyp=hyp,
            same_path=same_path,
            has_efd_feature=hyp.path_a in efd_data,
            shape_hint=shape_hint,
            cfg=cfg,
        )

        segments: List[BezierSegment] = []
        reason = f"hyp={hyp.hypothesis_id}, method={chosen_method}, conf={hyp.confidence:.3f}"

        if chosen_method == "efd" and same_path and 0 <= hyp.path_a < len(base_paths):
            try:
                segments = _build_efd_bridge(base_paths[hyp.path_a], p0, p3, cfg)
            except Exception as exc:
                logger.warning("EFD bridge failed for hypothesis %s: %s", hyp.hypothesis_id, exc)
                segments = []

            if not segments:
                # Adaptive one-method policy with deterministic fallback chain.
                if shape_hint == "symmetric":
                    chosen_method = "symmetry"
                    segments = _build_symmetry_bridge(p0, p3, t0, t3, symmetry_axis, cfg)
                else:
                    chosen_method = "bezier"
                    segments = _build_bezier_bridge(p0, p3, t0, t3, cfg)

        elif chosen_method == "symmetry":
            segments = _build_symmetry_bridge(p0, p3, t0, t3, symmetry_axis, cfg)
            if not segments:
                chosen_method = "bezier"
                segments = _build_bezier_bridge(p0, p3, t0, t3, cfg)

        else:
            chosen_method = "bezier"
            segments = _build_bezier_bridge(p0, p3, t0, t3, cfg)

        if not segments:
            continue

        new_path = BezierPath(segments=segments, is_closed=False, source_type=f"restored_{chosen_method}")
        result.new_paths.append(new_path)
        result.new_segments.extend(segments)

        for seg in segments:
            record = AddedSegmentRecord(
                segment_id=segment_counter,
                method=chosen_method,
                confidence=float(hyp.confidence),
                path_a=hyp.path_a,
                path_b=hyp.path_b,
                endpoint_a_id=hyp.endpoint_a_id,
                endpoint_b_id=hyp.endpoint_b_id,
                is_forced=bool(hyp.is_forced),
                reason=reason,
                control_points=seg.control_points.copy(),
            )
            result.additions.append(record)
            result.actions_applied.append(
                f"segment#{segment_counter}: {chosen_method} p{hyp.path_a}->{hyp.path_b} "
                f"e{hyp.endpoint_a_id}-e{hyp.endpoint_b_id} conf={hyp.confidence:.3f}"
            )
            segment_counter += 1

        occupied_endpoints.add(hyp.endpoint_a_id)
        occupied_endpoints.add(hyp.endpoint_b_id)

    result.restored_paths.extend(result.new_paths)
    result.metadata["num_hypotheses"] = float(len(hypotheses))
    result.metadata["num_actions"] = float(len(result.actions_applied))
    result.metadata["num_forced"] = float(sum(1 for a in result.additions if a.is_forced))

    return result


def _draw_segment_label(
    image: np.ndarray,
    seg: BezierSegment,
    label: str,
    color: Tuple[int, int, int],
) -> None:
    pts = np.round(seg.sample(90)).astype(np.int32)
    if len(pts) < 2:
        return

    poly = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [poly], False, color, 2, lineType=cv2.LINE_AA)

    mid = pts[len(pts) // 2]
    text_org = (int(mid[0]) + 4, int(mid[1]) - 4)
    cv2.putText(
        image,
        label,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )


def visualise(result: RestorationResult, image_path: str, output_path: str) -> None:
    """Save required outputs: original image and overlay with numbered additions."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    original_path = os.path.join(out_dir, "restoration_original.png")
    cv2.imwrite(original_path, img)

    overlay = img.copy()
    method_color = {
        "bezier": (0, 220, 0),
        "efd": (0, 170, 255),
        "symmetry": (255, 140, 0),
    }

    # Draw newly added segments only, each with explicit index label.
    for idx, rec in enumerate(result.additions, start=1):
        seg = BezierSegment(rec.control_points.copy(), source_type=f"overlay_{rec.method}")
        color = method_color.get(rec.method, (0, 220, 0))
        _draw_segment_label(overlay, seg, str(idx), color)

    cv2.imwrite(output_path, overlay)

    manifest = {
        "original_image": original_path,
        "overlay_image": output_path,
        "num_additions": len(result.additions),
        "additions": [
            {
                "segment_id": rec.segment_id,
                "method": rec.method,
                "confidence": rec.confidence,
                "path_a": rec.path_a,
                "path_b": rec.path_b,
                "endpoint_a_id": rec.endpoint_a_id,
                "endpoint_b_id": rec.endpoint_b_id,
                "is_forced": rec.is_forced,
                "reason": rec.reason,
                "control_points": rec.control_points.tolist(),
            }
            for rec in result.additions
        ],
    }
    manifest_path = os.path.join(out_dir, "restoration_additions.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
