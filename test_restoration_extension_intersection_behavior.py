import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.scoring import score_candidates
from restoration.synthesis import synthesize_bridges


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _line_segment(p0, p3, source_type="contour") -> BezierSegment:
    p0 = np.asarray(p0, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    d = p3 - p0
    cp = np.vstack([p0, p0 + d / 3.0, p0 + 2.0 * d / 3.0, p3])
    return BezierSegment(control_points=cp, source_type=source_type)


def _path_from_points(points, source_type="contour") -> BezierPath:
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    segs = [_line_segment(pts[i], pts[i + 1], source_type=source_type) for i in range(len(pts) - 1)]
    return BezierPath(segments=segs, is_closed=False, source_type=source_type)


def _endpoint_from_path(path: BezierPath, path_index: int, end: str, endpoint_id: int) -> EndpointInfo:
    if end == "start":
        seg = path.segments[0]
        pos = seg.control_points[0]
        tangent = _normalize(seg.control_points[0] - seg.control_points[1])
    else:
        seg = path.segments[-1]
        pos = seg.control_points[3]
        tangent = _normalize(seg.control_points[3] - seg.control_points[2])
    return EndpointInfo(
        endpoint_id=endpoint_id,
        path_index=path_index,
        end=end,
        position=pos.copy(),
        tangent=tangent,
        curvature=0.0,
        tangent_confidence=1.0,
    )


def _make_result(paths, endpoints, h=300, w=300) -> ExtractionResult:
    return ExtractionResult(
        paths=paths,
        endpoints=endpoints,
        efd_contours=[],
        image_shape=(h, w),
        diagonal=float(np.hypot(h, w)),
    )


def test_same_path_extension_generated_for_arrow_like_tip():
    path = _path_from_points([
        (15.0, 80.0),
        (95.0, 30.0),
        (200.0, 30.0),
        (200.0, 130.0),
        (95.0, 130.0),
        (15.0, 100.0),
    ])
    ep_start = _endpoint_from_path(path, 0, "start", 0)
    ep_end = _endpoint_from_path(path, 0, "end", 1)
    result = _make_result([path], [ep_start, ep_end])

    candidates = generate_candidates(result, lookahead_fraction=0.22, max_per_endpoint=5)

    ext_candidates = [
        c for c in candidates
        if c.same_path_closure and c.scenario == "extension_intersection"
    ]
    assert ext_candidates
    assert ext_candidates[0].intersection_point is not None
    assert ext_candidates[0].extension_quality >= 0.35


def test_same_path_extension_scores_above_continuation_on_sharp_closure():
    path = _path_from_points([
        (10.0, 40.0),
        (95.0, 0.0),
        (200.0, 0.0),
        (200.0, 100.0),
        (95.0, 100.0),
        (10.0, 60.0),
    ])
    ep_start = _endpoint_from_path(path, 0, "start", 0)
    ep_end = _endpoint_from_path(path, 0, "end", 1)
    result = _make_result([path], [ep_start, ep_end])

    candidates = generate_candidates(result, lookahead_fraction=0.22, max_per_endpoint=5)
    scored = score_candidates(candidates, result)

    same_path = [c for c in scored if c.same_path_closure]
    ext = next(c for c in same_path if c.scenario == "extension_intersection")
    cont = next(c for c in same_path if c.scenario == "self_closure")

    assert ext.score >= cont.score


def test_low_confidence_same_path_misalignment_is_rejected():
    path = _path_from_points([
        (0.0, 0.0),
        (80.0, 0.0),
        (80.0, 20.0),
        (0.0, 20.0),
    ])
    ep_start = EndpointInfo(
        endpoint_id=0,
        path_index=0,
        end="start",
        position=np.array([0.0, 0.0], dtype=np.float64),
        tangent=np.array([-1.0, 0.0], dtype=np.float64),
        curvature=0.0,
        tangent_confidence=0.2,
    )
    ep_end = EndpointInfo(
        endpoint_id=1,
        path_index=0,
        end="end",
        position=np.array([0.0, 20.0], dtype=np.float64),
        tangent=np.array([-1.0, 0.0], dtype=np.float64),
        curvature=0.0,
        tangent_confidence=0.2,
    )
    result = _make_result([path], [ep_start, ep_end])

    candidates = generate_candidates(result, lookahead_fraction=0.20, max_per_endpoint=5)
    assert len(candidates) == 0


def test_high_curvature_weak_context_rejects_same_path_extension():
    path = _path_from_points([
        (20.0, 52.0),
        (38.0, 25.0),
        (72.0, 14.0),
        (108.0, 25.0),
        (124.0, 52.0),
        (108.0, 79.0),
        (72.0, 90.0),
        (38.0, 79.0),
        (20.0, 58.0),
    ])

    ep_start = EndpointInfo(
        endpoint_id=0,
        path_index=0,
        end="start",
        position=np.array([20.0, 52.0], dtype=np.float64),
        tangent=_normalize(np.array([-0.95, 0.31], dtype=np.float64)),
        curvature=0.015,
        tangent_confidence=0.60,
    )
    ep_end = EndpointInfo(
        endpoint_id=1,
        path_index=0,
        end="end",
        position=np.array([20.0, 58.0], dtype=np.float64),
        tangent=_normalize(np.array([-0.95, -0.31], dtype=np.float64)),
        curvature=0.016,
        tangent_confidence=0.60,
    )
    result = _make_result([path], [ep_start, ep_end])

    candidates = generate_candidates(result, lookahead_fraction=0.22, max_per_endpoint=5)
    assert any(c.scenario == "self_closure" for c in candidates)
    assert not any(c.scenario == "extension_intersection" and c.same_path_closure for c in candidates)


def test_synthesis_uses_explicit_intersection_point_metadata():
    ep_a = EndpointInfo(
        endpoint_id=0,
        path_index=0,
        end="end",
        position=np.array([0.0, 0.0], dtype=np.float64),
        tangent=np.array([1.0, 0.0], dtype=np.float64),
        curvature=0.0,
        tangent_confidence=1.0,
    )
    ep_b = EndpointInfo(
        endpoint_id=1,
        path_index=0,
        end="start",
        position=np.array([10.0, 0.0], dtype=np.float64),
        tangent=np.array([-1.0, 0.0], dtype=np.float64),
        curvature=0.0,
        tangent_confidence=1.0,
    )

    candidate = ConnectionCandidate(
        id=7,
        ep_a=ep_a,
        ep_b=ep_b,
        scenario="extension_intersection",
        bridge_points=np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64),
        bridge_bezier=[],
        distance=10.0,
        tier=1,
        bilateral_alignment=0.8,
        misalignment_deg=100.0,
        same_path_closure=True,
        intersection_point=np.array([5.0, 5.0], dtype=np.float64),
        extension_quality=0.9,
    )

    synthesize_bridges([candidate])

    assert len(candidate.bridge_bezier) == 2
    first_end = candidate.bridge_bezier[0].control_points[3]
    second_start = candidate.bridge_bezier[1].control_points[0]
    assert np.linalg.norm(first_end - np.array([5.0, 5.0])) < 1e-6
    assert np.linalg.norm(second_start - np.array([5.0, 5.0])) < 1e-6
