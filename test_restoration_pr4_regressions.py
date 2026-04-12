import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.efd_closure import close_single_gaps
from restoration.extraction import EndpointInfo, ExtractionResult
from restoration.pipeline import _sanitize_accepted_candidates
from restoration.scoring import score_candidates
from restoration.synthesis import merge_restored_paths


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
    )


def _candidate(
    cid: int,
    ep_a: EndpointInfo,
    ep_b: EndpointInfo,
    score: float = 0.5,
    scenario: str = "continuation",
    source_type: str = "bridge",
) -> ConnectionCandidate:
    bridge_seg = _line_segment(ep_a.position, ep_b.position, source_type=source_type)
    return ConnectionCandidate(
        id=cid,
        ep_a=ep_a,
        ep_b=ep_b,
        scenario=scenario,
        bridge_points=bridge_seg.sample(10),
        bridge_bezier=[bridge_seg],
        distance=float(np.linalg.norm(ep_b.position - ep_a.position)),
        tier=1,
        score=score,
    )


def _open_loop_points(center=(100.0, 100.0), rx=40.0, ry=28.0, gap_angle=0.45, n=12):
    cx, cy = center
    angles = np.linspace(gap_angle / 2.0, 2.0 * np.pi - gap_angle / 2.0, n)
    points = []
    for a in angles:
        points.append((cx + rx * np.cos(a), cy + ry * np.sin(a)))
    return points


def test_sanitize_accepted_candidates_drops_endpoint_conflicts():
    e0 = EndpointInfo(0, 0, "end", np.array([10.0, 10.0]), np.array([1.0, 0.0]), 0.0)
    e1 = EndpointInfo(1, 1, "start", np.array([20.0, 10.0]), np.array([-1.0, 0.0]), 0.0)
    e2 = EndpointInfo(2, 2, "start", np.array([20.0, 20.0]), np.array([-1.0, 0.0]), 0.0)

    c1 = _candidate(1, e0, e1, score=0.9)
    c2 = _candidate(2, e0, e2, score=0.8)

    sanitized, dropped = _sanitize_accepted_candidates([c1, c2])

    assert dropped == 1
    assert len(sanitized) == 1
    assert sanitized[0].id == 1


def test_generate_candidates_includes_same_path_self_closure():
    path0 = _path_from_points(_open_loop_points(center=(100.0, 100.0), rx=34.0, ry=24.0, gap_angle=0.40, n=10))
    path1 = _path_from_points([(122.0, 90.0), (152.0, 90.0), (172.0, 100.0), (152.0, 110.0), (122.0, 110.0)])

    eps = [
        _endpoint_from_path(path0, 0, "start", 0),
        _endpoint_from_path(path0, 0, "end", 1),
        _endpoint_from_path(path1, 1, "start", 2),
        _endpoint_from_path(path1, 1, "end", 3),
    ]

    result = ExtractionResult(
        paths=[path0, path1],
        endpoints=eps,
        efd_contours=[],
        image_shape=(240, 240),
        diagonal=float(np.hypot(240.0, 240.0)),
    )

    candidates = generate_candidates(result, lookahead_fraction=0.18, max_per_endpoint=5)
    assert any(c.same_path_closure for c in candidates)

    scored = score_candidates(candidates, result)
    assert scored[0].same_path_closure


def test_bilateral_direction_gate_blocks_one_sided_pair():
    path0 = _path_from_points([(10.0, 60.0), (50.0, 60.0)])
    path1 = _path_from_points([(80.0, 60.0), (120.0, 60.0)])

    ep_a = EndpointInfo(0, 0, "end", np.array([50.0, 60.0]), np.array([1.0, 0.0]), 0.0)
    ep_b = EndpointInfo(1, 1, "start", np.array([80.0, 60.0]), np.array([1.0, 0.0]), 0.0)

    result = ExtractionResult(
        paths=[path0, path1],
        endpoints=[ep_a, ep_b],
        efd_contours=[],
        image_shape=(200, 200),
        diagonal=float(np.hypot(200.0, 200.0)),
    )

    candidates = generate_candidates(result, lookahead_fraction=0.16, max_per_endpoint=5)
    assert len(candidates) == 0


def test_merge_skips_conflicting_connection_without_orphan_bridge_path():
    p0 = _path_from_points([(0.0, 0.0), (20.0, 0.0)])
    p1 = _path_from_points([(30.0, 0.0), (50.0, 0.0)])
    p2 = _path_from_points([(30.0, 20.0), (50.0, 20.0)])

    e0 = EndpointInfo(0, 0, "end", np.array([20.0, 0.0]), np.array([1.0, 0.0]), 0.0)
    e1 = EndpointInfo(1, 1, "start", np.array([30.0, 0.0]), np.array([-1.0, 0.0]), 0.0)
    e2 = EndpointInfo(2, 2, "start", np.array([30.0, 20.0]), np.array([-1.0, 0.0]), 0.0)

    c1 = _candidate(1, e0, e1, score=0.9)
    c2 = _candidate(2, e0, e2, score=0.8)

    restored = merge_restored_paths([p0, p1, p2], [], [c1, c2])

    assert len(restored) == 2
    assert not any(all(seg.source_type == "bridge" for seg in path.segments) for path in restored)


def test_close_single_gaps_allows_bridge_far_from_gap():
    points = _open_loop_points(center=(120.0, 120.0), rx=45.0, ry=30.0, gap_angle=0.36, n=12)
    path = _path_from_points(points)

    # Mark a middle segment as bridge, far from the actual open gap at start/end.
    path.segments[len(path.segments) // 2].source_type = "bridge"

    closed = close_single_gaps([path], efd_contours=[], gap_threshold=0.30, skip_if_bridge_present=True)

    assert closed[0].is_closed


def test_close_single_gaps_skips_when_bridge_overlaps_gap():
    points = _open_loop_points(center=(120.0, 120.0), rx=45.0, ry=30.0, gap_angle=0.40, n=12)
    path = _path_from_points(points)

    gap_start = path.segments[-1].control_points[3]
    gap_end = path.segments[0].control_points[0]
    bridge = _line_segment(gap_start, gap_end, source_type="bridge")

    # Insert overlapping bridge without altering the path's start/end identity.
    path.segments.insert(2, bridge)

    closed = close_single_gaps([path], efd_contours=[], gap_threshold=0.30, skip_if_bridge_present=True)

    assert not closed[0].is_closed


def test_scale_aware_closure_tolerance_in_synthesis():
    long_points = [
        (0.0, 0.0),
        (300.0, 0.0),
        (300.0, 300.0),
        (0.0, 300.0),
        (0.0, 8.0),  # small endpoint gap relative to path length (~1200)
    ]

    short_points = [(0.0, 0.0), (30.0, 0.0), (60.0, 5.0), (92.0, 0.0)]

    long_path = _path_from_points(long_points)
    short_path = _path_from_points(short_points)

    restored = merge_restored_paths([long_path, short_path], [], [])

    # Long path should close under scale-aware tolerance; short should remain open.
    assert restored[0].is_closed
    assert not restored[1].is_closed
