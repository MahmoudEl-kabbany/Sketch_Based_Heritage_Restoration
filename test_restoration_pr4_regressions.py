import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.asp_engine import encode_facts
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


def test_cross_shape_suppressed_when_strong_self_closure_exists():
    loop = _path_from_points(_open_loop_points(center=(90.0, 90.0), rx=34.0, ry=24.0, gap_angle=0.34, n=10))
    bolt = _path_from_points([(132.0, 88.0), (170.0, 88.0)])

    e_loop_start = _endpoint_from_path(loop, 0, "start", 0)
    e_loop_end = _endpoint_from_path(loop, 0, "end", 1)
    e_bolt_start = _endpoint_from_path(bolt, 1, "start", 2)
    e_bolt_end = _endpoint_from_path(bolt, 1, "end", 3)

    result = ExtractionResult(
        paths=[loop, bolt],
        endpoints=[e_loop_start, e_loop_end, e_bolt_start, e_bolt_end],
        efd_contours=[],
        image_shape=(260, 260),
        diagonal=float(np.hypot(260.0, 260.0)),
    )

    self_candidate = _candidate(1, e_loop_end, e_loop_start, score=0.55, scenario="self_closure")
    self_candidate.same_path_closure = True
    cross_candidate = _candidate(2, e_loop_end, e_bolt_start, score=0.55)

    scored_with_self = score_candidates([self_candidate, cross_candidate], result)
    cross_score_with_self = next(c.score for c in scored_with_self if c.id == 2)

    self_candidate_no_flag = _candidate(1, e_loop_end, e_loop_start, score=0.55, scenario="self_closure")
    self_candidate_no_flag.same_path_closure = False
    cross_candidate_no_flag = _candidate(2, e_loop_end, e_bolt_start, score=0.55)
    scored_without_self = score_candidates([self_candidate_no_flag, cross_candidate_no_flag], result)
    cross_score_without_self = next(c.score for c in scored_without_self if c.id == 2)

    assert cross_score_with_self < cross_score_without_self - 0.08


def test_encode_facts_emits_self_closure_support_and_cross_shape_facts():
    p0 = _path_from_points(_open_loop_points(center=(80.0, 80.0), rx=32.0, ry=22.0, gap_angle=0.30, n=10))
    p1 = _path_from_points([(120.0, 80.0), (160.0, 80.0)])

    e0s = _endpoint_from_path(p0, 0, "start", 0)
    e0e = _endpoint_from_path(p0, 0, "end", 1)
    e1s = _endpoint_from_path(p1, 1, "start", 2)
    e1e = _endpoint_from_path(p1, 1, "end", 3)

    c_self = _candidate(10, e0e, e0s, score=0.80, scenario="self_closure")
    c_self.same_path_closure = True
    c_self.score = 0.80

    c_cross = _candidate(11, e0e, e1s, score=0.52)
    c_cross.score = 0.52

    facts = encode_facts([c_self, c_cross], [e0s, e0e, e1s, e1e])

    assert "path_self_closure_strength(0,2)." in facts
    assert "candidate_cross_shape(11)." in facts
    assert "candidate_touches_path(11,0)." in facts
    assert "candidate_touches_path(11,1)." in facts


def test_same_shape_affinity_prefers_overlapping_fragments():
    # A and B overlap strongly (likely one shape split into two paths).
    path_a = _path_from_points([(120.0, 760.0), (180.0, 860.0), (170.0, 980.0)])
    path_b = _path_from_points([(155.0, 780.0), (250.0, 860.0), (260.0, 970.0)])
    # C is a distant shape with similar endpoint distance to A.
    path_c = _path_from_points([(120.0, 420.0), (220.0, 500.0), (300.0, 560.0)])

    e_a_start = _endpoint_from_path(path_a, 0, "start", 0)
    e_a_end = _endpoint_from_path(path_a, 0, "end", 1)
    e_b_start = _endpoint_from_path(path_b, 1, "start", 2)
    e_c_start = _endpoint_from_path(path_c, 2, "start", 3)

    # Use the same endpoint source and similar candidate setup.
    c_overlap = _candidate(21, e_a_end, e_b_start, score=0.50)
    c_far = _candidate(22, e_a_end, e_c_start, score=0.50)

    result = ExtractionResult(
        paths=[path_a, path_b, path_c],
        endpoints=[e_a_start, e_a_end, e_b_start, e_c_start],
        efd_contours=[],
        image_shape=(1400, 2800),
        diagonal=float(np.hypot(1400.0, 2800.0)),
    )

    scored = score_candidates([c_overlap, c_far], result)
    score_overlap = next(c.score for c in scored if c.id == 21)
    score_far = next(c.score for c in scored if c.id == 22)

    assert score_overlap > score_far


def test_low_affinity_competitor_suppressed_by_high_affinity_support():
    # path_target is near/overlapping with path_good, but not with path_bad.
    path_target = _path_from_points([(1400.0, 800.0), (1480.0, 900.0), (1460.0, 1040.0)])
    path_good = _path_from_points([(1450.0, 820.0), (1560.0, 920.0), (1600.0, 1080.0)])
    path_bad = _path_from_points([(1300.0, 360.0), (1410.0, 500.0), (1490.0, 640.0)])

    e_t_end = _endpoint_from_path(path_target, 0, "end", 0)
    e_g_start = _endpoint_from_path(path_good, 1, "start", 1)
    e_b_start = _endpoint_from_path(path_bad, 2, "start", 2)

    c_good = _candidate(31, e_t_end, e_g_start, score=0.50)
    c_bad = _candidate(32, e_t_end, e_b_start, score=0.50)

    result = ExtractionResult(
        paths=[path_target, path_good, path_bad],
        endpoints=[e_t_end, e_g_start, e_b_start],
        efd_contours=[],
        image_shape=(1600, 2800),
        diagonal=float(np.hypot(1600.0, 2800.0)),
    )

    scored = score_candidates([c_good, c_bad], result)
    score_good = next(c.score for c in scored if c.id == 31)
    score_bad = next(c.score for c in scored if c.id == 32)

    assert score_good > score_bad + 0.05
