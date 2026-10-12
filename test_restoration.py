"""Checks for geometric and solver properties the restoration pipeline relies on.

Synthetic inputs only (no dataset needed). Run with:  python -m pytest
"""
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment, fit_straight_cubic
from restoration.asp_engine import solve_candidates
from restoration.candidates import (
    ConnectionCandidate,
    _build_continuation_bridge,
    _build_intersection_bridge,
    _sample_bridge,
)
from restoration.extraction import EndpointInfo
from restoration.geometry import safe_normalize
from restoration.scoring import compute_mjerk, context_curve


def polyline_path(points) -> BezierPath:
    """Open path made of straight cubic segments through *points*."""
    points = np.asarray(points, dtype=np.float64)
    segments = [BezierSegment(fit_straight_cubic(p, q), "skeleton") for p, q in zip(points[:-1], points[1:])]
    return BezierPath(segments=segments, source_type="skeleton")


def endpoint(endpoint_id: int, path_index: int, path: BezierPath, end: str) -> EndpointInfo:
    """Endpoint descriptor with the outward tangent of a straight-segment path."""
    cp = path.segments[0].control_points if end == "start" else path.segments[-1].control_points
    position, inner = (cp[0], cp[1]) if end == "start" else (cp[3], cp[2])
    return EndpointInfo(endpoint_id, path_index, end, position.copy(), safe_normalize(position - inner), 0.0)


def candidate(cid: int, ep_a: EndpointInfo, ep_b: EndpointInfo, score: float = 0.5) -> ConnectionCandidate:
    segments = _build_continuation_bridge(ep_a, ep_b)
    return ConnectionCandidate(
        id=cid, ep_a=ep_a, ep_b=ep_b, scenario="continuation",
        bridge_points=_sample_bridge(segments), bridge_bezier=segments,
        distance=float(np.linalg.norm(ep_b.position - ep_a.position)), score=score,
    )


def test_bridges_arrive_against_the_outward_tangent():
    left = polyline_path([[0, 0], [100, 0]])
    right = polyline_path([[160, 20], [260, 20]])
    ep_a, ep_b = endpoint(0, 0, left, "end"), endpoint(1, 1, right, "start")

    bridges = (
        _build_continuation_bridge(ep_a, ep_b),
        _build_intersection_bridge(ep_a, ep_b, np.array([130.0, 10.0])),
    )
    for segments in bridges:
        first, last = segments[0].control_points, segments[-1].control_points
        assert np.dot(first[1] - first[0], ep_a.tangent) > 0    # leaves A outward, into the gap
        assert np.dot(last[3] - last[2], ep_b.tangent) < 0      # enters B from the gap


def test_context_curve_is_continuous_for_every_end_combination():
    left = polyline_path([[0, 0], [100, 0]])
    right = polyline_path([[150, 0], [250, 0]])
    dense = [left.sample(pts_per_segment=50), right.sample(pts_per_segment=50)]
    tail_len = 20

    cid = 0
    for end_a in ("start", "end"):
        for end_b in ("start", "end"):
            ep_a, ep_b = endpoint(0, 0, left, end_a), endpoint(1, 1, right, end_b)
            c = candidate(cid, ep_a, ep_b)
            curve = context_curve(c, dense, tail_len)

            # The context runs into A, the bridge follows, then the context leaves B.
            assert np.allclose(curve[tail_len - 1], ep_a.position)
            assert np.allclose(curve[tail_len + len(c.bridge_points)], ep_b.position)
            # No jump to the far end of a path: within each piece, steps stay small.
            steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            assert steps[: tail_len - 1].max() < 5.0
            assert steps[-(tail_len - 1):].max() < 5.0
            cid += 1


def test_mjerk_measures_shape_not_sampling():
    t = np.linspace(0.0, 1.0, 40)[:, np.newaxis]
    line = np.hstack([100.0 * t, 50.0 * t])
    assert compute_mjerk(np.repeat(line, 2, axis=0)) < 1e-12   # repeated points do not matter

    # A circle of radius R has |r'''(s)| = 1 / R^2 everywhere.
    for num_samples in (20, 50, 200):
        angle = np.linspace(0.0, np.pi, num_samples)[:, np.newaxis]
        arc = 100.0 * np.hstack([np.cos(angle), np.sin(angle)])
        assert abs(compute_mjerk(arc) - 1e-8) < 1e-9


def test_solver_is_repeatable_and_drops_bridges_without_evidence():
    # Three collinear strokes with two gaps, and two unrelated strokes.
    paths = [
        polyline_path([[0, 0], [100, 0]]),
        polyline_path([[130, 0], [230, 0]]),
        polyline_path([[260, 0], [360, 0]]),
        polyline_path([[0, 300], [100, 300]]),
        polyline_path([[130, 300], [230, 300]]),
    ]
    ep = {
        (i, end): endpoint(2 * i + (end == "end"), i, path, end)
        for i, path in enumerate(paths) for end in ("start", "end")
    }
    candidates = [
        candidate(0, ep[0, "end"], ep[1, "start"], score=0.8),
        candidate(1, ep[1, "end"], ep[2, "start"], score=0.7),
        candidate(2, ep[0, "end"], ep[2, "start"], score=0.5),   # competes with both of the above
        candidate(3, ep[3, "end"], ep[4, "start"], score=0.0),   # no utility, completes no path
    ]

    first, _ = solve_candidates(candidates)
    second, _ = solve_candidates(candidates)
    assert first == second == [0, 1]
