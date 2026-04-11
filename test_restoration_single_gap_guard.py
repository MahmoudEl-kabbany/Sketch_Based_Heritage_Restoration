import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.asp.asp_inference import RankedHypothesis
from restoration.feature_bridge import FeatureBundle, GapRecord
from restoration.restoration import (
    RestorationConfig,
    _inject_single_gap_guard_hypotheses,
    execute_restoration,
)


def _make_open_line_path(x0: float, x1: float, y: float = 0.0) -> BezierPath:
    dx = (x1 - x0) / 3.0
    cp = np.array(
        [
            [x0, y],
            [x0 + dx, y],
            [x0 + 2.0 * dx, y],
            [x1, y],
        ],
        dtype=np.float64,
    )
    return BezierPath(segments=[BezierSegment(cp, source_type="test")], is_closed=False, source_type="test")


def _make_open_u_path() -> BezierPath:
    left = np.array(
        [[0.0, 0.0], [0.0, 33.0], [0.0, 67.0], [0.0, 100.0]],
        dtype=np.float64,
    )
    bottom = np.array(
        [[0.0, 100.0], [33.0, 100.0], [67.0, 100.0], [100.0, 100.0]],
        dtype=np.float64,
    )
    right = np.array(
        [[100.0, 100.0], [100.0, 67.0], [100.0, 33.0], [100.0, 0.0]],
        dtype=np.float64,
    )
    return BezierPath(
        segments=[
            BezierSegment(left, source_type="test"),
            BezierSegment(bottom, source_type="test"),
            BezierSegment(right, source_type="test"),
        ],
        is_closed=False,
        source_type="test",
    )


def _single_gap(
    endpoint_a_id: int,
    endpoint_b_id: int,
    path_idx: int,
    confidence: float = 0.2,
    symmetry_score: float = 0.0,
) -> GapRecord:
    return GapRecord(
        gap_id=0,
        endpoint_a_id=endpoint_a_id,
        endpoint_b_id=endpoint_b_id,
        path_a=path_idx,
        path_b=path_idx,
        distance=20.0,
        angle_a_deg=5.0,
        angle_b_deg=5.0,
        curvature_delta_deg=0.0,
        continuation_score=0.8,
        proximity_score=0.4,
        symmetry_score=symmetry_score,
        geometric_score=0.2,
        confidence=confidence,
        gap_kind="single_contour_gap",
        suggested_method="bezier",
        is_forced=False,
        reasons=["test"],
    )


def _hyp(
    endpoint_a_id: int,
    endpoint_b_id: int,
    path_a: int,
    path_b: int,
    confidence: float = 0.9,
    metadata=None,
) -> RankedHypothesis:
    return RankedHypothesis(
        rank=1,
        hypothesis_id="h1",
        gap_id=0,
        endpoint_a_id=endpoint_a_id,
        endpoint_b_id=endpoint_b_id,
        path_a=path_a,
        path_b=path_b,
        action="connect_endpoints",
        method="bezier",
        confidence=confidence,
        score=1.0 - confidence,
        is_forced=False,
        metadata=metadata or {},
    )


def test_guard_injects_single_gap_when_asp_misses() -> None:
    paths = [_make_open_line_path(0.0, 100.0)]
    features = FeatureBundle(gaps=[_single_gap(0, 1, 0, confidence=0.08)], endpoints=[])

    result = execute_restoration(
        hypotheses=[],
        paths=paths,
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=features,
    )

    assert len(result.additions) == 1
    addition = result.additions[0]
    assert addition.is_forced
    assert addition.path_a == 0 and addition.path_b == 0
    assert {addition.endpoint_a_id, addition.endpoint_b_id} == {0, 1}


def test_guard_disable_flag_keeps_baseline_behavior() -> None:
    paths = [_make_open_line_path(0.0, 100.0)]
    features = FeatureBundle(gaps=[_single_gap(0, 1, 0, confidence=0.08)], endpoints=[])

    result = execute_restoration(
        hypotheses=[],
        paths=paths,
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=features,
        disable_single_gap_guard=True,
    )

    assert len(result.additions) == 0


def test_guard_respects_endpoint_occupancy_precedence() -> None:
    paths = [
        _make_open_line_path(0.0, 100.0),
        _make_open_line_path(140.0, 240.0),
    ]
    features = FeatureBundle(gaps=[_single_gap(0, 1, 0, confidence=0.08)], endpoints=[])
    hypotheses = [_hyp(endpoint_a_id=0, endpoint_b_id=2, path_a=0, path_b=1)]

    result = execute_restoration(
        hypotheses=hypotheses,
        paths=paths,
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=features,
    )

    endpoint_pairs = {
        tuple(sorted((add.endpoint_a_id, add.endpoint_b_id))) for add in result.additions
    }
    assert (0, 2) in endpoint_pairs
    assert (0, 1) not in endpoint_pairs


def test_guard_does_not_duplicate_existing_single_gap_hypothesis() -> None:
    paths = [_make_open_line_path(0.0, 100.0)]
    features = FeatureBundle(gaps=[_single_gap(0, 1, 0, confidence=0.08)], endpoints=[])
    hypotheses = [
        _hyp(
            endpoint_a_id=0,
            endpoint_b_id=1,
            path_a=0,
            path_b=0,
            metadata={"is_single_gap": 1.0},
        )
    ]

    result = execute_restoration(
        hypotheses=hypotheses,
        paths=paths,
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=features,
    )

    assert len(result.additions) == 1
    assert not result.additions[0].is_forced


def test_guard_method_priority_symmetry_then_efd_then_bezier() -> None:
    cfg = RestorationConfig()

    sym_gap = _single_gap(0, 1, 0, confidence=0.9, symmetry_score=cfg.symmetry_method_min_confidence + 0.1)
    sym_injected = _inject_single_gap_guard_hypotheses(
        hypotheses=[],
        features=FeatureBundle(gaps=[sym_gap], endpoints=[]),
        efd_data={0: np.ones(8, dtype=np.float64)},
        cfg=cfg,
    )
    assert sym_injected
    assert sym_injected[0].method == "symmetry"

    efd_gap = _single_gap(0, 1, 0, confidence=0.9, symmetry_score=0.0)
    efd_injected = _inject_single_gap_guard_hypotheses(
        hypotheses=[],
        features=FeatureBundle(gaps=[efd_gap], endpoints=[]),
        efd_data={0: np.ones(8, dtype=np.float64)},
        cfg=cfg,
    )
    assert efd_injected
    assert efd_injected[0].method == "efd"

    bezier_injected = _inject_single_gap_guard_hypotheses(
        hypotheses=[],
        features=FeatureBundle(gaps=[efd_gap], endpoints=[]),
        efd_data={},
        cfg=cfg,
    )
    assert bezier_injected
    assert bezier_injected[0].method == "bezier"


def test_open_path_fallback_closes_near_closed_contour_without_feature_gap() -> None:
    result = execute_restoration(
        hypotheses=[],
        paths=[_make_open_u_path()],
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=None,
    )

    assert len(result.additions) == 1
    assert result.additions[0].is_forced
    assert {result.additions[0].endpoint_a_id, result.additions[0].endpoint_b_id} == {0, 1}


def test_open_path_fallback_skips_line_like_open_stroke() -> None:
    result = execute_restoration(
        hypotheses=[],
        paths=[_make_open_line_path(0.0, 120.0)],
        efd_data={},
        vocab=None,
        config=RestorationConfig(),
        features=None,
    )

    assert len(result.additions) == 0
