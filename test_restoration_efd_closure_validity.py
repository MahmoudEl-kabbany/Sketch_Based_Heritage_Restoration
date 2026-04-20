import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.efd_closure import _evaluate_closure_validity, close_single_gaps


def _line_segment(p0, p3, source_type="original"):
    p0 = np.asarray(p0, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    d = p3 - p0
    cp = np.vstack([
        p0,
        p0 + d / 3.0,
        p0 + 2.0 * d / 3.0,
        p3,
    ])
    return BezierSegment(control_points=cp, source_type=source_type)


def _open_three_side_rectangle():
    # Open contour with one recoverable left-side gap.
    seg1 = _line_segment([0.0, 0.0], [100.0, 0.0])
    seg2 = _line_segment([100.0, 0.0], [100.0, 80.0])
    seg3 = _line_segment([100.0, 80.0], [0.0, 80.0])
    return BezierPath(segments=[seg1, seg2, seg3], is_closed=False, source_type="extracted")


def test_evaluate_validity_rejects_long_gap_semantic_risk():
    metrics = {
        "bilateral_alignment": 0.18,
        "misalignment_deg": 132.0,
        "plausibility_score": 0.41,
        "long_gap_semantic_risk": 1.0,
    }

    accepted, reason = _evaluate_closure_validity(
        metrics,
        gap_dist=180.0,
        gap_ratio=0.29,
        has_symmetry=False,
        plausibility_threshold=0.5,
        min_gap_for_check=3.0,
    )

    assert accepted is False
    assert reason == "long_gap_semantic_risk"


def test_evaluate_validity_accepts_symmetry_supported_borderline():
    metrics = {
        "bilateral_alignment": 0.22,
        "misalignment_deg": 88.0,
        "plausibility_score": 0.39,
        "long_gap_semantic_risk": 0.0,
    }

    accepted, reason = _evaluate_closure_validity(
        metrics,
        gap_dist=42.0,
        gap_ratio=0.19,
        has_symmetry=True,
        plausibility_threshold=0.5,
        min_gap_for_check=3.0,
    )

    assert accepted is True
    assert reason == "symmetry_supported"


def test_close_single_gaps_returns_accepted_diagnostics_when_validity_disabled():
    path = _open_three_side_rectangle()

    closed_paths, diagnostics = close_single_gaps(
        [path],
        efd_contours=[],
        gap_threshold=0.40,
        validity_check_enabled=False,
        return_diagnostics=True,
    )

    assert len(closed_paths) == 1
    assert closed_paths[0].is_closed is True
    assert len(diagnostics) == 1
    assert diagnostics[0]["accepted"] is True
    assert diagnostics[0]["closure_method"] in {"symmetry_mirroring", "curvature_arc"}
    assert "metrics" in diagnostics[0]


def test_close_single_gaps_rejects_when_plausibility_threshold_is_strict():
    path = _open_three_side_rectangle()

    final_paths, diagnostics = close_single_gaps(
        [path],
        efd_contours=[],
        gap_threshold=0.40,
        symmetry_min_gap_ratio=0.95,
        symmetry_max_gap_ratio=1.00,
        validity_check_enabled=True,
        plausibility_threshold=0.95,
        return_diagnostics=True,
    )

    assert len(final_paths) == 1
    assert final_paths[0].is_closed is False
    assert len(diagnostics) == 1
    assert diagnostics[0]["accepted"] is False
    assert diagnostics[0]["reason"] in {
        "low_plausibility_score",
        "weak_continuation_support",
        "long_gap_semantic_risk",
    }


def test_evaluate_validity_allows_borderline_continuation_pass():
    metrics = {
        "bilateral_alignment": 0.19,
        "continuation_score": 0.40,
        "misalignment_deg": 127.5,
        "plausibility_score": 0.445,
        "long_gap_semantic_risk": 0.0,
    }

    accepted, reason = _evaluate_closure_validity(
        metrics,
        gap_dist=185.0,
        gap_ratio=0.13,
        has_symmetry=False,
        plausibility_threshold=0.5,
        min_gap_for_check=3.0,
    )

    assert accepted is True
    assert reason == "borderline_continuation_pass"


def test_evaluate_validity_allows_tiny_smooth_gap_grace():
    metrics = {
        "bilateral_alignment": -0.12,
        "continuation_score": 0.12,
        "misalignment_deg": 21.0,
        "plausibility_score": 0.252,
        "long_gap_semantic_risk": 0.0,
    }

    accepted, reason = _evaluate_closure_validity(
        metrics,
        gap_dist=36.5,
        gap_ratio=0.0267,
        has_symmetry=False,
        plausibility_threshold=0.5,
        min_gap_for_check=3.0,
    )

    assert accepted is True
    assert reason == "plausibility_pass"
