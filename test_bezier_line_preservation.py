import inspect

import numpy as np

from bezier_curves.bezier import (
    ContourBezierFitter,
    _estimate_tangent,
    _fit_cubic_single,
    fit_from_contours,
    fit_from_image,
    fit_from_image_skeleton,
)


def _point_line_distance(point: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    direction = p1 - p0
    denom = float(np.dot(direction, direction))
    if denom < 1e-12:
        return float(np.linalg.norm(point - p0))
    t = float(np.dot(point - p0, direction) / denom)
    proj = p0 + t * direction
    return float(np.linalg.norm(point - proj))


def test_noisy_straight_polyline_prefers_straight_cubic() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 120.0, 60)
    y = 0.10 * np.sin(x / 11.0) + rng.normal(0.0, 0.06, size=x.shape)
    points = np.column_stack([x, y]).astype(np.float64)

    left_tangent = _estimate_tangent(points, "start", lookahead=5)
    right_tangent = _estimate_tangent(points, "end", lookahead=5)

    curves = _fit_cubic_single(
        points,
        left_tangent,
        right_tangent,
        max_error=2.0 ** 2,
        straightness_scale=0.75,
    )

    assert len(curves) == 1

    cp = curves[0]
    p0, p1, p2, p3 = cp
    assert _point_line_distance(p1, p0, p3) < 1e-10
    assert _point_line_distance(p2, p0, p3) < 1e-10


def test_public_defaults_for_line_preservation_are_exposed() -> None:
    fit_single_sig = inspect.signature(_fit_cubic_single)
    fitter_init_sig = inspect.signature(ContourBezierFitter.__init__)
    contours_sig = inspect.signature(fit_from_contours)
    image_sig = inspect.signature(fit_from_image)
    skeleton_sig = inspect.signature(fit_from_image_skeleton)

    assert fit_single_sig.parameters["straightness_scale"].default == 0.75
    assert fitter_init_sig.parameters["max_error"].default == 2.0
    assert fitter_init_sig.parameters["straightness_scale"].default == 0.75
    assert contours_sig.parameters["max_error"].default == 2.0
    assert contours_sig.parameters["straightness_scale"].default == 0.75
    assert image_sig.parameters["max_error"].default == 2.0
    assert image_sig.parameters["straightness_scale"].default == 0.75
    assert skeleton_sig.parameters["max_error"].default == 2.0
    assert skeleton_sig.parameters["straightness_scale"].default == 0.75
