"""Small geometry helpers shared by the restoration phases."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from bezier_curves.bezier import BezierPath


def safe_normalize(v: np.ndarray) -> np.ndarray:
    """Unit vector along *v*; falls back to +x for a degenerate vector."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def polyline_length(points: np.ndarray) -> float:
    """Arc length of a polyline."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def bezier_derivative(cp: np.ndarray, t: float) -> np.ndarray:
    """First derivative of a cubic Bezier with control points *cp* (4, 2) at *t*."""
    u = 1.0 - t
    return (
        3.0 * u * u * (cp[1] - cp[0])
        + 6.0 * u * t * (cp[2] - cp[1])
        + 3.0 * t * t * (cp[3] - cp[2])
    )


def bezier_curvature(cp: np.ndarray, t: float) -> float:
    """Unsigned curvature κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)."""
    u = 1.0 - t
    d1 = bezier_derivative(cp, t)
    d2 = (
        6.0 * u * (cp[2] - 2.0 * cp[1] + cp[0])
        + 6.0 * t * (cp[3] - 2.0 * cp[2] + cp[1])
    )
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    speed_sq = float(d1[0] ** 2 + d1[1] ** 2)
    if speed_sq < 1e-24:
        return 0.0
    return abs(cross) / (speed_sq ** 1.5)


def endpoint_curvature(path: BezierPath, end: str) -> float:
    """Unsigned curvature at the "start" or "end" of *path*."""
    if end == "start":
        return bezier_curvature(path.segments[0].control_points, 0.0)
    return bezier_curvature(path.segments[-1].control_points, 1.0)


def visual_order_key(path: BezierPath) -> Tuple[int, float]:
    """Sort key for top-down, then left-right ordering (12 px row bands)."""
    pts = path.sample(5)
    return (round(np.min(pts[:, 1]) / 12.0), np.min(pts[:, 0]))
