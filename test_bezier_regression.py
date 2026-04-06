"""Regression checks for skeleton chain continuity and Bezier straightness fidelity.

Run from repo root:
    python test_bezier_regression.py
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from bezier_curves.bezier import (
    _build_skeleton_chains,
    _estimate_tangent,
    _fit_cubic_single,
)


def _xy_to_row_col(points_xy: list[tuple[float, float]]) -> np.ndarray:
    """Convert (x, y) polyline to sknw-style (row, col) = (y, x)."""
    arr = np.asarray(points_xy, dtype=np.float64)
    return arr[:, ::-1]


def _add_node_xy(graph: nx.MultiGraph, node_id: int, x: float, y: float) -> None:
    # sknw stores node center in row/col ordering.
    graph.add_node(node_id, o=np.array([y, x], dtype=np.float64))


def _polyline(start: tuple[float, float], end: tuple[float, float], n: int = 11) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def test_l_corner_continuity() -> None:
    g = nx.MultiGraph()
    _add_node_xy(g, 0, 0.0, 0.0)
    _add_node_xy(g, 1, 10.0, 0.0)
    _add_node_xy(g, 2, 10.0, 10.0)

    g.add_edge(0, 1, pts=_xy_to_row_col(_polyline((0.0, 0.0), (10.0, 0.0))))
    g.add_edge(1, 2, pts=_xy_to_row_col(_polyline((10.0, 0.0), (10.0, 10.0))))

    chains = _build_skeleton_chains(
        g,
        follow_junction_continuation=True,
        junction_min_alignment=-0.30,
        junction_min_score_margin=0.04,
    )

    assert len(chains) == 1, f"Expected 1 chain for an uninterrupted L-corner, got {len(chains)}"
    chain = chains[0]
    assert not chain.is_closed, "L-corner should remain an open chain"
    assert len(chain.points) >= 3, "Chain should preserve polyline samples"


def test_t_junction_branch_policy() -> None:
    g = nx.MultiGraph()
    _add_node_xy(g, 0, 0.0, 0.0)
    _add_node_xy(g, 1, 10.0, 0.0)
    _add_node_xy(g, 2, 24.0, 0.0)  # Longer continuation arm
    _add_node_xy(g, 3, 10.0, 8.0)  # Side branch

    g.add_edge(0, 1, pts=_xy_to_row_col(_polyline((0.0, 0.0), (10.0, 0.0))))
    g.add_edge(1, 2, pts=_xy_to_row_col(_polyline((10.0, 0.0), (24.0, 0.0))))
    g.add_edge(1, 3, pts=_xy_to_row_col(_polyline((10.0, 0.0), (10.0, 8.0))))

    chains = _build_skeleton_chains(
        g,
        follow_junction_continuation=True,
        junction_min_alignment=-0.15,
        junction_min_score_margin=0.03,
    )

    assert len(chains) == 2, f"Expected 2 chains (main continuation + side branch), got {len(chains)}"

    lengths = [float(np.sum(np.linalg.norm(np.diff(c.points, axis=0), axis=1))) for c in chains]
    longest = chains[int(np.argmax(lengths))]
    x_span = float(np.max(longest.points[:, 0]) - np.min(longest.points[:, 0]))
    y_span = float(np.max(longest.points[:, 1]) - np.min(longest.points[:, 1]))

    assert x_span > y_span, "Main continuation chain should align with the horizontal stem"


def test_straight_line_fidelity() -> None:
    x = np.linspace(0.0, 180.0, 180)
    y = 42.0 + 0.25 * np.sin(x / 9.0)
    pts = np.column_stack([x, y]).astype(np.float64)

    left = _estimate_tangent(pts, "start", lookahead=8)
    right = _estimate_tangent(pts, "end", lookahead=8)
    curves = _fit_cubic_single(pts, left, right, max_error=25.0)

    assert len(curves) == 1, f"Expected one cubic for near-straight stroke, got {len(curves)}"

    cp = curves[0]
    p0, p3 = cp[0], cp[3]
    line = p3 - p0
    denom = float(np.dot(line, line))
    assert denom > 1e-9, "Degenerate straight-line test"

    samples_t = np.linspace(0.0, 1.0, 120)
    u = 1.0 - samples_t
    samples = (
        (u**3)[:, np.newaxis] * cp[0]
        + 3 * (u**2)[:, np.newaxis] * samples_t[:, np.newaxis] * cp[1]
        + 3 * u[:, np.newaxis] * (samples_t**2)[:, np.newaxis] * cp[2]
        + (samples_t**3)[:, np.newaxis] * cp[3]
    )

    rel = samples - p0
    proj_t = np.sum(rel * line, axis=1) / denom
    proj = p0 + proj_t[:, np.newaxis] * line
    distances = np.linalg.norm(samples - proj, axis=1)

    assert float(np.max(distances)) <= 0.65, (
        f"Straight fit bowed too much: max perpendicular deviation={float(np.max(distances)):.3f}"
    )


def run_all() -> None:
    tests = [
        ("L-corner continuity", test_l_corner_continuity),
        ("T-junction branch policy", test_t_junction_branch_policy),
        ("Straight-line fidelity", test_straight_line_fidelity),
    ]

    print("\n" + "=" * 64)
    print("Bezier regression checks")
    print("=" * 64)

    for name, fn in tests:
        fn()
        print(f"[PASS] {name}")

    print("=" * 64)
    print("All regression checks passed.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    run_all()
