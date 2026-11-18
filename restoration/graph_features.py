"""A drawing as a graph, so that a candidate bridge can be judged in context (WP-7).

The scorecards judge a candidate from its two stroke ends alone. At a junction,
or where strokes run in parallel or mirror each other, the right pairing
depends on what lies around them. This module turns one extraction plus its
candidates into a graph of plain NumPy arrays (picklable, no torch needed),
which restoration/gnn.py passes messages over.

Nodes (one row of NODE_FEATURES each):
    endpoint   every free stroke end (node i = endpoint_id i)
    junction   path ends that sit on a junction, merged within one line width
    sample     points along every stroke, one every SAMPLE_SPACING_WIDTHS line widths

Relations (directed edges, both directions present):
    stroke     consecutive nodes along one stroke
    near       each node to its NEAR_K nearest nodes within NEAR_RADIUS_WIDTHS
    parallel   near pairs on different strokes whose directions agree (|cos| > 0.95)
    candidate  the two ends of every candidate bridge (competing candidates see each other)
    symmetric  each node to its mirror image's nearest node, across the drawing's
               reflection-symmetry axis (Phase 6's detector), when it has one

All sizes are in line widths and positions are divided by the image
diagonal, so the graph does not depend on resolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.spatial import cKDTree

RELATIONS = ("stroke", "near", "parallel", "candidate", "symmetric")
NODE_FEATURES = (
    "x", "y",                      # position / diagonal
    "dir_cos", "dir_sin",          # outward tangent (stroke ends only; 0 elsewhere)
    "axis_cos2", "axis_sin2",      # stroke orientation as a doubled angle (direction-free)
    "curvature",                   # signed curvature x line width (stroke ends), clipped to [-2, 2]
    "radius",                      # stroke radius / line width
    "is_endpoint", "is_junction", "is_sample",
)
SAMPLE_SPACING_WIDTHS = 4.0
NEAR_RADIUS_WIDTHS = 8.0
NEAR_K = 6
PARALLEL_COS = 0.95
SYMMETRY_MATCH_WIDTHS = 2.0


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(steps)))
    if arc[-1] < 1e-9:
        return points[:1]
    count = max(2, int(np.ceil(arc[-1] / spacing)) + 1)
    grid = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(grid, arc, points[:, 0]), np.interp(grid, arc, points[:, 1])])


def build_graph(extraction: Any, candidates: Sequence[Any]) -> Dict[str, Any]:
    """The graph of one drawing and its candidates (see the module docstring).

    Returns a dict: "x" (nodes x features, float32), "edges" {relation: (2, E) int64},
    "candidate_nodes" (candidates x 2, the node of each candidate's two ends, in
    the order of *candidates*).
    """
    lw = max(float(extraction.line_width), 1.0)
    diag = max(float(extraction.diagonal), 1.0)
    rows: List[List[float]] = []
    path_of: List[int] = []

    def add(xy, direction, axis, curvature, radius, kind, path) -> int:
        axis = _unit(np.asarray(axis, float))
        angle2 = (2 * axis[0] ** 2 - 1, 2 * axis[0] * axis[1]) if np.any(axis) else (0.0, 0.0)
        rows.append([xy[0] / diag, xy[1] / diag, direction[0], direction[1], angle2[0], angle2[1],
                     float(np.clip(curvature * lw, -2.0, 2.0)), radius / lw,
                     kind == "endpoint", kind == "junction", kind == "sample"])
        path_of.append(path)
        return len(rows) - 1

    # Stroke ends first, so that node i is endpoint i.
    end_node: Dict[tuple, int] = {}
    for ep in extraction.endpoints:
        end_node[(ep.path_index, ep.end)] = add(ep.position, ep.tangent, ep.tangent, ep.signed_curvature,
                                                ep.radius, "endpoint", ep.path_index)

    # Junction ends, merged within a line width.
    junction_positions: List[np.ndarray] = []
    junction_ids: List[int] = []
    stroke_edges: List[tuple] = []
    for p, path in enumerate(extraction.paths):
        if not path.segments:
            continue
        samples = _resample(path.sample(pts_per_segment=20), SAMPLE_SPACING_WIDTHS * lw)
        chain: List[int] = []
        for end, free, xy in (("start", path.start_free, samples[0]), ("end", path.end_free, samples[-1])):
            if (p, end) in end_node or path.is_closed:
                continue
            match = next((j for j, q in zip(junction_ids, junction_positions) if np.linalg.norm(q - xy) <= lw), None)
            if match is None:
                match = add(xy, (0.0, 0.0), samples[1] - samples[0] if len(samples) > 1 else (0, 0), 0.0,
                            lw / 2.0, "junction", -1)
                junction_ids.append(match)
                junction_positions.append(np.asarray(xy, float))
            end_node[(p, end)] = match
        inner = samples[1:-1] if len(samples) > 2 else samples[:0]
        start = end_node.get((p, "start"))
        if start is not None:
            chain.append(start)
        for k, xy in enumerate(inner):
            before = samples[k]
            after = samples[k + 2]
            chain.append(add(xy, (0.0, 0.0), after - before, 0.0, lw / 2.0, "sample", p))
        end = end_node.get((p, "end"))
        if end is not None and end != start:
            chain.append(end)
        if path.is_closed and len(chain) > 2:
            chain.append(chain[0])
        stroke_edges += list(zip(chain[:-1], chain[1:]))

    x = np.asarray(rows, dtype=np.float32).reshape(-1, len(NODE_FEATURES))
    positions = x[:, :2].astype(np.float64) * diag
    edges: Dict[str, List[tuple]] = {r: [] for r in RELATIONS}
    edges["stroke"] = stroke_edges

    # Spatial neighbours; parallel strokes among them.
    if len(x) > 1:
        tree = cKDTree(positions)
        distance, index = tree.query(positions, k=min(NEAR_K + 1, len(x)), distance_upper_bound=NEAR_RADIUS_WIDTHS * lw)
        stroke_pairs = {frozenset(e) for e in stroke_edges}
        for i in range(len(x)):
            for d, j in zip(distance[i], index[i]):
                if j == i or not np.isfinite(d) or frozenset((i, int(j))) in stroke_pairs:
                    continue
                edges["near"].append((i, int(j)))
                a, b = x[i, 4:6], x[int(j), 4:6]
                # cos of the doubled-angle difference > 2*0.95^2-1 <=> |cos of the angle| > 0.95
                if path_of[i] != path_of[int(j)] and np.any(a) and np.any(b) and float(a @ b) > 2 * PARALLEL_COS ** 2 - 1:
                    edges["parallel"].append((i, int(j)))

    # Candidate bridges between their two ends.
    candidate_nodes = np.array([[c.ep_a.endpoint_id, c.ep_b.endpoint_id] for c in candidates], dtype=np.int64).reshape(-1, 2)
    edges["candidate"] = sorted({tuple(pair) for pair in candidate_nodes.tolist()})

    # Mirror partners across the drawing's reflection axis, if it has one.
    if len(x) > 4:
        from restoration.single_gap_closure import _detect_symmetry

        centroid = positions.mean(axis=0)
        axis = _detect_symmetry(positions, centroid, hausdorff_threshold=max(2.0 * lw, 0.02 * diag))
        if axis is not None:
            c2, s2 = np.cos(2 * axis), np.sin(2 * axis)
            mirrored = (positions - centroid) @ np.array([[c2, s2], [s2, -c2]]).T + centroid
            d, j = cKDTree(positions).query(mirrored)
            edges["symmetric"] = [(i, int(k)) for i, (dd, k) in enumerate(zip(d, j))
                                  if dd <= SYMMETRY_MATCH_WIDTHS * lw and int(k) != i]

    def both_ways(pairs):
        if not pairs:
            return np.zeros((2, 0), dtype=np.int64)
        arr = np.asarray(sorted(set(pairs) | {(b, a) for a, b in pairs}), dtype=np.int64)
        return arr.T.copy()

    return {
        "x": x,
        "edges": {r: both_ways(edges[r]) for r in RELATIONS},
        "candidate_nodes": candidate_nodes,
    }
