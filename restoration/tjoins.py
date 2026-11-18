"""Phase 5b — T-joins: a stroke end extended until it meets another stroke.

Bridges (Phases 2-5) join two free stroke ends. Some damage cannot be repaired
that way: a stroke whose tip was erased next to another stroke it used to
touch, or the stem of an erased T-junction, whose bar is redrawn by a bridge
and which must then meet that bridge. A T-join extends such a free end to the
nearest point of a stroke ahead of it.

T-joins run after the bridges are merged, so they can land on a bridge just
drawn. Every free end left over is a source; its candidate target is the
nearest point of any final path (other than the stretch right behind the end)
within REACH_WIDTHS line widths and within AIM_CONE_DEG of the end's direction.
Each candidate is described by TJOIN_FEATURES and scored by the "tjoin"
scorecard of the model file (fitted by train_scoring.py like the others); an
end takes its candidate if the probability is above the model's threshold.
Without a "tjoin" scorecard in the model, this phase does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.extraction import EndpointInfo, ExtractionResult

REACH_WIDTHS = 12.0      # longest T-join, in line widths
AIM_CONE_DEG = 25.0      # the target must lie within this angle of the end's direction
OWN_STROKE_SKIP_WIDTHS = 2.0   # samples this close behind the end belong to its own stroke

TJOIN_FEATURES: Dict[str, str] = {
    "tjoin_length": "length of the T-join in stroke widths",
    "tjoin_aim": "degrees between the end's direction and the target",
    "tjoin_incidence": "degrees between the T-join and the target stroke (90 = square on)",
    "tjoin_crossed_ink": "existing stroke the T-join runs over before its target, in stroke widths",
    "tjoin_stub": "length of the stroke behind the end, in stroke widths",
    "tjoin_free_end_near": "gap length relative to the distance to the nearest other free end (above 1: a free end is nearer than the target)",
    "tjoin_on_bridge": "the target lies on a bridge drawn in this restoration (1 = yes)",
}
TJOIN_FEATURE_NAMES = tuple(TJOIN_FEATURES)
TJOIN_SCALES = {"tjoin_length": "log2(x)", "tjoin_stub": "log2(1 + x)", "tjoin_free_end_near": "log2(x)",
                "tjoin_crossed_ink": "log2(1 + x)"}
TJOIN_DIRECTIONS = {"tjoin_length": -1, "tjoin_aim": -1, "tjoin_crossed_ink": -1}


@dataclass
class TJoinCandidate:
    endpoint: EndpointInfo
    target: np.ndarray                  # (x, y) on the target path
    target_path: int                    # index into the final paths
    points: np.ndarray                  # sampled T-join, from the end to the target
    segment: BezierSegment
    measurements: Dict[str, float] = field(default_factory=dict)
    probability: float = 0.0
    contributions: Dict[str, float] = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(2)


def _segment(p0: np.ndarray, tangent: np.ndarray, q: np.ndarray) -> BezierSegment:
    """A cubic that leaves p0 along its tangent and arrives straight at q."""
    length = float(np.linalg.norm(q - p0))
    arrive = _unit(q - p0)
    cp = np.vstack([p0, p0 + tangent * length / 3.0, q - arrive * length / 3.0, q])
    return BezierSegment(control_points=cp, source_type="tjoin")


def free_sources(extraction: ExtractionResult, final_paths: Sequence[BezierPath]) -> List[EndpointInfo]:
    """Free stroke ends still free after the bridges: ends of final paths that are free and open."""
    open_ends = []
    for path in final_paths:
        if path.is_closed or not path.segments:
            continue
        if path.start_free:
            open_ends.append(path.segments[0].control_points[0])
        if path.end_free:
            open_ends.append(path.segments[-1].control_points[3])
    if not open_ends:
        return []
    tree = cKDTree(np.array(open_ends))
    return [ep for ep in extraction.endpoints if tree.query(ep.position)[0] < 1e-6]


def find_candidates(
    extraction: ExtractionResult, final_paths: Sequence[BezierPath],
) -> List[TJoinCandidate]:
    """One candidate per free end that sees a stroke ahead of it (see the module docstring)."""
    lw = max(float(extraction.line_width), 1.0)
    reach = REACH_WIDTHS * lw
    samples, owner, on_bridge = [], [], []
    for p, path in enumerate(final_paths):
        for seg in path.segments:
            pts = seg.sample(max(8, int(np.ceil(np.linalg.norm(seg.end - seg.start) / (lw / 2.0)))))
            samples.append(pts)
            owner += [p] * len(pts)
            on_bridge += [seg.source_type == "bridge"] * len(pts)
    if not samples:
        return []
    samples = np.vstack(samples)
    owner = np.array(owner)
    on_bridge = np.array(on_bridge)
    tree = cKDTree(samples)
    sources = free_sources(extraction, final_paths)
    source_positions = np.array([ep.position for ep in sources]).reshape(-1, 2)
    cos_cone = np.cos(np.radians(AIM_CONE_DEG))

    candidates = []
    for ep in sources:
        idx = np.array(tree.query_ball_point(ep.position, reach), dtype=int)
        if not len(idx):
            continue
        rel = samples[idx] - ep.position
        dist = np.linalg.norm(rel, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_aim = (rel @ ep.tangent) / dist
        ok = (dist > OWN_STROKE_SKIP_WIDTHS * lw) & (cos_aim >= cos_cone)
        if not ok.any():
            continue
        best = idx[ok][np.argmin(dist[ok])]
        q = samples[best]
        segment = _segment(ep.position, ep.tangent, q)
        points = segment.sample(max(10, int(np.ceil(np.linalg.norm(q - ep.position) / (lw / 2.0)))))
        candidates.append(TJoinCandidate(endpoint=ep, target=q, target_path=int(owner[best]), points=points,
                                         segment=segment))
        candidates[-1].measurements = _measure(candidates[-1], samples, owner, on_bridge[best], tree,
                                               source_positions, extraction, final_paths)
    return candidates


def _measure(c: TJoinCandidate, samples, owner, target_on_bridge, tree, source_positions, extraction, final_paths) -> Dict[str, float]:
    lw = max(float(extraction.line_width), 1.0)
    width = max(2.0 * c.endpoint.radius, 1e-6)
    gap = float(np.linalg.norm(c.target - c.endpoint.position))
    direction = _unit(c.target - c.endpoint.position)
    aim = float(np.degrees(np.arccos(np.clip(direction @ c.endpoint.tangent, -1.0, 1.0))))
    # Direction of the target stroke at the target: from its neighbouring samples.
    near = np.array(tree.query_ball_point(c.target, 2.0 * lw), dtype=int)
    near = near[owner[near] == c.target_path]
    stroke_dir = np.zeros(2)
    if len(near) >= 2:
        pts = samples[near] - samples[near].mean(axis=0)
        stroke_dir = np.linalg.svd(pts, full_matrices=False)[2][0]
    incidence = float(np.degrees(np.arccos(np.clip(abs(direction @ _unit(stroke_dir)), 0.0, 1.0)))) if np.any(stroke_dir) else 90.0
    # Existing stroke crossed between the end and the target (not counting the last line width).
    crossed = 0.0
    if extraction.dist_map is not None:
        h, w = extraction.dist_map.shape
        xy = np.round(c.points).astype(int)
        along = np.linalg.norm(c.points - c.endpoint.position, axis=1)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h) & (along > lw) & (along < gap - lw)
        step = gap / max(len(c.points) - 1, 1)
        crossed = float((extraction.dist_map[xy[inside, 1], xy[inside, 0]] == 0).sum() * step)
    path = extraction.paths[c.endpoint.path_index]
    stub = float(np.sum(np.linalg.norm(np.diff(path.sample(20), axis=0), axis=1)))
    others = source_positions[np.linalg.norm(source_positions - c.endpoint.position, axis=1) > 1e-6]
    nearest_end = float(np.min(np.linalg.norm(others - c.endpoint.position, axis=1))) if len(others) else np.inf
    return {
        "tjoin_length": max(gap / width, 1e-6),
        "tjoin_aim": aim,
        "tjoin_incidence": incidence,
        "tjoin_crossed_ink": crossed / width,
        "tjoin_stub": stub / width,
        "tjoin_free_end_near": max(gap / max(nearest_end, 1e-6), 1e-6) if np.isfinite(nearest_end) else 1e-3,
        "tjoin_on_bridge": float(bool(target_on_bridge)),
    }


def feature_matrix(candidates: Sequence[TJoinCandidate]) -> np.ndarray:
    """Measurements on their scorecard scale, one row per candidate (TJOIN_FEATURE_NAMES order)."""
    from restoration.scoring import _SCALE_FUNCTIONS

    m = np.array([[c.measurements[n] for n in TJOIN_FEATURE_NAMES] for c in candidates], dtype=np.float64).reshape(-1, len(TJOIN_FEATURE_NAMES))
    for name, scale in TJOIN_SCALES.items():
        col = TJOIN_FEATURE_NAMES.index(name)
        m[:, col] = _SCALE_FUNCTIONS[scale](m[:, col])
    return m


def score(candidates: Sequence[TJoinCandidate], card) -> None:
    """Probability of every candidate from the "tjoin" scorecard (restoration.scoring.Scorecard)."""
    if not candidates:
        return
    x = feature_matrix(candidates)
    columns = [TJOIN_FEATURE_NAMES.index(n) for n in card.names]
    for c, row in zip(candidates, x):
        terms = card.contributions(row[columns])
        c.contributions = dict(zip(card.names, map(float, terms)))
        c.probability = float(1.0 / (1.0 + np.exp(-(card.intercept + float(terms.sum())))))


def apply(final_paths: List[BezierPath], chosen: Sequence[TJoinCandidate]) -> List[BezierPath]:
    """Extend each chosen T-join's source path by its segment; the end is then no longer free."""
    for c in chosen:
        for path in final_paths:
            if path.is_closed or not path.segments:
                continue
            if path.end_free and np.linalg.norm(path.segments[-1].control_points[3] - c.endpoint.position) < 1e-6:
                path.segments.append(c.segment)
                path.end_free = False
                break
            if path.start_free and np.linalg.norm(path.segments[0].control_points[0] - c.endpoint.position) < 1e-6:
                cp = c.segment.control_points[::-1].copy()
                path.segments.insert(0, BezierSegment(control_points=cp, source_type="tjoin"))
                path.start_free = False
                break
    return final_paths
