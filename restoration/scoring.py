"""Phase 3 — Scoring.

Each candidate bridge is described by a handful of geometric measurements
that do not depend on the image's resolution or line width. A scorecard, a
logistic regression with one weight per measurement, turns them into the
probability that the bridge redraws a missing stroke. Smooth bridges and
corner bridges each have their own scorecard. The weights are fitted by
train_scoring.py and stored in scoring_model.json; none is set by hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from dtaidistance import dtw
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from restoration.candidates import ConnectionCandidate
from restoration.extraction import EndpointInfo, ExtractionResult

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scoring_model.json")

# Arc-length spacing (pixels) of the samples MJerk is computed on.
JERK_STEP_PX = 1.0

# MDTW and MJerk of a candidate are measured on its bridge together with the
# stroke on either side, resampled to this many points and scaled to this
# length, so that the image's resolution does not enter.
CONTEXT_SAMPLES = 64
CONTEXT_LENGTH = 100.0

# dtaidistance's compiled DTW computes the same recurrence as its pure-Python
# one, about 100x faster. Use it when the C library was built.
_DTW_OPTIONS = {"use_c": True} if dtw.try_import_c() else {}

# The measurements, with what each one means. A smooth bridge is judged by how
# the two strokes continue into each other, a corner bridge by the corner it makes.
FEATURES: Dict[str, str] = {
    # ---- any bridge
    "gap_length": "gap length in stroke widths",
    "width_mismatch": "width of the wider stroke relative to the narrower",
    "stub_shortness": "gap length relative to the length of the shorter stroke",
    "mutual_nearest": "share of the two ends for which the other is the nearest free end",
    "nearest_ratio": "gap length relative to the distance from each end to its nearest free end",
    "same_stroke": "the two ends belong to one stroke (1 = yes: the bridge closes it)",
    "crossed_ink": "existing stroke the bridge runs over between its ends, in stroke widths",
    # ---- smooth bridges
    "aim_worst": "degrees by which the worse-aimed end points away from the other end",
    "aim_best": "degrees by which the better-aimed end points away from the other end",
    "turn": "change of direction between the two strokes, in degrees",
    "sideways_offset": "stroke widths by which one stroke, continued straight, passes beside the other end",
    "arc_miss": "gap lengths by which one stroke, continued with its own curvature, passes beside the other end",
    "mdtw": "MDTW: deviation of the bridge and the stroke on either side from a straight line",
    "mjerk": "MJerk: abruptness of the change of curvature along the bridge and the stroke on either side",
    # ---- corner bridges
    "corner_turn": "change of direction at the corner, in degrees",
    "corner_detour": "extra length of the corner bridge over the straight gap, in gap lengths",
    "corner_balance": "how unequal the two legs of the corner are (0 = equal, 1 = one leg only)",
    "corner_bend": "how curved the two strokes are where the corner bridge meets them (0 = straight)",
}
FEATURE_NAMES = tuple(FEATURES)

# The measurements on the scorecard of each bridge shape.
_ANY_BRIDGE = FEATURE_NAMES[:FEATURE_NAMES.index("aim_worst")]
SHAPE_FEATURES: Dict[str, Tuple[str, ...]] = {
    "smooth": _ANY_BRIDGE + FEATURE_NAMES[FEATURE_NAMES.index("aim_worst"):FEATURE_NAMES.index("corner_turn")],
    "corner": _ANY_BRIDGE + FEATURE_NAMES[FEATURE_NAMES.index("corner_turn"):],
}

# How a measurement enters its scorecard. Lengths and ratios range over orders
# of magnitude and enter by their logarithm; the others enter as they are.
FEATURE_SCALES: Dict[str, str] = {
    "gap_length": "log2(x)", "width_mismatch": "log2(x)", "stub_shortness": "log2(x)", "nearest_ratio": "log2(x)",
    "crossed_ink": "log2(1 + x)", "sideways_offset": "log2(1 + x)",
    "mdtw": "ln(1 + x)", "mjerk": "log10(1 + 1e10 x)",
}
_SCALE_FUNCTIONS = {
    "log2(x)": np.log2,
    "log2(1 + x)": lambda x: np.log2(1.0 + x),
    "ln(1 + x)": np.log1p,
    "log10(1 + 1e10 x)": lambda x: np.log10(1.0 + 1e10 * x),
}

# Which way a measurement may move a bridge's odds: -1 where more of it can only
# count against the bridge, +1 where it can only count for it. The fit keeps to
# these directions, so that every weight reads the way its meaning suggests.
# Measurements not listed may count either way.
FEATURE_DIRECTIONS: Dict[str, int] = {
    "gap_length": -1, "width_mismatch": -1, "stub_shortness": -1, "nearest_ratio": -1, "crossed_ink": -1,
    "aim_worst": -1, "aim_best": -1, "turn": -1, "sideways_offset": -1, "arc_miss": -1, "mdtw": -1, "mjerk": -1,
    "corner_detour": -1, "corner_balance": -1, "corner_bend": -1,
    "mutual_nearest": 1,
}


# ═══════════════════════════════════════════════════════════════════════════
# Metric computations
# ═══════════════════════════════════════════════════════════════════════════

def context_curve(
    candidate: ConnectionCandidate,
    dense_samples: List[np.ndarray],
    tail_len: int = 20,
) -> np.ndarray:
    """The candidate's bridge with its local context, as one continuous curve.

    *dense_samples[i]* are the samples of path i from its start to its end.
    The curve is the last *tail_len* samples of the source path running into
    ep_a, the bridge from ep_a to ep_b, then the first *tail_len* samples of
    the target path running away from ep_b.
    """
    source = dense_samples[candidate.ep_a.path_index]
    target = dense_samples[candidate.ep_b.path_index]
    into_a = source[-tail_len:] if candidate.ep_a.end == "end" else source[:tail_len][::-1]
    away_from_b = target[:tail_len] if candidate.ep_b.end == "start" else target[-tail_len:][::-1]
    return np.vstack([into_a, candidate.bridge_points, away_from_b])


def compute_mdtw(combined: np.ndarray) -> float:
    """DTW distance between a curve and the straight line joining its ends.

    Lower = smoother, more natural connection.
    """
    if len(combined) < 4:
        return 0.0

    # Baseline: straight line from start to end of combined
    baseline = np.linspace(combined[0], combined[-1], len(combined))

    # Compute DTW on x and y independently, then combine
    dtw_x = dtw.distance(combined[:, 0].astype(np.double),
                         baseline[:, 0].astype(np.double), **_DTW_OPTIONS)
    dtw_y = dtw.distance(combined[:, 1].astype(np.double),
                         baseline[:, 1].astype(np.double), **_DTW_OPTIONS)
    return float(np.hypot(dtw_x, dtw_y))


def compute_mjerk(combined: np.ndarray) -> float:
    """Mean squared jerk (3rd derivative with respect to arc length) along a curve.

    Lower = smoother. The samples are first replaced by points every
    JERK_STEP_PX along a cubic spline through them. Differentiating the raw
    samples measures the sampling instead of the shape: Bezier samples repeat
    the point at each segment joint (a zero-length step) and are unevenly
    spaced, and a polyline has a small kink at every sample.
    """
    steps = np.linalg.norm(np.diff(combined, axis=0), axis=1)
    points = combined[np.concatenate(([True], steps > 1e-9))]
    if len(points) < 4:
        return 0.0

    # Chord length along the samples approximates arc length.
    chord = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    length = float(chord[-1])
    if length < 4.0 * JERK_STEP_PX:
        return 0.0

    num_points = int(np.ceil(length / JERK_STEP_PX)) + 1
    step = length / (num_points - 1)
    resampled = CubicSpline(chord, points)(np.linspace(0.0, length, num_points))
    jerk = np.diff(resampled, n=3, axis=0) / step ** 3
    return float(np.mean(np.sum(jerk ** 2, axis=1)))


# ═══════════════════════════════════════════════════════════════════════════
# Measurements of one candidate
# ═══════════════════════════════════════════════════════════════════════════

class _End(NamedTuple):
    """What the scorer needs to know about the stroke behind a free end."""

    into: np.ndarray       # samples of the stroke, ordered so that they run into the end
    back: np.ndarray       # arc length from each of those samples to the end
    stroke_length: float
    nearest: float         # distance to the nearest other free end


def _signed_angle(reference: np.ndarray, direction: np.ndarray) -> float:
    """Angle from *reference* to *direction*, in (-pi, pi]."""
    cross = float(reference[0] * direction[1] - reference[1] * direction[0])
    return float(np.arctan2(cross, float(np.dot(reference, direction))))


def _end_contexts(result: ExtractionResult) -> Dict[int, _End]:
    """The stroke behind every free end, keyed by endpoint id."""
    endpoints = result.endpoints
    positions = np.array([ep.position for ep in endpoints]).reshape(-1, 2)
    if len(endpoints) > 1:
        nearest = cKDTree(positions).query(positions, k=2)[0][:, 1]
    else:
        nearest = np.full(len(endpoints), np.inf)

    dense: Dict[int, np.ndarray] = {}
    contexts: Dict[int, _End] = {}
    for ep, nearest_distance in zip(endpoints, nearest):
        if ep.path_index not in dense:
            dense[ep.path_index] = result.paths[ep.path_index].sample(pts_per_segment=50)
        samples = dense[ep.path_index]
        into = samples if ep.end == "end" else samples[::-1]
        steps = np.linalg.norm(np.diff(into, axis=0), axis=1)
        back = np.concatenate((np.cumsum(steps[::-1])[::-1], [0.0]))
        contexts[ep.endpoint_id] = _End(into, back, float(back[0]), float(nearest_distance))
    return contexts


def _tail(end: _End, length: float) -> np.ndarray:
    """The last *length* of the stroke running into an end (at least two samples)."""
    first = int(np.searchsorted(-end.back, -length))   # end.back decreases towards the end
    return end.into[max(0, min(first, len(end.into) - 2)):]


def _resample(points: np.ndarray, count: Optional[int] = None, spacing: float = 1.0) -> Optional[np.ndarray]:
    """Points at equal arc-length steps along a polyline: *count* of them, or one every *spacing*."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate(([True], steps > 1e-9))]
    if len(points) < 2:
        return None
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if count is None:
        count = int(np.ceil(arc[-1] / spacing)) + 1
    grid = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(grid, arc, points[:, 0]), np.interp(grid, arc, points[:, 1])])


def _arc_miss(ep: EndpointInfo, curvature: float, target: np.ndarray, gap: float) -> float:
    """How close the stroke gets to *target* when continued with its own curvature, in gap lengths."""
    s = np.linspace(0.0, 2.0 * gap, 41)
    tangent = ep.tangent
    normal = np.array([-tangent[1], tangent[0]])
    if abs(curvature) * gap < 1e-4:
        along, across = s, np.zeros_like(s)
    else:
        along, across = np.sin(curvature * s) / curvature, (1.0 - np.cos(curvature * s)) / curvature
    arc = ep.position + along[:, np.newaxis] * tangent + across[:, np.newaxis] * normal
    return float(np.min(np.linalg.norm(arc - target, axis=1)) / gap)


def _crossed_ink(bridge_points: np.ndarray, dist_map: Optional[np.ndarray]) -> float:
    """Length (pixels) of existing stroke the bridge runs over, not counting the two strokes it joins."""
    if dist_map is None:
        return 0.0
    samples = _resample(bridge_points, spacing=1.0)
    if samples is None:
        return 0.0
    h, w = dist_map.shape
    xy = np.round(samples).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ink = np.zeros(len(xy), dtype=bool)
    ink[inside] = dist_map[xy[inside, 1], xy[inside, 0]] == 0
    if ink.all():
        return 0.0
    # The bridge starts and ends on the strokes it joins; only ink in between counts.
    first_clear = int(np.argmin(ink))
    last_clear = len(ink) - 1 - int(np.argmin(ink[::-1]))
    return float(ink[first_clear:last_clear + 1].sum())


def _measure(candidate: ConnectionCandidate, ends: Dict[int, _End], result: ExtractionResult) -> List[float]:
    """The FEATURES of one candidate, in order; NaN for those not on the scorecard of its shape."""
    ep_a, ep_b = candidate.ep_a, candidate.ep_b
    end_a, end_b = ends[ep_a.endpoint_id], ends[ep_b.endpoint_id]
    gap = max(candidate.distance, 1e-6)
    width = max(ep_a.radius + ep_b.radius, 1e-6)   # mean width of the two strokes
    toward_b = (ep_b.position - ep_a.position) / gap

    angle_a = _signed_angle(toward_b, ep_a.tangent)
    angle_b = _signed_angle(-toward_b, ep_b.tangent)
    # ep_a's tangent and the reverse of ep_b's are the direction of travel before and after the gap.
    turn = abs((angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi)
    radii = sorted((max(ep_a.radius, 1e-6), max(ep_b.radius, 1e-6)))

    values = dict.fromkeys(FEATURE_NAMES, float("nan"))
    values.update(
        gap_length=gap / width,
        width_mismatch=radii[1] / radii[0],
        stub_shortness=gap / max(min(end_a.stroke_length, end_b.stroke_length), 1e-6),
        mutual_nearest=((gap <= end_a.nearest + 1e-9) + (gap <= end_b.nearest + 1e-9)) / 2.0,
        nearest_ratio=gap / max(np.sqrt(end_a.nearest * end_b.nearest), 1e-6),   # geometric mean of the two ratios
        same_stroke=float(candidate.same_path_closure),
        crossed_ink=_crossed_ink(candidate.bridge_points, result.dist_map) / width,
    )

    if candidate.shape == "corner":
        corner = np.asarray(candidate.corner_point, dtype=np.float64)
        leg_a = float(np.linalg.norm(corner - ep_a.position))
        leg_b = float(np.linalg.norm(ep_b.position - corner))
        values.update(
            corner_turn=np.degrees(turn),
            corner_detour=(leg_a + leg_b) / gap - 1.0,
            corner_balance=abs(leg_a - leg_b) / max(leg_a + leg_b, 1e-6),
            corner_bend=min(1.0, (ep_a.curvature + ep_b.curvature) * gap / 2.0),
        )
        return [float(values[name]) for name in FEATURE_NAMES]

    def straight_miss(angle: float) -> float:
        return gap * np.sin(abs(angle)) if abs(angle) < np.pi / 2.0 else gap

    context = _resample(
        np.vstack([_tail(end_a, gap), candidate.bridge_points, _tail(end_b, gap)[::-1]]),
        count=CONTEXT_SAMPLES,
    )
    if context is None:
        mdtw = mjerk = 0.0
    else:
        length = float(np.sum(np.linalg.norm(np.diff(context, axis=0), axis=1)))
        context = (context - context[0]) * (CONTEXT_LENGTH / max(length, 1e-9))
        mdtw, mjerk = compute_mdtw(context), compute_mjerk(context)

    values.update(
        aim_worst=np.degrees(max(abs(angle_a), abs(angle_b))),
        aim_best=np.degrees(min(abs(angle_a), abs(angle_b))),
        turn=np.degrees(turn),
        sideways_offset=max(straight_miss(angle_a), straight_miss(angle_b)) / width,
        arc_miss=max(_arc_miss(ep_a, ep_a.signed_curvature, ep_b.position, gap),
                     _arc_miss(ep_b, ep_b.signed_curvature, ep_a.position, gap)),
        mdtw=mdtw,
        mjerk=mjerk,
    )
    return [float(values[name]) for name in FEATURE_NAMES]


def measure_candidates(candidates: List[ConnectionCandidate], result: ExtractionResult) -> np.ndarray:
    """Matrix of FEATURES in their natural units, one row per candidate; NaN for those not on the scorecard of its shape."""
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)))
    ends = _end_contexts(result)
    return np.array([_measure(c, ends, result) for c in candidates], dtype=np.float64)


def on_scorecard_scale(measurements: np.ndarray) -> np.ndarray:
    """Measurements (columns in FEATURES order) on the scale they enter a scorecard with."""
    scaled = np.array(measurements, dtype=np.float64)
    for name, scale in FEATURE_SCALES.items():
        column = FEATURE_NAMES.index(name)
        scaled[:, column] = _SCALE_FUNCTIONS[scale](scaled[:, column])
    return scaled


def candidate_features(candidates: List[ConnectionCandidate], result: ExtractionResult) -> np.ndarray:
    """What the scorecards are fitted on: the candidates' measurements on their scorecard scale."""
    return on_scorecard_scale(measure_candidates(candidates, result))


# ═══════════════════════════════════════════════════════════════════════════
# The fitted model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Scorecard:
    """A logistic regression on standardized measurements, for bridges of one shape."""

    names: List[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def contributions(self, features: np.ndarray) -> np.ndarray:
        """Log-odds each measurement (on its scorecard scale) adds, relative to its average value."""
        return (features - self.mean) / self.scale * self.weights


@dataclass
class ScoringModel:
    """A scorecard per bridge shape, and the probability above which a bridge is drawn."""

    scorecards: Dict[str, Scorecard]
    threshold: float

    @property
    def threshold_log_odds(self) -> float:
        return float(np.log(self.threshold / (1.0 - self.threshold)))


def load_model(path: str = MODEL_PATH) -> ScoringModel:
    """Read a model written by train_scoring.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scoring model at {path}; run train_scoring.py to fit one.")
    with open(path) as f:
        data = json.load(f)

    scorecards: Dict[str, Scorecard] = {}
    for shape, card in data["scorecards"].items():
        unknown = [feature["name"] for feature in card["features"] if feature["name"] not in SHAPE_FEATURES.get(shape, ())]
        if unknown:
            raise ValueError(f"Scoring model {path}: unknown measurements for {shape} bridges: {unknown}")
        scorecards[shape] = Scorecard(
            names=[feature["name"] for feature in card["features"]],
            mean=np.array([feature["mean"] for feature in card["features"]], dtype=np.float64),
            scale=np.array([feature["scale"] for feature in card["features"]], dtype=np.float64),
            weights=np.array([feature["weight"] for feature in card["features"]], dtype=np.float64),
            intercept=float(card["intercept"]),
        )
    return ScoringModel(scorecards=scorecards, threshold=float(data["threshold"]))


def held_out_models(directory: str, drawings: List[str]) -> Dict[str, str]:
    """For each drawing, the path of the model that train_scoring.py --cv-dir fitted without it."""
    with open(os.path.join(directory, "folds.json")) as f:
        model_of = {name: os.path.join(directory, filename) for name, filename in json.load(f).items()}
    unseen = [name for name in drawings if name not in model_of]
    if unseen:
        raise KeyError(f"No held-out model in {directory} for: {', '.join(unseen)}")
    return model_of


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def score_candidates(
    candidates: List[ConnectionCandidate],
    result: ExtractionResult,
    model: Union[ScoringModel, str, None] = None,
) -> List[ConnectionCandidate]:
    """Score all candidates in place and return them best first.

    Each candidate gets its measurements (in their natural units), the
    log-odds every measurement contributes, its probability, and its score:
    the log-odds above the model's acceptance threshold.
    """
    if not candidates:
        return []
    if not isinstance(model, ScoringModel):
        model = load_model(model or MODEL_PATH)

    measurements = measure_candidates(candidates, result)
    for c, measured, scaled in zip(candidates, measurements, on_scorecard_scale(measurements)):
        card = model.scorecards[c.shape]
        columns = [FEATURE_NAMES.index(name) for name in card.names]
        terms = card.contributions(scaled[columns])
        log_odds = card.intercept + float(terms.sum())
        c.features = dict(zip(card.names, map(float, measured[columns])))
        c.contributions = dict(zip(card.names, map(float, terms)))
        c.probability = float(1.0 / (1.0 + np.exp(-log_odds)))
        c.score = log_odds - model.threshold_log_odds

    candidates.sort(key=lambda c: (-c.score, c.distance, c.id))
    return candidates
