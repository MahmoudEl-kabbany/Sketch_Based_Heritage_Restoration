"""Checks for geometric, scoring and solver properties the restoration pipeline relies on.

Synthetic inputs only (no dataset needed). Run with:  python -m pytest
"""
import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment, fit_straight_cubic
from restoration.asp_engine import solve_candidates
from restoration.candidates import ConnectionCandidate, corner_bridge, sample_bridge, smooth_bridge
from restoration.extraction import EndpointInfo, extract_paths
from restoration.geometry import safe_normalize
from restoration.pipeline import restore
from restoration.scoring import FEATURE_DIRECTIONS, FEATURES, compute_mjerk, context_curve, load_model


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
    segments = smooth_bridge(ep_a, ep_b)
    return ConnectionCandidate(
        id=cid, ep_a=ep_a, ep_b=ep_b, shape="smooth",
        bridge_points=sample_bridge(segments), bridge_bezier=segments,
        distance=float(np.linalg.norm(ep_b.position - ep_a.position)), score=score,
    )


def test_bridges_arrive_against_the_outward_tangent():
    left = polyline_path([[0, 0], [100, 0]])
    right = polyline_path([[160, 20], [260, 20]])
    ep_a, ep_b = endpoint(0, 0, left, "end"), endpoint(1, 1, right, "start")

    bridges = (
        smooth_bridge(ep_a, ep_b),
        corner_bridge(ep_a, ep_b, np.array([130.0, 10.0])),
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


# ═══════════════════════════════════════════════════════════════════════════
# Extraction, scoring and the whole pipeline on small drawn images
# ═══════════════════════════════════════════════════════════════════════════

def draw(path, strokes, size=(240, 320), thickness=5, scale=1):
    """Write a white image with black polyline strokes; return its path as a string."""
    image = np.full((size[0] * scale, size[1] * scale), 255, np.uint8)
    for stroke in strokes:
        points = (np.asarray(stroke, dtype=np.float64) * scale).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, 0, thickness * scale, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), image)
    return str(path)


def test_only_free_stroke_ends_are_endpoints(tmp_path):
    # A T: the stem ends on the bar. The bar's two tips and the stem's foot are free ends.
    bar, stem = [[60, 60], [260, 60]], [[160, 60], [160, 200]]
    extraction = extract_paths(draw(tmp_path / "t.png", [bar, stem]))

    tips = np.array([[60, 60], [260, 60], [160, 200]], dtype=np.float64)
    positions = np.array([ep.position for ep in extraction.endpoints])
    assert len(positions) == 3
    for tip in tips:
        assert np.linalg.norm(positions - tip, axis=1).min() < 10.0
    assert np.linalg.norm(positions - np.array([160.0, 60.0]), axis=1).min() > 40.0   # nothing at the junction


def test_extraction_does_not_depend_on_scale(tmp_path):
    strokes = [
        [[40, 40], [280, 40]],                       # a line
        [[40, 100], [140, 100], [140, 200]],         # an L
        [[200, 100], [200, 200]], [[160, 150], [280, 150]],   # a cross
    ]
    topologies, widths = [], []
    for scale in (1, 2, 4):
        extraction = extract_paths(draw(tmp_path / f"s{scale}.png", strokes, scale=scale))
        topologies.append((len(extraction.paths), len(extraction.endpoints)))
        widths.append(extraction.line_width)
    assert topologies[0] == topologies[1] == topologies[2] == (5, 8)
    # Strokes are drawn 5 * scale thick, plus an anti-aliased fringe.
    assert all(5 * scale <= width <= 5 * scale + 4 for scale, width in zip((1, 2, 4), widths))


def test_an_undamaged_drawing_is_left_alone(tmp_path):
    square = [[60, 40], [260, 40], [260, 200], [60, 200], [60, 40]]
    strokes = [square, [[100, 80], [220, 80]], [[100, 120], [220, 160]]]
    result = restore(draw(tmp_path / "whole.png", strokes), output_dir=str(tmp_path), save_images=False)
    assert result.bridges == []
    assert result.report["summary"]["total_restoration_events"] == 0


def test_a_gap_in_a_line_is_bridged_and_explained(tmp_path):
    strokes = [[[30, 120], [130, 120]], [[160, 120], [290, 120]], [[30, 60], [290, 60]]]
    result = restore(draw(tmp_path / "gap.png", strokes), output_dir=str(tmp_path), save_images=False)

    events = result.report["detailed_events"]
    assert len(events) == 1
    candidate = events[0]["candidate"]
    assert candidate["probability"] > candidate["threshold"]
    assert candidate["in_favour"]                                   # reasons are named
    assert set(candidate["measurements"]) <= set(FEATURES)
    middle = np.array(events[0]["coordinates"])
    assert np.linalg.norm(middle - np.array([145.0, 120.0])) < 6.0  # the bridge sits in the gap
    # The four outer tips stay open, each with a stated reason.
    assert len(result.report["open_ends"]) == 4
    assert all(entry["reason"] for entry in result.report["open_ends"])


def test_fitted_weights_point_the_expected_way():
    model = load_model()
    assert 0.5 <= model.threshold < 1.0
    assert set(model.scorecards) == {"smooth", "corner"}
    for card in model.scorecards.values():
        for name, weight in zip(card.names, card.weights):
            assert weight * FEATURE_DIRECTIONS.get(name, 0) >= 0.0, name
    # The measurements that carry the scoring do count against a bridge.
    smooth = dict(zip(model.scorecards["smooth"].names, model.scorecards["smooth"].weights))
    assert smooth["aim_best"] < 0.0 and smooth["crossed_ink"] < 0.0


def test_old_phase6_names_still_work(tmp_path):
    import importlib
    import warnings

    import pytest

    image = draw(tmp_path / "loop.png", [[[60, 40], [260, 40], [260, 200], [60, 200], [60, 60]]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = restore(image, output_dir=str(tmp_path), save_images=False, efd_closure=True, efd_gap_threshold=0.3)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    new = restore(image, output_dir=str(tmp_path), save_images=False, gap_closure=True, gap_closure_threshold=0.3)
    assert old.report["analysis"]["gap_closure_phase"] == new.report["analysis"]["gap_closure_phase"]
    assert old.report["analysis"]["gap_closure_phase"]["enabled"]
    with pytest.raises(TypeError):
        restore(image, efd_closure=True, gap_closure=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("restoration.efd_closure")
    from restoration import single_gap_closure
    assert legacy.close_single_gaps is single_gap_closure.close_single_gaps
    assert legacy._detect_symmetry is single_gap_closure._detect_symmetry


def test_spline_matches_scipy_bit_for_bit():
    from scipy.interpolate import CubicSpline

    from restoration.scoring import not_a_knot_spline

    rng = np.random.default_rng(0)
    for n in (4, 5, 17, 64):
        for _ in range(50):
            points = np.cumsum(rng.normal(size=(n, 2)), axis=0)
            x = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
            at = np.linspace(0.0, x[-1], 3 * n)
            assert np.array_equal(not_a_knot_spline(x, points, at), CubicSpline(x, points)(at))


def test_preprocessing_variants_are_validated_and_run(tmp_path, monkeypatch):
    import json

    import pytest

    from restoration.preprocess import ENV_VAR, PreprocessConfig

    with pytest.raises(ValueError):
        PreprocessConfig(threshold="nope")
    with pytest.raises(ValueError):
        PreprocessConfig.from_dict({"no_such_option": 1})
    monkeypatch.setenv(ENV_VAR, json.dumps({"threshold": "sauvola"}))
    assert PreprocessConfig.from_env().threshold == "sauvola"
    monkeypatch.delenv(ENV_VAR)
    assert PreprocessConfig.from_env() == PreprocessConfig()

    image = draw(tmp_path / "gap.png", [[[30, 120], [130, 120]], [[160, 120], [290, 120]], [[30, 60], [290, 60]]])
    variants = [
        {}, {"threshold": "sauvola"}, {"threshold": "niblack"}, {"threshold": "adaptive_gaussian"},
        {"denoise": "median"}, {"denoise": "nlmeans", "denoise_size": 10}, {"close_px": 3},
        {"skeleton": "lee"}, {"junction_aware_pruning": True},
    ]
    for overrides in variants:
        extraction = extract_paths(image, preprocess=PreprocessConfig(**overrides))
        assert len(extraction.endpoints) == 6, overrides     # three lines, two free ends each


def test_synthetic_damage_is_deterministic_and_matches_ground_truth(tmp_path):
    from evaluate_gaps import build_ground_truth
    from synthetic_damage import make_variant, split_of

    strokes = [
        [[40, 40], [460, 40]], [[40, 120], [460, 120]],                      # long straight lines
        [[60, 200], [250, 200], [250, 420]],                                 # a corner
        [[300, 180], [300, 460]], [[300, 320], [460, 320]],                  # a T junction
    ]
    clean = cv2.imread(draw(tmp_path / "clean.png", strokes, size=(500, 500)), cv2.IMREAD_GRAYSCALE)
    first = make_variant(clean, "t", 0, "hard")
    again = make_variant(clean, "t", 0, "hard")
    assert np.array_equal(first["damaged"], again["damaged"]) and first["gaps"] == again["gaps"]
    assert first["gaps"], "no gaps were placed"
    assert {g["type"] for g in first["gaps"]} <= {"straight", "curved", "corner", "junction"}
    # Every erased gap shows up in the ground truth built from (clean, gapped).
    gt = build_ground_truth(clean, first["gapped"])
    assert len(gt.gaps) >= len(first["gaps"])
    assert split_of("t") == split_of("t") and split_of("t") in ("train", "val", "test")


def test_committed_synthetic_samples_regenerate_exactly():
    import json
    import os

    from synthetic_damage import load_dataset, make_variant

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regression", "synthetic_samples")
    with open(os.path.join(root, "manifest.json")) as f:
        seed = json.load(f)["seed"]
    entries = load_dataset(root)
    assert len(entries) >= 20
    for e in entries:
        clean = cv2.imread(e["clean"], cv2.IMREAD_GRAYSCALE)
        result = make_variant(clean, e["drawing"], e["variant"], e["difficulty"], seed)
        assert np.array_equal(result["damaged"], cv2.imread(e["damaged"], cv2.IMREAD_GRAYSCALE)), e["damaged"]
        assert np.array_equal(result["gapped"], cv2.imread(e["gapped"], cv2.IMREAD_GRAYSCALE)), e["gapped"]
        with open(e["gaps"]) as f:
            assert json.load(f)["gaps"] == json.loads(json.dumps(result["gaps"]))


def test_boosted_model_round_trips_and_respects_directions(tmp_path):
    import train_scoring
    from restoration.scoring import FEATURE_DIRECTIONS, FEATURE_NAMES, SHAPE_FEATURES, load_model

    rng = np.random.default_rng(0)
    gap = FEATURE_NAMES.index("gap_length")
    images = []
    for _ in range(4):
        n = 400
        features = rng.normal(size=(n, len(FEATURE_NAMES)))
        shape = np.where(rng.random(n) < 0.7, "smooth", "corner").astype(object)
        correct = features[:, gap] + 0.3 * rng.normal(size=n) < -0.5          # short gaps are right
        images.append({"features": features, "shape": shape, "correct": correct})
    model = train_scoring.fit(images, kind="gbt")
    path = str(tmp_path / "gbt.json")
    train_scoring.write_model(path, model, 0.6, {})
    loaded = load_model(path)
    assert loaded.kind == "gbt" and loaded.threshold == 0.6
    for shape in SHAPE_FEATURES:
        rows = images[0]["shape"] == shape
        columns = [FEATURE_NAMES.index(n) for n in loaded.scorecards[shape].names]
        direct = train_scoring.predict(model, images[0])[rows]
        assert np.allclose(loaded.scorecards[shape].probability(images[0]["features"][rows][:, columns]), direct)
    # gap_length may only lower the probability (FEATURE_DIRECTIONS = -1).
    assert FEATURE_DIRECTIONS["gap_length"] == -1
    probe = np.zeros((50, len(FEATURE_NAMES)))
    probe[:, gap] = np.linspace(-3, 3, 50)
    image = {"features": probe, "shape": np.array(["smooth"] * 50, dtype=object)}
    assert np.all(np.diff(train_scoring.predict(model, image)) <= 1e-12)
    # The whole pipeline, report and figures included, runs with a boosted model.
    strokes = [[[30, 120], [130, 120]], [[160, 120], [290, 120]], [[30, 60], [290, 60]]]
    result = restore(draw(tmp_path / "gap.png", strokes), model_path=path, output_dir=str(tmp_path), save_images=True)
    assert result.report["analysis"]["scoring_model"]["kind"] == "gbt"
