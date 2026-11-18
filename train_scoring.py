"""Fit the candidate scorer from drawings whose missing strokes are known.

For every drawing with an undamaged original, candidate bridges are generated
on its damaged versions and labelled from the original: a candidate is correct
when it redraws the erased stroke of a true gap. For each bridge shape a
scorecard is fitted to these labels: a logistic regression with one weight per
geometric measurement, each weight held to the direction its meaning allows.

Everything reported is cross-validated by drawing: the numbers for a drawing
come from a model that has never seen that drawing in any damaged form.

Usage:
    python train_scoring.py                       # fit, report, write restoration/scoring_model.json
    python train_scoring.py --cv-dir models/      # also write the held-out model of every drawing
    python train_scoring.py --dataset data.pkl    # keep the labelled candidates between runs
"""

from __future__ import annotations

import argparse
import json
import dataclasses
import os
import pickle
import random
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from add_gaps import apply_gaps
from evaluate_gaps import build_ground_truth, label_candidates, load_pair
from evaluate_restoration import discover_pairs
from restoration.asp_engine import solve_candidates
from restoration.candidates import generate_candidates
from restoration.extraction import extract_paths
from restoration.pipeline import default_workers, map_in_processes
from restoration.render_restored_sketch import estimate_line_width
from restoration.graph_features import build_graph
from restoration.preprocess import ENV_VAR as PREPROCESS_ENV_VAR, PreprocessConfig
from restoration.scoring import (
    FEATURE_DIRECTIONS, FEATURE_NAMES, FEATURE_SCALES, FEATURES, MODEL_PATH, SHAPE_FEATURES, candidate_features,
)

FOLDS = 5
# Extra damaged versions made of every drawing, and the size of their gaps in stroke widths.
SYNTHETIC_VERSIONS = 3
SYNTHETIC_GAP_TYPES = ("box", "circle")
SYNTHETIC_HALF_SIZE_WIDTHS = (1.0, 4.0)
# A bridge is drawn when it is at least this many times as likely to be right as wrong.
# Fewer wrong strokes matter more than more repaired gaps, hence more than even odds.
ACCEPTANCE_ODDS = 2.0
THRESHOLD = ACCEPTANCE_ODDS / (1.0 + ACCEPTANCE_ODDS)
# Thresholds whose outcome on held-out drawings is reported next to the one in use.
REPORTED_THRESHOLDS = (0.33, 0.5, THRESHOLD, 0.8, 0.9)


# ═══════════════════════════════════════════════════════════════════════════
# Labelled candidates
# ═══════════════════════════════════════════════════════════════════════════

def make_synthetic_damage(original_path: str, out_path: str, version: int, gap_count: int) -> None:
    """Write a damaged copy of a drawing, with gaps a few stroke widths across."""
    image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    line_width = float(estimate_line_width(image))
    low, high = (max(2, int(round(factor * line_width))) for factor in SYNTHETIC_HALF_SIZE_WIDTHS)
    random.seed(f"{os.path.basename(original_path)}-{version}")
    damaged = apply_gaps(
        image,
        num_gaps=gap_count,
        gap_size_min=low,
        gap_size_max=max(high, low + 1),
        gap_type=SYNTHETIC_GAP_TYPES[version % len(SYNTHETIC_GAP_TYPES)],
    )
    cv2.imwrite(out_path, damaged)


def training_preprocess() -> PreprocessConfig:
    """How training images are binarized: as at restoration time, but without background flattening.

    The shipped model was fitted on plain-Otsu images; flattening the synthetic
    "medium" pages in training shifted its weights and cost 13 line widths off
    the drawing on the real pairs (docs/experiments/hard-binarization.md), while
    at restoration time flattening only touches unevenly lit pages. An explicit
    SKETCH_PREPROCESS experiment applies to training and restoration alike.
    """
    if os.environ.get(PREPROCESS_ENV_VAR, "").strip():
        return PreprocessConfig.from_env()
    return dataclasses.replace(PreprocessConfig(), background="none")


def labelled_candidates(
    drawing: str, kind: str, original_path: str, image_path: str, reference_path: Optional[str] = None,
    with_graph: bool = False,
) -> Dict[str, Any]:
    """Candidates of one image with their measurements and what the original says about them.

    *reference_path*: for degraded synthetic damage, the gapped image without
    degradation, from which ground truth is taken (see synthetic_damage.py).
    """
    original, damaged, agreement = load_pair(original_path, reference_path or image_path)
    if original.shape != damaged.shape or agreement < 0.98:
        return {"drawing": drawing, "kind": kind, "skipped": True}
    gt = build_ground_truth(original, damaged)

    extraction = extract_paths(image_path, preprocess=training_preprocess())
    candidates = generate_candidates(extraction)
    labels = label_candidates(gt, candidates)
    return {
        "drawing": drawing,
        "kind": kind,
        "features": candidate_features(candidates, extraction),
        "shape": np.array([c.shape for c in candidates], dtype=object),
        "correct": np.array([row["correct"] for row in labels], dtype=bool),
        "unfounded": np.array([row["unfounded"] for row in labels], dtype=bool),
        "on_drawing": np.array([row["on_drawing"] for row in labels], dtype=float),
        "length_widths": np.array([row["length_widths"] for row in labels], dtype=float),
        "gaps": [row["gaps"] for row in labels],
        "endpoints": np.array([[c.ep_a.endpoint_id, c.ep_b.endpoint_id] for c in candidates], dtype=int).reshape(-1, 2),
        "bridges": [c.bridge_points.astype(np.float32) for c in candidates],
        "gap_count": len(gt.gaps),
        **({"graph": build_graph(extraction, candidates)} if with_graph else {}),
    }


def build_dataset(
    work_dir: str,
    workers: int,
    name_filter: Optional[str] = None,
    extra_data: Optional[str] = None,
    extra_split: str = "train",
    extra_difficulties: Optional[Sequence[str]] = None,
    with_graph: bool = False,
) -> List[Dict[str, Any]]:
    """Labelled candidates of every drawing: its provided damage, synthetic damage, and no damage.

    *extra_data*: a dataset written by scripts/make_synthetic.py; the variants
    of its *extra_split* ("train", "val", "test" or "all") are added, as kind
    "extra_<difficulty>", under their source drawing's name (so a benchmark
    drawing's synthetic variants share its cross-validation fold).
    *extra_difficulties* limits them to these levels (default: all).
    """
    jobs = dataset_jobs(work_dir, name_filter, extra_data, extra_split, extra_difficulties)
    if with_graph:   # graphs for the context-aware scorer (--model gnn)
        jobs = [tuple(job) + (None,) * (5 - len(job)) + (True,) for job in jobs]
    images = []
    for i, image in enumerate(map_in_processes(labelled_candidates, jobs, workers), 1):
        if image.get("skipped"):
            print(f"  [{i}/{len(jobs)}] {image['drawing']} ({image['kind']}): skipped, does not line up with its original")
            continue
        print(f"  [{i}/{len(jobs)}] {image['drawing']} ({image['kind']}): "
              f"{len(image['correct'])} candidates, {int(image['correct'].sum())} correct, "
              f"{image['gap_count']} gaps")
        images.append(image)
    return images


def dataset_jobs(
    work_dir: str,
    name_filter: Optional[str] = None,
    extra_data: Optional[str] = None,
    extra_split: str = "train",
    extra_difficulties: Optional[Sequence[str]] = None,
) -> List[Tuple[Any, ...]]:
    """The training images as (drawing, kind, original, image[, reference]) jobs; writes the synthetic damage."""
    os.makedirs(work_dir, exist_ok=True)
    jobs: List[Tuple[Any, ...]] = []
    for name, original_path, damaged_path in discover_pairs(name_filter=name_filter):
        original, damaged, agreement = load_pair(original_path, damaged_path)
        gap_count = 20
        if original.shape == damaged.shape and agreement >= 0.98:
            gap_count = max(5, len(build_ground_truth(original, damaged).gaps))
        jobs.append((name, "provided", original_path, damaged_path))
        jobs.append((name, "undamaged", original_path, original_path))
        for version in range(SYNTHETIC_VERSIONS):
            synthetic_path = os.path.join(work_dir, f"{name}_synthetic{version}.png")
            make_synthetic_damage(original_path, synthetic_path, version, gap_count)
            jobs.append((name, f"synthetic{version}", original_path, synthetic_path))
    if extra_data:
        from synthetic_damage import load_dataset
        for entry in load_dataset(extra_data, None if extra_split == "all" else extra_split):
            if name_filter and name_filter.lower() not in entry["drawing"].lower():
                continue
            if extra_difficulties and entry["difficulty"] not in extra_difficulties:
                continue
            jobs.append((entry["drawing"], f"extra_{entry['difficulty']}", entry["clean"], entry["damaged"],
                         entry["gapped"]))
    return jobs


# ═══════════════════════════════════════════════════════════════════════════
# T-joins (restoration/tjoins.py)
# ═══════════════════════════════════════════════════════════════════════════

def labelled_tjoins(
    drawing: str, kind: str, original_path: str, image_path: str, reference_path: Optional[str] = None,
    model_path: str = MODEL_PATH,
) -> Dict[str, Any]:
    """T-join candidates of one image, after its bridges, with measurements and labels from the original.

    The bridges come from a restoration with *model_path* (without T-joins);
    a T-join is correct when it redraws erased stroke next to a true gap, as
    for bridges (evaluate_gaps.label_candidates).
    """
    import contextlib
    import io

    from restoration import tjoins
    from restoration.pipeline import restore

    original, damaged, agreement = load_pair(original_path, reference_path or image_path)
    if original.shape != damaged.shape or agreement < 0.98:
        return {"drawing": drawing, "kind": kind, "skipped": True}
    gt = build_ground_truth(original, damaged)
    with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
        result = restore(image_path, model_path=model_path, output_dir=out, save_images=False,
                         preprocess=training_preprocess())
    candidates = tjoins.find_candidates(result.extraction, result.restored_paths)
    proxies = [SimpleNamespace(ep_a=SimpleNamespace(position=c.endpoint.position),
                               ep_b=SimpleNamespace(position=c.target), bridge_points=c.points) for c in candidates]
    labels = label_candidates(gt, proxies) if proxies else []
    return {
        "drawing": drawing, "kind": kind,
        "features": tjoins.feature_matrix(candidates),
        "correct": np.array([row["correct"] for row in labels], dtype=bool),
        "unfounded": np.array([row["unfounded"] for row in labels], dtype=bool),
        "gap_count": len(gt.gaps),
    }


def build_tjoin_dataset(
    work_dir: str, workers: int, extra_data: Optional[str] = None,
    extra_difficulties: Optional[Sequence[str]] = None, model_path: str = MODEL_PATH,
) -> List[Dict[str, Any]]:
    """labelled_tjoins of every training image (the same images as build_dataset)."""
    jobs = dataset_jobs(work_dir, None, extra_data, "train", extra_difficulties)
    jobs = [tuple(job) + (None,) * (5 - len(job)) + (model_path,) for job in jobs]
    return [image for image in map_in_processes(labelled_tjoins, jobs, workers) if not image.get("skipped")]


def fit_tjoin(images: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The T-join scorecard: a sign-constrained logistic regression like the bridge scorecards."""
    from restoration.tjoins import TJOIN_DIRECTIONS, TJOIN_FEATURE_NAMES

    rows = [i for i in images if len(i["correct"])]
    x = np.vstack([i["features"] for i in rows]) if rows else np.zeros((0, len(TJOIN_FEATURE_NAMES)))
    y = np.concatenate([i["correct"] for i in rows]).astype(float) if rows else np.zeros(0)
    card = _fit_logistic(x, y, list(TJOIN_FEATURE_NAMES), TJOIN_DIRECTIONS)
    card["kind"] = "tjoin"
    return card


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

def fit_scorecard(images: Sequence[Dict[str, Any]], shape: str, names: Sequence[str]) -> Dict[str, Any]:
    """Logistic regression on the standardized measurements of one bridge shape: one weight each.

    The weights minimize the log loss plus half their squared length (a mild
    pull towards zero), each within the direction FEATURE_DIRECTIONS allows.
    """
    columns = [FEATURE_NAMES.index(name) for name in names]
    x = np.vstack([image["features"][image["shape"] == shape][:, columns] for image in images])
    y = np.concatenate([image["correct"][image["shape"] == shape] for image in images]).astype(float)
    card = _fit_logistic(x, y, list(names), FEATURE_DIRECTIONS)
    card["columns"] = [columns[names.index(n)] for n in card["names"]]
    return card


def _fit_logistic(x: np.ndarray, y: np.ndarray, names: List[str], directions: Dict[str, int]) -> Dict[str, Any]:
    """Logistic regression on standardized columns, each weight within the direction *directions* allows."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale

    def loss(parameters: np.ndarray) -> Tuple[float, np.ndarray]:
        weights, intercept = parameters[:-1], parameters[-1]
        margin = z @ weights + intercept
        value = np.logaddexp(0.0, np.where(y > 0.5, -margin, margin)).sum() + 0.5 * weights @ weights
        error = expit(margin) - y
        return float(value), np.append(z.T @ error + weights, error.sum())

    allowed = {-1: (None, 0.0), 1: (0.0, None), 0: (None, None)}
    bounds = [allowed[directions.get(name, 0)] for name in names] + [(None, None)]
    result = minimize(loss, np.zeros(len(names) + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000})
    # A measurement the fit gives no weight (it would have counted the wrong way) is left off the scorecard.
    used = np.flatnonzero(result.x[:-1] != 0.0)
    return {"names": [names[i] for i in used],
            "mean": mean[used], "scale": scale[used],
            "weights": result.x[:-1][used], "intercept": float(result.x[-1]),
            "unused": [name for i, name in enumerate(names) if i not in used]}


MODEL_KINDS = ("scorecard", "gbt", "gnn")
# Options of the graph model (--gnn-relations, --gnn-epochs); see restoration/gnn.py.
GNN_OPTIONS: Dict[str, Any] = {}
# Gradient-boosted trees: small and regularized, a fixed seed and no early stopping
# (which would hold out a random share), so the fit is repeatable.
GBT_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=0)


def fit_boosted(images: Sequence[Dict[str, Any]], shape: str, names: Sequence[str]) -> Dict[str, Any]:
    """Gradient-boosted trees on the measurements of one bridge shape.

    Each measurement may only move the probability in the direction
    FEATURE_DIRECTIONS allows (monotonic constraints), as for the scorecards.
    No class weights: the probabilities are compared with a threshold, so they
    must stay calibrated.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    columns = [FEATURE_NAMES.index(name) for name in names]
    x = np.vstack([image["features"][image["shape"] == shape][:, columns] for image in images])
    y = np.concatenate([image["correct"][image["shape"] == shape] for image in images]).astype(int)
    estimator = HistGradientBoostingClassifier(
        monotonic_cst=[FEATURE_DIRECTIONS.get(name, 0) for name in names], **GBT_PARAMS,
    ).fit(x, y)
    return {"kind": "gbt", "names": list(names), "columns": columns, "estimator": estimator, "unused": []}


def fit(
    images: Sequence[Dict[str, Any]], names: Optional[Sequence[str]] = None, kind: str = "scorecard",
) -> Dict[str, Dict[str, Any]]:
    """A model for each bridge shape, on the measurements of that shape (only those in *names*, if given).

    *kind*: "scorecard" (logistic regression, the shipped model) or "gbt" (gradient-boosted trees).
    """
    if kind == "gnn":
        from restoration.gnn import GnnConfig, train as train_gnn

        scorer = train_gnn(images, GnnConfig(**GNN_OPTIONS))
        return {"all": {"kind": "gnn", "scorer": scorer, "names": list(FEATURE_NAMES), "unused": []}}
    fitter = fit_boosted if kind == "gbt" else fit_scorecard
    return {
        shape: fitter(images, shape, [n for n in shape_names if names is None or n in names])
        for shape, shape_names in SHAPE_FEATURES.items()
    }


def predict(model: Dict[str, Dict[str, Any]], image: Dict[str, Any]) -> np.ndarray:
    """Probability of every candidate of an image, each from the scorecard of its shape."""
    probability = np.zeros(len(image["shape"]))
    if "all" in model and model["all"].get("kind") == "gnn":
        return model["all"]["scorer"].probability(image["graph"], image["features"], list(image["shape"]))
    for shape, card in model.items():
        if card.get("kind") == "tjoin":
            continue
        rows = image["shape"] == shape
        if card.get("kind") == "gbt":
            if rows.any():
                probability[rows] = card["estimator"].predict_proba(image["features"][rows][:, card["columns"]])[:, 1]
            continue
        standardized = (image["features"][rows][:, card["columns"]] - card["mean"]) / card["scale"]
        probability[rows] = expit(card["intercept"] + standardized @ card["weights"])
    return probability


def fold_of_drawings(images: Sequence[Dict[str, Any]], folds: int = FOLDS) -> Dict[str, int]:
    """Assign every drawing to a fold, in a fixed shuffled order.

    Drawings that only appear through --extra-data are assigned after the
    others, so that adding extra data leaves the benchmark drawings' folds as they were.
    """
    drawings = sorted({image["drawing"] for image in images if not image["kind"].startswith("extra_")})
    order = np.random.RandomState(0).permutation(len(drawings))
    assignment = {drawings[index]: position % folds for position, index in enumerate(order)}
    extra = sorted({image["drawing"] for image in images} - set(assignment))
    assignment.update({drawing: position % folds for position, drawing in enumerate(extra)})
    return assignment


# ═══════════════════════════════════════════════════════════════════════════
# Selection at a threshold
# ═══════════════════════════════════════════════════════════════════════════

def select(image: Dict[str, Any], probabilities: np.ndarray, threshold: float) -> List[int]:
    """Candidates the solver accepts when bridges above *threshold* are worth drawing."""
    offset = np.log(threshold / (1.0 - threshold))
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    scores = np.log(clipped / (1.0 - clipped)) - offset
    candidates = [
        SimpleNamespace(
            id=i,
            ep_a=SimpleNamespace(endpoint_id=int(image["endpoints"][i, 0])),
            ep_b=SimpleNamespace(endpoint_id=int(image["endpoints"][i, 1])),
            score=float(scores[i]),
            bridge_points=image["bridges"][i],
        )
        for i in range(len(scores))
    ]
    return solve_candidates(candidates)[0]


TALLY_KEYS = ("bridges", "unfounded", "wrong", "off_drawing_widths", "repaired", "gaps")


def tally(image: Dict[str, Any], probabilities: np.ndarray, thresholds: Sequence[float]) -> List[Dict[str, float]]:
    """Outcome of selection on one image at each threshold."""
    rows = []
    for threshold in thresholds:
        accepted = select(image, probabilities, threshold)
        repaired = {gap for i in accepted for gap in image["gaps"][i]}
        rows.append({
            "bridges": len(accepted),
            "unfounded": int(sum(image["unfounded"][i] for i in accepted)),
            "wrong": int(sum(image["on_drawing"][i] < 0.5 for i in accepted)),
            "off_drawing_widths": float(sum(image["length_widths"][i] * (1.0 - image["on_drawing"][i]) for i in accepted)),
            "repaired": len(repaired),
            "gaps": image["gap_count"],
        })
    return rows


def sweep(
    images: Sequence[Dict[str, Any]], probabilities: Sequence[np.ndarray],
    thresholds: Sequence[float], workers: int,
) -> List[Dict[str, float]]:
    """Totals over *images* of the selection outcome at each threshold."""
    light = [{key: image[key] for key in ("endpoints", "bridges", "gaps", "on_drawing", "unfounded", "length_widths", "gap_count")}
             for image in images]
    jobs = [(image, p, tuple(thresholds)) for image, p in zip(light, probabilities)]
    totals = [dict.fromkeys(TALLY_KEYS, 0.0) for _ in thresholds]
    for rows in map_in_processes(tally, jobs, workers):
        for total, row in zip(totals, rows):
            for key, value in row.items():
                total[key] += value
    return totals


# ═══════════════════════════════════════════════════════════════════════════
# Reporting and output
# ═══════════════════════════════════════════════════════════════════════════

def separation(images: Sequence[Dict[str, Any]], scores: Sequence[np.ndarray]) -> Dict[str, float]:
    """How well *scores* tell correct candidates from the rest."""
    y = np.concatenate([image["correct"] for image in images])
    s = np.concatenate(scores)

    # For each stroke end that has a correct candidate: is its best-scored candidate a correct one?
    hits = total = 0
    for image, score in zip(images, scores):
        for endpoint in np.unique(image["endpoints"][image["correct"]]):
            involved = np.flatnonzero((image["endpoints"] == endpoint).any(axis=1))
            hits += bool(image["correct"][involved[np.argmax(score[involved])]])
            total += 1
    # Chance that a correct candidate outscores a wrong one (the area under the ROC curve).
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    auc = float("nan")
    if positives and negatives:
        auc = float((rankdata(s)[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))
    return {
        "auc": auc,
        "best_candidate_correct": hits / max(total, 1),
        "candidates": int(len(y)),
        "correct": int(y.sum()),
    }


def model_document(model: Dict[str, Dict[str, Any]], threshold: float, training: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "description": (
            "Scorecards giving the probability that a candidate bridge redraws a missing stroke, one per "
            "bridge shape. log-odds = intercept + sum of weight * (transform(value) - mean) / scale. "
            "Fitted by train_scoring.py."
        ),
        "threshold": threshold,
        "scorecards": {
            shape: {
                "intercept": card["intercept"],
                "features": [
                    {"name": name, "meaning": FEATURES[name], "transform": FEATURE_SCALES.get(name, "x"),
                     "weight": float(weight), "mean": float(mean), "scale": float(scale)}
                    for name, weight, mean, scale in zip(card["names"], card["weights"], card["mean"], card["scale"])
                ],
            }
            for shape, card in model.items() if card.get("kind") != "tjoin"
        },
        "training": training,
        **({"tjoin": _card_document(model["tjoin"], _tjoin_meaning, _tjoin_scale)} if "tjoin" in model else {}),
    }


def _tjoin_meaning(name: str) -> str:
    from restoration.tjoins import TJOIN_FEATURES
    return TJOIN_FEATURES[name]


def _tjoin_scale(name: str) -> str:
    from restoration.tjoins import TJOIN_SCALES
    return TJOIN_SCALES.get(name, "x")


def _card_document(card: Dict[str, Any], meaning, transform) -> Dict[str, Any]:
    return {
        "intercept": card["intercept"],
        "features": [
            {"name": name, "meaning": meaning(name), "transform": transform(name),
             "weight": float(weight), "mean": float(mean), "scale": float(scale)}
            for name, weight, mean, scale in zip(card["names"], card["weights"], card["mean"], card["scale"])
        ],
    }


def write_model(path: str, model: Dict[str, Dict[str, Any]], threshold: float, training: Dict[str, Any]) -> None:
    """Write a fitted model where load_model() reads it: a scorecard JSON, or for
    boosted trees a JSON pointing to a joblib file next to it."""
    if "all" in model and model["all"].get("kind") == "gnn":
        import torch

        weights_file = os.path.splitext(os.path.basename(path))[0] + ".pt"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        scorer = model["all"]["scorer"]
        torch.save(scorer.state(), os.path.join(os.path.dirname(os.path.abspath(path)), weights_file))
        write_json(path, {
            "description": "A graph network over the drawing (restoration/gnn.py) giving the probability that a "
                           "candidate bridge redraws a missing stroke, from its two stroke ends in context and its "
                           "measurements. Fitted by train_scoring.py --model gnn.",
            "kind": "gnn",
            "threshold": threshold,
            "weights": weights_file,
            "config": scorer.config.to_dict(),
            "torch_version": torch.__version__,
            "training": training,
        })
        return
    if not any(card.get("kind") == "gbt" for card in model.values()):
        write_json(path, model_document(model, threshold, training))
        return
    import joblib
    import sklearn

    estimators_file = os.path.splitext(os.path.basename(path))[0] + ".joblib"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({shape: card["estimator"] for shape, card in model.items()},
                os.path.join(os.path.dirname(os.path.abspath(path)), estimators_file))
    write_json(path, {
        "description": "Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier) giving the probability "
                       "that a candidate bridge redraws a missing stroke, one per bridge shape, on the "
                       "scorecard-scale measurements. Fitted by train_scoring.py --model gbt.",
        "kind": "gbt",
        "threshold": threshold,
        "estimators": estimators_file,
        "features": {shape: card["names"] for shape, card in model.items()},
        "sklearn_version": sklearn.__version__,
        "parameters": GBT_PARAMS,
        "training": training,
    })


# The shipped model's training (adopted after WP-5): the benchmark plus the easy and
# medium variants of the synthetic dataset, with the threshold chosen on its val split.
DEFAULT_EXTRA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic")
DEFAULT_EXTRA_DIFFICULTIES = "easy,medium"


def add_training_options(parser: argparse.ArgumentParser) -> None:
    """The options that say what a model is trained on; shared with check_changes.py."""
    parser.add_argument("--extra-data", default=DEFAULT_EXTRA_DATA, metavar="DIR",
                        help="Synthetic dataset (scripts/make_synthetic.py) whose train split is added to the "
                             "training data (default: data/synthetic).")
    parser.add_argument("--extra-split", default="train", help="Which split of --extra-data to add (train/val/test/all).")
    parser.add_argument("--extra-difficulties", default=DEFAULT_EXTRA_DIFFICULTIES,
                        help=f"Comma-separated levels of --extra-data to add (default: {DEFAULT_EXTRA_DIFFICULTIES}; "
                             "all for every level).")
    parser.add_argument("--no-extra-data", action="store_true",
                        help="Train on the benchmark drawings only, at the fixed threshold (the model before WP-5).")
    parser.add_argument("--fixed-threshold", action="store_true",
                        help=f"Use the fixed threshold {THRESHOLD:.3f} instead of choosing it on the val split of --extra-data.")
    parser.add_argument("--model", default="scorecard", choices=MODEL_KINDS,
                        help="scorecard (logistic regression, the shipped kind), gbt (gradient-boosted trees) "
                             "or gnn (graph network over the drawing, restoration/gnn.py).")
    parser.add_argument("--gnn-relations", default=None,
                        help="Comma-separated relations the graph model passes messages over "
                             "(default: all of stroke,near,parallel,candidate,symmetric; none for no context).")
    parser.add_argument("--gnn-epochs", type=int, default=None, help="Training epochs of the graph model (default 30).")


def configure_gnn(args: argparse.Namespace) -> None:
    """Set GNN_OPTIONS from --gnn-relations / --gnn-epochs (used by fit(kind="gnn"))."""
    GNN_OPTIONS.clear()
    if getattr(args, "gnn_relations", None):
        from restoration.graph_features import RELATIONS
        # "none": no message passing, the network sees only each candidate's own measurements (ablation).
        relations = () if args.gnn_relations == "none" else tuple(r for r in args.gnn_relations.split(",") if r)
        unknown = set(relations) - set(RELATIONS)
        if unknown:
            raise SystemExit(f"unknown relations {sorted(unknown)}; known: {', '.join(RELATIONS)}")
        GNN_OPTIONS["relations"] = relations
    if getattr(args, "gnn_epochs", None):
        GNN_OPTIONS["epochs"] = int(args.gnn_epochs)


def training_options(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
) -> Tuple[Optional[str], Optional[List[str]], bool]:
    """(extra data folder or None, its difficulty levels or None for all, whether to tune the threshold)."""
    if args.no_extra_data:
        return None, None, False
    if not os.path.exists(os.path.join(args.extra_data, "manifest.json")):
        parser.error(f"no synthetic dataset at {args.extra_data}. Build it with "
                     "python scripts/fetch_quickdraw.py and python scripts/make_synthetic.py, "
                     "or pass --no-extra-data.")
    levels = None if args.extra_difficulties in (None, "", "all") else args.extra_difficulties.split(",")
    return args.extra_data, levels, not args.fixed_threshold


THRESHOLD_GRID = tuple(np.round(np.arange(0.05, 0.901, 0.025), 3))   # wide enough that no model sits on its floor


def choose_threshold(
    images: Sequence[Dict[str, Any]], probabilities: Sequence[np.ndarray],
    reference: Sequence[np.ndarray], workers: int,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """The threshold that repairs most gaps of *images* while drawing no more line off the drawing
    than the *reference* probabilities do at THRESHOLD (ties: the higher threshold).

    Returns (threshold, its totals, the reference totals). Meant for a validation set the
    models were not fitted on.
    """
    ref = sweep(images, reference, (THRESHOLD,), workers)[0]
    totals = sweep(images, probabilities, THRESHOLD_GRID, workers)
    allowed = [(t, row) for t, row in zip(THRESHOLD_GRID, totals) if row["off_drawing_widths"] <= ref["off_drawing_widths"]]
    if not allowed:
        return THRESHOLD, sweep(images, probabilities, (THRESHOLD,), workers)[0], ref
    threshold, row = max(allowed, key=lambda item: (item[1]["repaired"], item[0]))
    return float(threshold), row, ref


def write_json(path: str, document: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def validation_images(
    extra_data: str, difficulties: Optional[Sequence[str]], workers: int, exclude: Sequence[str] = (),
    with_graph: bool = False,
) -> List[Dict[str, Any]]:
    """Labelled candidates of the val split of a synthetic dataset, without the drawings in *exclude*
    (the benchmark's, so that no threshold is tuned on a drawing it is later tested on)."""
    from synthetic_damage import load_dataset

    jobs = [(e["drawing"], f"val_{e['difficulty']}", e["clean"], e["damaged"], e["gapped"], with_graph)
            for e in load_dataset(extra_data, "val")
            if e["drawing"] not in set(exclude) and (not difficulties or e["difficulty"] in difficulties)]
    return [image for image in map_in_processes(labelled_candidates, jobs, workers) if not image.get("skipped")]


def fit_with_threshold(
    train: Sequence[Dict[str, Any]], kind: str, names: Optional[Sequence[str]],
    val: Optional[Sequence[Dict[str, Any]]], workers: int,
) -> Tuple[Dict[str, Dict[str, Any]], float, Dict[str, Any]]:
    """Fit a model; with *val*, choose its threshold there (choose_threshold), against a scorecard
    fitted on the same images minus the synthetic extra data, at the usual threshold."""
    model = fit(train, names, kind)
    if not val:
        return model, THRESHOLD, {}
    reference = fit([image for image in train if not image["kind"].startswith("extra_")], names, "scorecard")
    threshold, row, ref = choose_threshold(val, [predict(model, i) for i in val], [predict(reference, i) for i in val], workers)
    return model, threshold, {
        "threshold_chosen_on": f"{len(val)} validation images",
        "validation": {"repaired": row["repaired"], "gaps": row["gaps"], "off_drawing_widths": round(row["off_drawing_widths"], 1)},
        "validation_reference": {"repaired": ref["repaired"], "off_drawing_widths": round(ref["off_drawing_widths"], 1)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the candidate scorer; report cross-validated results.")
    parser.add_argument("--output", default=MODEL_PATH, help="Where to write the model fitted on all drawings.")
    parser.add_argument("--cv-dir", default=None, help="Also write each fold's held-out model and folds.json here.")
    parser.add_argument("--dataset", default=None, help="File to keep the labelled candidates in between runs.")
    parser.add_argument("--work-dir", default=None, help="Directory for the synthetic damaged images.")
    parser.add_argument("--features", default=None, help="Comma-separated measurements to use (default: all).")
    parser.add_argument("--filter", default=None, help="Only drawings whose name contains this substring.")
    add_training_options(parser)
    parser.add_argument("--tjoins", action="store_true",
                        help="Also fit a T-join scorecard (restoration/tjoins.py): restorations with the model then "
                             "extend free stroke ends to the stroke ahead of them. Not in the shipped model.")
    parser.add_argument("--tjoin-threshold", type=float, default=None,
                        help="Probability a T-join needs (default: the bridges' threshold).")
    parser.add_argument("--include-feedback", default=None, metavar="FILE",
                        help="Also train on user corrections logged by scripts/refine.py (feedback/feedback.jsonl).")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    extra_data, difficulties, tune = training_options(args, parser)
    configure_gnn(args)

    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset, "rb") as f:
            images = pickle.load(f)
        print(f"  Loaded {len(images)} labelled images from {args.dataset}")
    else:
        work_dir = args.work_dir or tempfile.mkdtemp(prefix="train_scoring_")
        os.makedirs(work_dir, exist_ok=True)
        images = build_dataset(work_dir, args.workers, args.filter, extra_data, args.extra_split, difficulties,
                               with_graph=args.model == "gnn")
        if args.dataset:
            with open(args.dataset, "wb") as f:
                pickle.dump(images, f)

    if args.include_feedback:
        from restoration.feedback import feedback_images
        corrections = feedback_images(args.include_feedback)
        print(f"  Added {sum(len(i['correct']) for i in corrections)} user corrections "
              f"from {len(corrections)} drawings ({args.include_feedback})")
        images = list(images) + corrections

    names = args.features.split(",") if args.features else None
    folds = fold_of_drawings(images)
    val = None
    if tune:
        benchmark = {image["drawing"] for image in images if not image["kind"].startswith("extra_")}
        val = validation_images(extra_data, difficulties, args.workers, sorted(benchmark), with_graph=args.model == "gnn")
        print(f"  {len(val)} validation images for the threshold")

    # ---- every candidate's probability from a model fitted without its drawing
    fitted = {
        fold: fit_with_threshold([image for image in images if folds[image["drawing"]] != fold],
                                 args.model, names, val, args.workers)
        for fold in sorted(set(folds.values()))
    }
    fold_models = {fold: model for fold, (model, _, _) in fitted.items()}
    held_out = [predict(fold_models[folds[image["drawing"]]], image) for image in images]
    final_model, final_threshold, threshold_info = fit_with_threshold(images, args.model, names, val, args.workers)

    # ---- report
    sep = "=" * 78
    stats = separation(images, held_out)
    gap_column = FEATURE_NAMES.index("gap_length")
    by_distance = separation(images, [-image["features"][:, gap_column] for image in images])
    print(f"\n{sep}\n  CANDIDATE SCORER, cross-validated by drawing "
          f"({len(folds)} drawings, {len(images)} images, "
          f"{stats['candidates']} candidates, {stats['correct']} correct)\n{sep}")
    print(f"  Chance that a correct candidate outscores a wrong one: {100 * stats['auc']:.1f}%   "
          f"(gap length alone: {100 * by_distance['auc']:.1f}%)")
    print(f"  Stroke ends whose best candidate is a correct one:     {100 * stats['best_candidate_correct']:.1f}%   "
          f"(gap length alone: {100 * by_distance['best_candidate_correct']:.1f}%)")

    print("  Do the probabilities mean what they say?  predicted, candidates, share correct")
    probability = np.concatenate(held_out)
    correct = np.concatenate([image["correct"] for image in images])
    calibration = []
    for low, high in ((0.0, 0.1), (0.1, 0.33), (0.33, 0.5), (0.5, THRESHOLD), (THRESHOLD, 0.8), (0.8, 0.9), (0.9, 1.01)):
        chosen = (probability >= low) & (probability < high)
        share = float(correct[chosen].mean()) if chosen.any() else float("nan")
        calibration.append({"from": round(low, 3), "to": round(min(high, 1.0), 3),
                            "candidates": int(chosen.sum()), "share_correct": round(share, 4)})
        print(f"    {low:4.2f}-{min(high, 1.0):4.2f}   {int(chosen.sum()):6d}   {100 * share:5.1f}%")

    print("  Held-out selection by threshold:  gaps repaired | bridges, unfounded (repair nothing, or leave "
          "the drawing), mostly off the drawing")
    kinds = sorted({image["kind"] for image in images})
    for threshold in REPORTED_THRESHOLDS:
        print(f"    threshold {threshold:.2f}{'  <- in use' if threshold == THRESHOLD else ''}")
        for kind in kinds:
            chosen = [i for i, image in enumerate(images) if image["kind"] == kind]
            total = sweep([images[i] for i in chosen], [held_out[i] for i in chosen], (threshold,), args.workers)[0]
            recall = f"{100 * total['repaired'] / total['gaps']:5.1f}% of {int(total['gaps'])}" if total["gaps"] else "   (no gaps)"
            print(f"      {kind:<11s} {recall:<15s} | {int(total['bridges']):5d} {int(total['unfounded']):5d} {int(total['wrong']):4d}")

    print(f"\n  Model fitted on all drawings; a bridge is drawn above probability {THRESHOLD:.2f} "
          f"({ACCEPTANCE_ODDS:g} to 1 that it is right).")
    if tune:
        print(f"  Threshold chosen on the validation split: {final_threshold:.3f}  {threshold_info}")
    for shape, card in final_model.items():
        if card.get("kind") == "gbt":
            print(f"  Boosted trees for {shape} bridges on: {', '.join(card['names'])}")
            continue
        if card.get("kind") == "gnn":
            print(f"  Graph network over relations: {', '.join(card['scorer'].config.relations)}")
            continue
        print(f"  Scorecard for {shape} bridges, weights on standardized measurements:")
        for name, weight in sorted(zip(card["names"], card["weights"]), key=lambda item: -abs(item[1])):
            scale = f"  [{FEATURE_SCALES[name]}]" if name in FEATURE_SCALES else ""
            print(f"    {name:<16s} {weight:+7.3f}   {FEATURES[name]}{scale}")
        if card["unused"]:
            print(f"    given no weight and left off: {', '.join(card['unused'])}")
    print(sep)

    # ---- output
    training_summary = {
        "drawings": len(folds),
        "images": len(images),
        "candidates": stats["candidates"],
        "correct_candidates": stats["correct"],
        "cross_validated_auc": round(stats["auc"], 4),
        "cross_validated_calibration": calibration,
        "extra_data": {"difficulties": difficulties or "all", "split": args.extra_split} if extra_data else None,
        "threshold_rule": (f"chosen on the synthetic validation split: {threshold_info}" if tune else
                           f"a bridge is drawn when it is at least {ACCEPTANCE_ODDS:g} times as likely to be right as wrong"),
    }
    if args.tjoins:
        tjoin_images = build_tjoin_dataset(work_dir if not args.dataset else tempfile.mkdtemp(prefix="train_tjoins_"),
                                           args.workers, extra_data, difficulties)
        final_model["tjoin"] = fit_tjoin(tjoin_images)
        card = final_model["tjoin"]
        print(f"  T-join scorecard on {sum(len(i['correct']) for i in tjoin_images)} candidates: "
              + ", ".join(f"{n} {w:+.2f}" for n, w in zip(card["names"], card["weights"])))
    write_model(args.output, final_model, final_threshold, training_summary)
    if args.tjoins and args.tjoin_threshold is not None:
        with open(args.output) as f:
            document = json.load(f)
        document["tjoin"]["threshold"] = float(args.tjoin_threshold)
        write_json(args.output, document)
    print(f"  Model saved -> {args.output}")
    if args.cv_dir:
        for fold, (model, threshold, info) in fitted.items():
            write_model(os.path.join(args.cv_dir, f"fold_{fold}.json"), model, threshold,
                        {"held_out_fold": fold, **info})
        write_json(os.path.join(args.cv_dir, "folds.json"),
                   {drawing: f"fold_{fold}.json" for drawing, fold in sorted(folds.items())})
        print(f"  Held-out models saved -> {args.cv_dir}")


if __name__ == "__main__":
    main()
