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


def labelled_candidates(drawing: str, kind: str, original_path: str, image_path: str) -> Dict[str, Any]:
    """Candidates of one image with their measurements and what the original says about them."""
    original, damaged, agreement = load_pair(original_path, image_path)
    if original.shape != damaged.shape or agreement < 0.98:
        return {"drawing": drawing, "kind": kind, "skipped": True}
    gt = build_ground_truth(original, damaged)

    extraction = extract_paths(image_path)
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
    }


def build_dataset(work_dir: str, workers: int, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Labelled candidates of every drawing: its provided damage, synthetic damage, and no damage."""
    os.makedirs(work_dir, exist_ok=True)
    jobs: List[Tuple[str, str, str, str]] = []
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
    bounds = [allowed[FEATURE_DIRECTIONS.get(name, 0)] for name in names] + [(None, None)]
    result = minimize(loss, np.zeros(len(names) + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 5000})
    # A measurement the fit gives no weight (it would have counted the wrong way) is left off the scorecard.
    used = np.flatnonzero(result.x[:-1] != 0.0)
    return {"names": [names[i] for i in used], "columns": [columns[i] for i in used],
            "mean": mean[used], "scale": scale[used],
            "weights": result.x[:-1][used], "intercept": float(result.x[-1]),
            "unused": [name for i, name in enumerate(names) if i not in used]}


def fit(images: Sequence[Dict[str, Any]], names: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """A scorecard for each bridge shape, on the measurements of that shape (only those in *names*, if given)."""
    return {
        shape: fit_scorecard(images, shape, [n for n in shape_names if names is None or n in names])
        for shape, shape_names in SHAPE_FEATURES.items()
    }


def predict(model: Dict[str, Dict[str, Any]], image: Dict[str, Any]) -> np.ndarray:
    """Probability of every candidate of an image, each from the scorecard of its shape."""
    probability = np.zeros(len(image["shape"]))
    for shape, card in model.items():
        rows = image["shape"] == shape
        standardized = (image["features"][rows][:, card["columns"]] - card["mean"]) / card["scale"]
        probability[rows] = expit(card["intercept"] + standardized @ card["weights"])
    return probability


def fold_of_drawings(images: Sequence[Dict[str, Any]], folds: int = FOLDS) -> Dict[str, int]:
    """Assign every drawing to a fold, in a fixed shuffled order."""
    drawings = sorted({image["drawing"] for image in images})
    order = np.random.RandomState(0).permutation(len(drawings))
    return {drawings[index]: position % folds for position, index in enumerate(order)}


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
            for shape, card in model.items()
        },
        "training": training,
    }


def write_json(path: str, document: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(document, f, indent=2)
        f.write("\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the candidate scorer; report cross-validated results.")
    parser.add_argument("--output", default=MODEL_PATH, help="Where to write the model fitted on all drawings.")
    parser.add_argument("--cv-dir", default=None, help="Also write each fold's held-out model and folds.json here.")
    parser.add_argument("--dataset", default=None, help="File to keep the labelled candidates in between runs.")
    parser.add_argument("--work-dir", default=None, help="Directory for the synthetic damaged images.")
    parser.add_argument("--features", default=None, help="Comma-separated measurements to use (default: all).")
    parser.add_argument("--filter", default=None, help="Only drawings whose name contains this substring.")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()

    if args.dataset and os.path.exists(args.dataset):
        with open(args.dataset, "rb") as f:
            images = pickle.load(f)
        print(f"  Loaded {len(images)} labelled images from {args.dataset}")
    else:
        work_dir = args.work_dir or tempfile.mkdtemp(prefix="train_scoring_")
        os.makedirs(work_dir, exist_ok=True)
        images = build_dataset(work_dir, args.workers, args.filter)
        if args.dataset:
            with open(args.dataset, "wb") as f:
                pickle.dump(images, f)

    names = args.features.split(",") if args.features else None
    folds = fold_of_drawings(images)

    # ---- every candidate's probability from a model fitted without its drawing
    fold_models = {
        fold: fit([image for image in images if folds[image["drawing"]] != fold], names)
        for fold in sorted(set(folds.values()))
    }
    held_out = [predict(fold_models[folds[image["drawing"]]], image) for image in images]
    final_model = fit(images, names)

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
    for shape, card in final_model.items():
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
        "threshold_rule": f"a bridge is drawn when it is at least {ACCEPTANCE_ODDS:g} times as likely to be right as wrong",
    }
    write_json(args.output, model_document(final_model, THRESHOLD, training_summary))
    print(f"  Model saved -> {args.output}")
    if args.cv_dir:
        for fold, model in fold_models.items():
            write_json(os.path.join(args.cv_dir, f"fold_{fold}.json"),
                       model_document(model, THRESHOLD, {"held_out_fold": fold}))
        write_json(os.path.join(args.cv_dir, "folds.json"),
                   {drawing: f"fold_{fold}.json" for drawing, fold in sorted(folds.items())})
        print(f"  Held-out models saved -> {args.cv_dir}")


if __name__ == "__main__":
    main()
