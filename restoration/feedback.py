"""User corrections as training data.

Every forbid, force or hint made in scripts/refine.py is appended to a JSON
Lines file (feedback/feedback.jsonl by default, git-ignored) together with the
measurements of the affected candidates, on the scale the scorer is fitted on:

    forbid  every candidate between the two ends     label 0
    force   the bridge drawn between the two ends    label 1
    hint    the bridge drawn between the two ends    label 1

train_scoring.py --include-feedback FILE adds these rows to the training data
as one pseudo-image per corrected drawing (kind "extra_feedback"). A later
action on the same pair of ends in the same image replaces an earlier one.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from restoration.candidates import ConnectionCandidate
from restoration.extraction import ExtractionResult
from restoration.scoring import FEATURE_NAMES, candidate_features

DEFAULT_FEEDBACK_PATH = os.path.join("feedback", "feedback.jsonl")
LABELS = {"forbid": 0, "force": 1, "hint": 1}


def _pair_key(candidate: ConnectionCandidate) -> frozenset:
    return frozenset((candidate.ep_a.endpoint_id, candidate.ep_b.endpoint_id))


def feedback_rows(
    image_path: str,
    action: str,
    endpoint_pair: Sequence[int],
    candidates: Sequence[ConnectionCandidate],
    accepted: Sequence[ConnectionCandidate],
    extraction: ExtractionResult,
) -> List[Dict[str, Any]]:
    """Labelled rows for one user action on the pair of ends *endpoint_pair*."""
    pair = frozenset(endpoint_pair)
    if action == "forbid":
        chosen = [c for c in candidates if _pair_key(c) == pair]
    else:   # force / hint: the bridge that was actually drawn for the pair
        chosen = [c for c in accepted if _pair_key(c) == pair]
    if not chosen:
        return []
    features = candidate_features(list(chosen), extraction)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    return [{
        "time": stamp,
        "image": os.path.basename(image_path),
        "action": action,
        "label": LABELS[action],
        "shape": c.shape,
        "endpoints": [np.round(c.ep_a.position, 1).tolist(), np.round(c.ep_b.position, 1).tolist()],
        "probability": round(float(c.probability), 4),
        "features": {name: (None if not np.isfinite(v) else float(v)) for name, v in zip(FEATURE_NAMES, row)},
    } for c, row in zip(chosen, features)]


def append_feedback(rows: Iterable[Dict[str, Any]], path: str = DEFAULT_FEEDBACK_PATH) -> int:
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def read_feedback(path: str) -> List[Dict[str, Any]]:
    """The rows of a feedback file, a later action on the same pair and shape replacing an earlier one."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ends = tuple(sorted(tuple(round(v) for v in p) for p in row["endpoints"]))
            latest[(row["image"], ends, row["shape"])] = row
    return list(latest.values())


def feedback_images(path: str) -> List[Dict[str, Any]]:
    """Feedback as train_scoring images: one per corrected drawing, kind "extra_feedback"."""
    by_image: Dict[str, List[Dict[str, Any]]] = {}
    for row in read_feedback(path):
        by_image.setdefault(row["image"], []).append(row)
    images = []
    for image, rows in sorted(by_image.items()):
        n = len(rows)
        features = np.array([[np.nan if row["features"].get(name) is None else row["features"][name]
                              for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
        correct = np.array([row["label"] == 1 for row in rows], dtype=bool)
        images.append({
            "drawing": f"feedback:{os.path.splitext(image)[0]}",
            "kind": "extra_feedback",
            "features": features,
            "shape": np.array([row["shape"] for row in rows], dtype=object),
            "correct": correct,
            # Fields the selection report expects; feedback carries no gap ground truth.
            "unfounded": ~correct,
            "on_drawing": correct.astype(float),
            "length_widths": np.ones(n),
            "gaps": [[] for _ in range(n)],
            "endpoints": np.arange(2 * n, dtype=int).reshape(n, 2),
            "bridges": [np.zeros((2, 2), np.float32) for _ in range(n)],
            "gap_count": 0,
        })
    return images
