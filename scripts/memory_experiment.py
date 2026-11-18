"""WP-8 experiment: does a retrieval memory of solved gaps improve the scorer?

A memory bank holds labelled candidate bridges (their scorecard-scale
measurements and whether they were right). For a new candidate, the
distance-weighted share of correct labels among its K nearest neighbours of
the same bridge shape is its "memory vote". The vote's log-odds is stacked onto
the scorecard's log-odds with one fitted weight (and intercept), so the
scorecard itself is unchanged and the memory only corrects it.

Everything is held out by drawing, with the same folds as check_changes.py:
  * per fold, the scorecard and its threshold are fitted as for the shipped model
    (train_scoring.fit_with_threshold, threshold chosen on the synthetic val split);
  * training candidates get their memory vote from a bank without their own drawing
    (the training drawings are split into MEMORY_GROUPS groups; each group is
    voted on by the other groups);
  * the held-out drawings are never in the bank.
Results are on the held-out "provided" images (the real damage), by selecting
with the ASP solver at the fold's threshold (train_scoring.sweep).

Growth: with the stacking weights fixed, the bank of each fold is limited to
the synthetic drawings plus N real benchmark drawings, for growing N, to see
whether adding solved real drawings helps without retraining.

Usage (needs the label caches written by check_changes.py --dataset-cache DIR):
    python scripts/memory_experiment.py --cache DIR
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_scoring as ts  # noqa: E402
from restoration.scoring import FEATURE_NAMES, SHAPE_FEATURES  # noqa: E402

K = 15
MEMORY_GROUPS = 5


def shape_matrix(image: Dict[str, Any], shape: str) -> np.ndarray:
    columns = [FEATURE_NAMES.index(n) for n in SHAPE_FEATURES[shape]]
    return image["features"][image["shape"] == shape][:, columns]


class Bank:
    """Labelled candidates of some images, one nearest-neighbour index per shape."""

    def __init__(self, images: Sequence[Dict[str, Any]], scale: Dict[str, tuple]):
        self.scale = scale
        self.index, self.labels = {}, {}
        for shape in SHAPE_FEATURES:
            x = [shape_matrix(i, shape) for i in images]
            y = [i["correct"][i["shape"] == shape] for i in images]
            x = np.vstack(x) if x else np.zeros((0, len(SHAPE_FEATURES[shape])))
            if len(x) >= K:
                self.index[shape] = NearestNeighbors(n_neighbors=K).fit(self._z(x, shape))
                self.labels[shape] = np.concatenate(y).astype(float)

    def _z(self, x: np.ndarray, shape: str) -> np.ndarray:
        mean, scale = self.scale[shape]
        return np.nan_to_num((x - mean) / scale)

    def vote(self, image: Dict[str, Any]) -> np.ndarray:
        """Distance-weighted share of correct neighbours, per candidate of *image* (0.5 without a bank)."""
        out = np.full(len(image["shape"]), 0.5)
        for shape, nn in self.index.items():
            rows = image["shape"] == shape
            if not rows.any():
                continue
            d, j = nn.kneighbors(self._z(shape_matrix(image, shape), shape))
            w = 1.0 / (d + 1e-3)
            out[rows] = (w * self.labels[shape][j]).sum(axis=1) / w.sum(axis=1)
        return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.01, 0.99)   # a vote is an estimate from K neighbours; keep it finite
    return np.log(p / (1 - p))


def fit_stack(base_logodds: np.ndarray, vote: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(a, b): log-odds = base + a * logit(vote) + b, by maximum likelihood (a >= 0)."""
    v = logit(vote)

    def loss(p):
        m = base_logodds + p[0] * v + p[1]
        value = np.logaddexp(0.0, np.where(y > 0.5, -m, m)).sum()
        e = expit(m) - y
        return float(value), np.array([(e * v).sum(), e.sum()])

    return minimize(loss, np.zeros(2), jac=True, method="L-BFGS-B", bounds=[(0.0, None), (None, None)]).x


def base_logodds(model, image) -> np.ndarray:
    p = np.clip(ts.predict(model, image), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def memory_predict(model, stack, bank: Bank, image) -> np.ndarray:
    return expit(base_logodds(model, image) + stack[0] * logit(bank.vote(image)) + stack[1])


def scale_of(images) -> Dict[str, tuple]:
    out = {}
    for shape in SHAPE_FEATURES:
        x = np.vstack([shape_matrix(i, shape) for i in images])
        mean, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        out[shape] = (np.nan_to_num(mean), np.where(sd > 1e-9, sd, 1.0))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="check_changes.py --dataset-cache folder")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with open(os.path.join(args.cache, "train_extra-easy-medium.pkl"), "rb") as f:
        images = pickle.load(f)
    with open(os.path.join(args.cache, "val_extra-easy-medium.pkl"), "rb") as f:
        val = pickle.load(f)
    folds = ts.fold_of_drawings(images)
    benchmark = sorted({i["drawing"] for i in images if not i["kind"].startswith("extra_")})
    rng = np.random.default_rng(0)

    totals = {"base": [], "memory": []}
    growth: Dict[int, List[Dict[str, float]]] = {}
    for fold in sorted(set(folds[d] for d in benchmark)):
        train = [i for i in images if folds[i["drawing"]] != fold]
        test = [i for i in images if folds[i["drawing"]] == fold and i["kind"] == "provided"]
        model, threshold, _ = ts.fit_with_threshold(train, "scorecard", None, val, args.workers)
        scale = scale_of(train)

        # memory votes for the training images, without their own drawing
        drawings = sorted({i["drawing"] for i in train})
        group = {d: k % MEMORY_GROUPS for k, d in enumerate(rng.permutation(drawings))}
        votes = {}
        for g in range(MEMORY_GROUPS):
            bank_g = Bank([i for i in train if group[i["drawing"]] != g], scale)
            for idx, i in enumerate(train):
                if group[i["drawing"]] == g:
                    votes[idx] = bank_g.vote(i)
        y = np.concatenate([i["correct"] for i in train]).astype(float)
        base = np.concatenate([base_logodds(model, i) for i in train])
        vote = np.concatenate([votes[idx] for idx in range(len(train))])
        stack = fit_stack(base, vote, y)

        bank = Bank(train, scale)
        val_memory = [memory_predict(model, stack, bank, i) for i in val]
        reference = [ts.predict(ts.fit([i for i in train if not i["kind"].startswith("extra_")], None, "scorecard"), i) for i in val]
        mem_threshold, _, _ = ts.choose_threshold(val, val_memory, reference, args.workers)

        base_row = ts.sweep(test, [ts.predict(model, i) for i in test], (threshold,), args.workers)[0]
        mem_row = ts.sweep(test, [memory_predict(model, stack, bank, i) for i in test], (mem_threshold,), args.workers)[0]
        totals["base"].append(base_row)
        totals["memory"].append(mem_row)
        print(f"fold {fold}: stack weight {stack[0]:.2f}, shift {stack[1]:+.2f}; threshold base {threshold:.3f} memory {mem_threshold:.3f}; "
              f"repaired {base_row['repaired']:.0f} -> {mem_row['repaired']:.0f} of {base_row['gaps']:.0f}, "
              f"off-drawing {base_row['off_drawing_widths']:.1f} -> {mem_row['off_drawing_widths']:.1f}", flush=True)

        # growth: synthetic-only bank plus N real benchmark drawings from the training folds
        real_train = [d for d in benchmark if folds[d] != fold]
        order = list(rng.permutation(real_train))
        for n in (0, 6, 12, len(order)):
            keep = set(order[:n])
            bank_n = Bank([i for i in train if i["kind"].startswith("extra_") or i["drawing"] in keep], scale)
            row = ts.sweep(test, [memory_predict(model, stack, bank_n, i) for i in test], (mem_threshold,), args.workers)[0]
            growth.setdefault(n if n < len(order) else -1, []).append(row)

    def total(rows):
        return {k: sum(r[k] for r in rows) for k in ("repaired", "gaps", "off_drawing_widths", "wrong", "bridges")}

    print("\nHeld-out provided images (32 real pairs), candidate-level selection:")
    for name in ("base", "memory"):
        t = total(totals[name])
        print(f"  {name:7s} recall {100 * t['repaired'] / t['gaps']:.1f}%  off-drawing {t['off_drawing_widths']:.1f}  "
              f"wrong {t['wrong']:.0f}  bridges {t['bridges']:.0f}")
    print("Growth (memory model, stack fixed; bank = synthetic + N real drawings):")
    for n in sorted(growth, key=lambda v: (v < 0, v)):
        t = total(growth[n])
        label = "all training" if n < 0 else str(n)
        print(f"  N = {label:13s} recall {100 * t['repaired'] / t['gaps']:.1f}%  off-drawing {t['off_drawing_widths']:.1f}  wrong {t['wrong']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
