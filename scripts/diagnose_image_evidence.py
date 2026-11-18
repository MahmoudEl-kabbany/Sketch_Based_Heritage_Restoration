"""Is there image evidence inside gaps? Grey-level statistics along every candidate bridge.

For the 32 real pairs and the synthetic val split (easy + medium), it measures four
statistics of the damaged grey image along each labelled candidate bridge, relative to the
paper around it, and reports how well each separates correct from wrong candidates, alone and
among the candidates the shipped scorecard is unsure about (probability 0.2-0.8).
See docs/experiments/image-evidence.md. Run from the repository root:
    python scripts/diagnose_image_evidence.py
"""
import os
import sys

import cv2
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.getcwd())
import train_scoring as ts  # noqa: E402
from evaluate_restoration import discover_pairs  # noqa: E402
from restoration.pipeline import map_in_processes  # noqa: E402
from restoration.scoring import load_model  # noqa: E402
from synthetic_damage import load_dataset  # noqa: E402

STATS = ("gap_darkness", "gap_min_darkness", "gap_texture", "gap_bright_share")


def patch_stats(gray, bridge, lw):
    h, w = gray.shape
    pts = bridge
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    inner = pts[(along > 1.5 * lw) & (along < along[-1] - 1.5 * lw)]
    if len(inner) < 2:
        return [np.nan] * len(STATS)
    xy = np.clip(np.round(inner).astype(int), [0, 0], [w - 1, h - 1])
    vals = gray[xy[:, 1], xy[:, 0]].astype(float)
    x0, y0 = np.maximum(inner.min(axis=0) - 4 * lw, 0).astype(int)
    x1, y1 = np.minimum(inner.max(axis=0) + 4 * lw, [w - 1, h - 1]).astype(int)
    window = gray[y0:y1 + 1, x0:x1 + 1].astype(float)
    paper = np.percentile(window, 90)
    light = window[window >= np.percentile(window, 50)]
    paper_sd = float(light.std()) if light.size else 1.0
    return [
        (paper - np.median(vals)) / max(paper, 1.0),
        (paper - vals.min()) / max(paper, 1.0),
        vals.std() / max(paper_sd, 1e-3),
        float((vals > np.percentile(window, 95)).mean()),
    ]


def one(drawing, kind, original, damaged, reference=None):
    image = ts.labelled_candidates(drawing, kind, original, damaged, reference)
    if image.get("skipped"):
        return None
    gray = cv2.imread(damaged, cv2.IMREAD_GRAYSCALE)
    from restoration.render_restored_sketch import estimate_line_width
    lw = float(estimate_line_width(cv2.imread(original, cv2.IMREAD_GRAYSCALE)))
    image["patch"] = np.array([patch_stats(gray, b.astype(float), lw) for b in image["bridges"]]).reshape(-1, len(STATS))
    return image


def shipped_probability(model, image):
    from restoration.scoring import FEATURE_NAMES
    out = np.zeros(len(image["shape"]))
    for shape, card in model.scorecards.items():
        rows = image["shape"] == shape
        cols = [FEATURE_NAMES.index(n) for n in card.names]
        z = card.intercept + card.contributions(image["features"][rows][:, cols]).sum(axis=1)
        out[rows] = 1.0 / (1.0 + np.exp(-z))
    return out


def auc(score, y):
    ok = np.isfinite(score)
    s, y = score[ok], y[ok]
    p, n = int(y.sum()), int(len(y) - y.sum())
    if not p or not n:
        return float("nan")
    return float((rankdata(s)[y].sum() - p * (p + 1) / 2) / (p * n))


if __name__ == "__main__":
    model = load_model()
    sets = {
        "real pairs (provided damage)": [(n, "provided", o, d) for n, o, d in discover_pairs()],
        "synthetic val easy+medium": [(e["drawing"], "val", e["clean"], e["damaged"], e["gapped"])
                                      for e in load_dataset("data/synthetic", "val") if e["difficulty"] in ("easy", "medium")],
    }
    for label, jobs in sets.items():
        images = [i for i in map_in_processes(one, jobs, 8) if i]
        y = np.concatenate([i["correct"] for i in images])
        patch = np.vstack([i["patch"] for i in images])
        print(f"\n{label}: {len(images)} images, {len(y)} candidates, {int(y.sum())} correct")
        for k, name in enumerate(STATS):
            a = auc(patch[:, k], y)
            print(f"  {name:18s} AUC alone {a:.3f}   (0.5 = no signal; <0.5 = signal the other way)   "
                  f"median correct {np.nanmedian(patch[y, k]):.3f} vs wrong {np.nanmedian(patch[~y, k]):.3f}")
        # Within candidates the scorecard is unsure about (probability 0.2-0.8): does the patch still separate?
        probs = np.concatenate([shipped_probability(model, i) for i in images])
        unsure = (probs > 0.2) & (probs < 0.8)
        print(f"  among {int(unsure.sum())} candidates the scorecard puts at 0.2-0.8 ({int(y[unsure].sum())} correct):")
        for k, name in enumerate(STATS):
            print(f"    {name:18s} AUC {auc(patch[unsure, k], y[unsure]):.3f}")
