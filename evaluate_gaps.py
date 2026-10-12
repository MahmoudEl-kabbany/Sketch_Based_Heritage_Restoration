"""Gap-level evaluation of the restoration pipeline against ground truth.

PSNR and SSIM barely move when a sketch gains or loses a stroke. This script
asks the questions that matter for restoration instead:

  * Gap recall      - which of the truly missing stroke pieces were redrawn?
  * Wrong strokes   - how much line was drawn where the original has none?
  * Harm            - what does the pipeline draw on an undamaged drawing?
  * Stage budget    - for each gap, how far did the pipeline get?

A true gap is a set of stroke pixels present in the original image and missing
in the damaged one.

Usage:
    python evaluate_gaps.py                         # all matched pairs
    python evaluate_gaps.py --filter ankh --limit 5
    python evaluate_gaps.py --output-dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from evaluate_restoration import OUTPUT_DIR, discover_pairs
from restoration.candidates import generate_candidates
from restoration.extraction import extract_paths
from restoration.pipeline import _iter_changes, default_workers, map_in_processes, restore
from restoration.render_restored_sketch import estimate_line_width

# A gap counts as covered when this share of its centre line lies within the
# tolerance of a drawn stroke; the tolerance is this many line widths.
COVERAGE_THRESHOLD = 0.6
TOLERANCE_LINE_WIDTHS = 1.0

STAGES = (
    "repaired",                  # a restoration stroke covers the gap
    "bridged_by_extraction",     # the extracted path already ran across it
    "candidate_not_selected",    # a covering candidate existed but was not drawn
    "no_candidate_generated",    # endpoints were found, no candidate covers the gap
    "endpoints_missing",         # fewer than two endpoints next to the gap
)


# ═══════════════════════════════════════════════════════════════════════════
# Ground truth
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GroundTruth:
    """What is missing from a damaged image, derived from its original."""

    shape: Tuple[int, int]
    line_width: float
    tolerance: int                 # pixels: "on the stroke" radius
    original_strokes: np.ndarray   # bool mask of the original's strokes
    gap_mask: np.ndarray           # bool mask of all true gaps
    gap_labels: np.ndarray         # int32 image: gap id (1..n) on gap pixels
    center_labels: np.ndarray      # int32 image: gap id (1..n) on each gap's centre line
    center_counts: np.ndarray      # centre-line pixels per gap id (index 0 unused)
    gaps: List[Dict[str, Any]]     # id, stubs, length_px, length_widths
    dist_to_original: np.ndarray   # distance of every pixel to the original's strokes

    @property
    def stroke_thickness(self) -> int:
        """Thickness that makes a drawn centre line cover its tolerance band."""
        return 2 * self.tolerance + 1


def stroke_mask(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0


def _place(image: np.ndarray, shape: Tuple[int, int], dx: int, dy: int, fill: int) -> np.ndarray:
    """Copy *image* into a canvas of *shape*, shifted by (dx, dy); uncovered pixels get *fill*."""
    canvas = np.full(shape, fill, dtype=image.dtype)
    h, w = image.shape
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + w, shape[1]), min(dy + h, shape[0])
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = image[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return canvas


def load_pair(original_path: str, damaged_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Read a pair as grayscale, with the original brought into the damaged image's frame.

    Returns (original, damaged, agreement), where agreement is the share of the
    damaged image's stroke pixels that lie on the original's strokes. Some
    pairs are offset by a pixel or differ in height by a few rows; the original
    is shifted to the offset that maximizes the agreement.
    """
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    damaged = cv2.imread(damaged_path, cv2.IMREAD_GRAYSCALE)
    if original is None or damaged is None:
        raise FileNotFoundError(f"Cannot read pair: {original_path} / {damaged_path}")

    damaged_strokes = stroke_mask(damaged)
    original_strokes = stroke_mask(original).astype(np.uint8)
    total = max(int(damaged_strokes.sum()), 1)

    def agreement(dx: int, dy: int) -> float:
        placed = _place(original_strokes, damaged.shape, dx, dy, 0) > 0
        return float((damaged_strokes & placed).sum() / total)

    best = (agreement(0, 0), 0, 0) if original.shape == damaged.shape else (-1.0, 0, 0)
    if best[0] < 0.999:
        slack = 2
        dy_range = sorted((0, damaged.shape[0] - original.shape[0]))
        dx_range = sorted((0, damaged.shape[1] - original.shape[1]))
        for dy in range(dy_range[0] - slack, dy_range[1] + slack + 1):
            for dx in range(dx_range[0] - slack, dx_range[1] + slack + 1):
                best = max(best, (agreement(dx, dy), dx, dy))
    score, dx, dy = best
    return _place(original, damaged.shape, dx, dy, 255), damaged, score


def build_ground_truth(
    original: np.ndarray,
    damaged: np.ndarray,
    tolerance_line_widths: float = TOLERANCE_LINE_WIDTHS,
) -> GroundTruth:
    """Find the true gaps of a damaged image whose original has the same frame."""
    shape = original.shape
    line_width = float(estimate_line_width(original))
    tolerance = int(round(max(3.0, line_width * tolerance_line_widths)))
    original_strokes, damaged_strokes = stroke_mask(original), stroke_mask(damaged)

    missing = (original_strokes & ~damaged_strokes).astype(np.uint8)
    missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop 1 px rims
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, 8)

    skeleton = skeletonize(original_strokes)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    min_area = max(8.0, 0.3 * line_width * line_width)

    gap_labels = np.zeros(shape, dtype=np.int32)
    center_labels = np.zeros(shape, dtype=np.int32)
    gaps: List[Dict[str, Any]] = []
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < min_area:
            continue
        region = labels == component
        # Stroke stubs entering the gap: pieces of surviving stroke on its rim.
        rim = (cv2.dilate(region.astype(np.uint8), ring_kernel) > 0) & (missing == 0) & damaged_strokes
        stubs = cv2.connectedComponents(rim.astype(np.uint8), 8)[0] - 1
        center = skeleton & region
        if center.sum() < 2:
            center = region
        gap_id = len(gaps) + 1
        gap_labels[region] = gap_id
        center_labels[center] = gap_id
        ys, xs = np.nonzero(center)
        length = float(max(np.ptp(xs), np.ptp(ys)) + 1)
        gaps.append({
            "id": gap_id,
            "stubs": int(stubs),
            "length_px": length,
            "length_widths": length / line_width,
        })

    return GroundTruth(
        shape=shape,
        line_width=line_width,
        tolerance=tolerance,
        original_strokes=original_strokes,
        gap_mask=gap_labels > 0,
        gap_labels=gap_labels,
        center_labels=center_labels,
        center_counts=np.bincount(center_labels.ravel(), minlength=len(gaps) + 1).astype(float),
        gaps=gaps,
        dist_to_original=cv2.distanceTransform((~original_strokes).astype(np.uint8), cv2.DIST_L2, 5),
    )


def _rasterize(shape: Tuple[int, int], polylines: List[np.ndarray], thickness: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for points in polylines:
        if len(points) >= 2:
            pts = np.round(points).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], False, 255, thickness, lineType=cv2.LINE_8)
    return mask > 0


def gap_coverage(gt: GroundTruth, polylines: List[np.ndarray]) -> np.ndarray:
    """Share of each gap's centre line that lies within tolerance of the polylines."""
    if not gt.gaps:
        return np.zeros(0)
    polylines = [p for p in polylines if len(p) >= 2]
    if not polylines:
        return np.zeros(len(gt.gaps))

    # Work in the bounding box of the polylines only.
    points = np.vstack(polylines)
    margin = gt.stroke_thickness
    x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int) - margin, 0)
    x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + margin + 1, [gt.shape[1], gt.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(gt.gaps))
    drawn = _rasterize((y1 - y0, x1 - x0), [p - [x0, y0] for p in polylines], gt.stroke_thickness)
    hits = np.bincount(gt.center_labels[y0:y1, x0:x1][drawn], minlength=len(gt.gaps) + 1).astype(float)
    return hits[1:] / np.maximum(gt.center_counts[1:], 1.0)


def on_drawing_fraction(gt: GroundTruth, points: np.ndarray) -> float:
    """Share of a drawn stroke's samples that lie on the original drawing."""
    ij = np.clip(np.round(points).astype(int), 0, [gt.shape[1] - 1, gt.shape[0] - 1])
    return float(np.mean(gt.dist_to_original[ij[:, 1], ij[:, 0]] <= gt.tolerance))


def endpoints_near_gaps(gt: GroundTruth, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For endpoint positions: (how many lie next to each gap, which ones lie next to any gap)."""
    per_gap = np.zeros(len(gt.gaps), dtype=int)
    at_gap = np.zeros(len(positions), dtype=bool)
    if not gt.gaps or len(positions) == 0:
        return per_gap, at_gap
    ys, xs = np.nonzero(gt.gap_mask)
    distance, nearest = cKDTree(np.column_stack([xs, ys])).query(positions)
    at_gap = distance <= gt.tolerance + 6
    for j in nearest[at_gap]:
        per_gap[gt.gap_labels[ys[j], xs[j]] - 1] += 1
    return per_gap, at_gap


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


# ═══════════════════════════════════════════════════════════════════════════
# One image pair
# ═══════════════════════════════════════════════════════════════════════════

def _stroke_rows(gt: GroundTruth, restored_paths) -> List[Dict[str, Any]]:
    """One row per restoration stroke: how much of it lies on the original drawing."""
    rows = []
    for source, _, points in _iter_changes(restored_paths):
        on_drawing = on_drawing_fraction(gt, points)
        length = _polyline_length(points)
        rows.append({
            "source": source,
            "on_drawing": on_drawing,
            "length_widths": length / gt.line_width,
            "off_drawing_widths": length * (1.0 - on_drawing) / gt.line_width,
        })
    return rows


def evaluate_pair(
    name: str,
    original_path: str,
    damaged_path: str,
    output_dir: str,
    coverage_threshold: float = COVERAGE_THRESHOLD,
    tolerance_line_widths: float = TOLERANCE_LINE_WIDTHS,
) -> Dict[str, Any]:
    """Run the pipeline on one damaged image and on its original; compare with ground truth."""
    original, damaged, agreement = load_pair(original_path, damaged_path)
    if original.shape != damaged.shape or agreement < 0.98:
        return {"name": name, "skipped": f"original and damaged image do not line up (agreement {agreement:.2f})"}
    gt = build_ground_truth(original, damaged, tolerance_line_widths)

    # ---- damaged image
    result = restore(damaged_path, output_dir=os.path.join(output_dir, "reports"), save_images=False)
    strokes = [points for _, _, points in _iter_changes(result.restored_paths)]
    restored = gap_coverage(gt, strokes)
    any_path = gap_coverage(gt, [p.sample(30) for p in result.restored_paths])

    extraction = extract_paths(damaged_path)
    candidates = generate_candidates(extraction)
    best_candidate = np.zeros(len(gt.gaps))
    for candidate in candidates:
        best_candidate = np.maximum(best_candidate, gap_coverage(gt, [candidate.bridge_points]))
    positions = np.array([e.position for e in extraction.endpoints]).reshape(-1, 2)

    # Share of the damaged image's own strokes that no extracted path represents.
    damaged_skeleton = skeletonize(stroke_mask(damaged))
    represented = _rasterize(gt.shape, [p.sample(30) for p in extraction.paths], gt.stroke_thickness)
    strokes_missed = float(1.0 - represented[damaged_skeleton].mean()) if damaged_skeleton.any() else 0.0
    nearby, at_gap = endpoints_near_gaps(gt, positions)

    for i, gap in enumerate(gt.gaps):
        if restored[i] >= coverage_threshold:
            stage = "repaired"
        elif any_path[i] >= coverage_threshold:
            stage = "bridged_by_extraction"
        elif best_candidate[i] >= coverage_threshold:
            stage = "candidate_not_selected"
        elif nearby[i] >= 2:
            stage = "no_candidate_generated"
        else:
            stage = "endpoints_missing"
        gap.update(stage=stage, coverage=float(restored[i]), endpoints=int(nearby[i]))

    # ---- the undamaged original: everything drawn there is unnecessary
    clean = restore(original_path, output_dir=os.path.join(output_dir, "reports_undamaged"), save_images=False)
    clean_original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    clean_gt = build_ground_truth(clean_original, clean_original, tolerance_line_widths)
    clean_extraction_endpoints = clean.report["analysis"]["extraction"]["endpoints"]

    return {
        "name": name,
        "line_width": gt.line_width,
        "gaps": gt.gaps,
        "strokes": _stroke_rows(gt, result.restored_paths),
        "endpoints": int(len(positions)),
        "endpoints_at_gaps": int(at_gap.sum()),
        "candidates": len(candidates),
        "strokes_missed_by_extraction": strokes_missed,
        "undamaged_strokes": _stroke_rows(clean_gt, clean.restored_paths),
        "undamaged_endpoints": int(clean_extraction_endpoints),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation and output
# ═══════════════════════════════════════════════════════════════════════════

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dataset-level numbers from the per-image rows."""
    rows = [r for r in rows if "skipped" not in r]
    gaps = [g for r in rows for g in r["gaps"]]
    strokes = [s for r in rows for s in r["strokes"]]
    undamaged = [s for r in rows for s in r["undamaged_strokes"]]

    def recall(selection: List[Dict[str, Any]]) -> Optional[float]:
        return sum(g["stage"] == "repaired" for g in selection) / len(selection) if selection else None

    def stroke_stats(selection: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "strokes": len(selection),
            "wrong_strokes": sum(s["on_drawing"] < 0.5 for s in selection),
            "partly_off_strokes": sum(0.5 <= s["on_drawing"] < 0.8 for s in selection),
            "off_drawing_widths": round(sum(s["off_drawing_widths"] for s in selection), 1),
            "drawn_widths": round(sum(s["length_widths"] for s in selection), 1),
        }

    length_bins = (("under 2 widths", 0, 2), ("2-5 widths", 2, 5), ("5-10 widths", 5, 10), ("over 10 widths", 10, float("inf")))
    stage_counts = Counter(g["stage"] for g in gaps)
    return {
        "images": len(rows),
        "gaps": len(gaps),
        "gap_recall": recall(gaps),
        "stages": {stage: stage_counts[stage] / max(len(gaps), 1) for stage in STAGES},
        "recall_by_length": {
            label: {"gaps": sum(lo <= g["length_widths"] < hi for g in gaps),
                    "recall": recall([g for g in gaps if lo <= g["length_widths"] < hi])}
            for label, lo, hi in length_bins
        },
        "recall_by_stubs": {
            label: {"gaps": len(sel), "recall": recall(sel)}
            for label, sel in (
                ("erased end (0-1 stub)", [g for g in gaps if g["stubs"] <= 1]),
                ("simple gap (2 stubs)", [g for g in gaps if g["stubs"] == 2]),
                ("junction gap (3+ stubs)", [g for g in gaps if g["stubs"] >= 3]),
            )
        },
        "damaged": stroke_stats(strokes),
        "undamaged": {**stroke_stats(undamaged),
                      "images_with_off_drawing_strokes": sum(any(s["on_drawing"] < 0.8 for s in r["undamaged_strokes"]) for r in rows),
                      "endpoints": sum(r["undamaged_endpoints"] for r in rows)},
        "endpoints": sum(r["endpoints"] for r in rows),
        "endpoints_at_gaps": sum(r["endpoints_at_gaps"] for r in rows),
        "candidates": sum(r["candidates"] for r in rows),
        "gaps_with_two_endpoints": sum(g["endpoints"] >= 2 for g in gaps) / max(len(gaps), 1),
        "strokes_missed_by_extraction": {
            "mean": float(np.mean([r["strokes_missed_by_extraction"] for r in rows])) if rows else 0.0,
            "worst": max((r["strokes_missed_by_extraction"] for r in rows), default=0.0),
        },
        # The number later changes are judged by first: line drawn off the drawing, in line widths.
        "off_drawing_widths_total": round(sum(s["off_drawing_widths"] for s in strokes + undamaged), 1),
    }


def print_summary(summary: Dict[str, Any], skipped: List[Dict[str, Any]]) -> None:
    def pct(value: Optional[float]) -> str:
        return "   n/a" if value is None else f"{100 * value:5.1f}%"

    sep = "=" * 78
    print(f"\n{sep}\n  GAP-LEVEL EVALUATION  ({summary['images']} image pairs, {summary['gaps']} true gaps)\n{sep}")
    print(f"  Gap recall: {pct(summary['gap_recall'])}")
    for stage in STAGES:
        print(f"    {stage.replace('_', ' '):<26s} {pct(summary['stages'][stage])}")
    print("  Recall by gap length:")
    for label, row in summary["recall_by_length"].items():
        print(f"    {label:<26s} {pct(row['recall'])}   ({row['gaps']} gaps)")
    print("  Recall by gap type:")
    for label, row in summary["recall_by_stubs"].items():
        print(f"    {label:<26s} {pct(row['recall'])}   ({row['gaps']} gaps)")

    for title, key in (("Strokes drawn on damaged images", "damaged"), ("Strokes drawn on UNDAMAGED originals", "undamaged")):
        s = summary[key]
        print(f"  {title}: {s['strokes']}")
        print(f"    wrong (under 50% on the drawing)   {s['wrong_strokes']}")
        print(f"    partly off (50-80%)                {s['partly_off_strokes']}")
        print(f"    line drawn off the drawing         {s['off_drawing_widths']} line widths (of {s['drawn_widths']} drawn)")
    print(f"    undamaged images with off-drawing strokes: {summary['undamaged']['images_with_off_drawing_strokes']}")
    print(f"  Endpoints: {summary['endpoints']} extracted, {summary['endpoints_at_gaps']} at a true gap; "
          f"{summary['undamaged']['endpoints']} on undamaged originals")
    missed = summary["strokes_missed_by_extraction"]
    print(f"  Extraction: {pct(summary['gaps_with_two_endpoints'])} of gaps have two endpoints next to them; "
          f"strokes not represented by any path: mean {pct(missed['mean'])}, worst {pct(missed['worst'])}")
    print(f"  Off-drawing line, damaged + undamaged: {summary['off_drawing_widths_total']} line widths")
    for row in skipped:
        print(f"  SKIPPED {row['name']}: {row['skipped']}")
    print(sep)


def export(rows: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: str) -> None:
    with open(os.path.join(output_dir, "gap_evaluation.json"), "w") as f:
        json.dump({"summary": summary, "per_image": rows}, f, indent=2)

    with open(os.path.join(output_dir, "gap_evaluation.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "gaps", "repaired", "recall", "strokes", "wrong_strokes",
                         "off_drawing_widths", "undamaged_strokes", "undamaged_off_drawing_widths"])
        for r in rows:
            if "skipped" in r:
                continue
            repaired = sum(g["stage"] == "repaired" for g in r["gaps"])
            writer.writerow([
                r["name"], len(r["gaps"]), repaired,
                round(repaired / len(r["gaps"]), 4) if r["gaps"] else "",
                len(r["strokes"]), sum(s["on_drawing"] < 0.5 for s in r["strokes"]),
                round(sum(s["off_drawing_widths"] for s in r["strokes"]), 2),
                len(r["undamaged_strokes"]),
                round(sum(s["off_drawing_widths"] for s in r["undamaged_strokes"]), 2),
            ])
    print(f"  Results saved -> {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Gap-level evaluation against ground truth.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of image pairs.")
    parser.add_argument("--filter", type=str, default=None, help="Only pairs whose name contains this substring.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory for output files.")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Image pairs evaluated in parallel (1 = one at a time).")
    parser.add_argument("--coverage", type=float, default=COVERAGE_THRESHOLD,
                        help="Share of a gap's centre line that must be covered.")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_LINE_WIDTHS,
                        help="How close a stroke must be to count, in line widths.")
    args = parser.parse_args()

    pairs = discover_pairs(name_filter=args.filter)
    if args.limit:
        pairs = pairs[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)

    jobs = [(name, orig, dmg, args.output_dir, args.coverage, args.tolerance) for name, orig, dmg in pairs]
    rows = []
    for i, row in enumerate(map_in_processes(_evaluate_quietly, jobs, args.workers), 1):
        rows.append(row)
        status = row.get("skipped") or f"{sum(g['stage'] == 'repaired' for g in row['gaps'])}/{len(row['gaps'])} gaps repaired"
        print(f"  [{i}/{len(pairs)}] {row['name']}: {status}")

    summary = summarize(rows)
    print_summary(summary, [r for r in rows if "skipped" in r])
    export(rows, summary, args.output_dir)


def _evaluate_quietly(*job) -> Dict[str, Any]:
    """evaluate_pair without the pipeline's per-phase console output."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_pair(*job)


if __name__ == "__main__":
    main()
