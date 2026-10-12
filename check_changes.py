"""Test a change to the pipeline with one command.

Runs the unit tests, refits the scorecards without each drawing, restores
every test drawing with a model that has not seen it, and compares the
results with the recorded baseline (check_baseline.json).

Usage:
    python check_changes.py                   # test the code as it is now
    python check_changes.py --save-baseline   # record its results as the new baseline

Nothing is written to restoration/outputs/ or to the shipped scoring model;
all files go to --output-dir (a folder in the system's temp directory by default).

How a change is judged (fewer wrong strokes come first):
    better     less line drawn off the drawing and no fewer gaps repaired, or the reverse
    worse      more line off the drawing and no more gaps repaired, or the reverse
    trade-off  kept only when the line off the drawing falls by at least 25%
               for a loss of at most 2 points of gap recall
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

import train_scoring
from evaluate_gaps import COVERAGE_THRESHOLD, TOLERANCE_LINE_WIDTHS, evaluate_pair, summarize
from evaluate_restoration import discover_pairs, evaluate_single
from restoration.pipeline import default_workers, map_in_processes
from restoration.scoring import MODEL_PATH

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(ROOT, "check_baseline.json")
DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "restoration_check")

# Differences smaller than these are rounding, not change.
RECALL_TOLERANCE_POINTS = 0.05
OFF_DRAWING_TOLERANCE_WIDTHS = 0.05
# The keep rule for a change that draws less line off the drawing but repairs fewer gaps.
MIN_OFF_DRAWING_DROP = 0.25
MAX_RECALL_LOSS_POINTS = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Steps
# ═══════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> bool:
    """Run test_restoration.py; True if every test passed."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_restoration.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    lines = (result.stdout.strip() or result.stderr.strip()).splitlines()
    print(f"    {lines[-1] if lines else 'no output from pytest'}")
    if result.returncode != 0:
        print("\n".join("    " + line for line in lines[-25:-1]))
    return result.returncode == 0


def _same_model(fitted: Dict[str, Dict[str, Any]], shipped_path: str) -> bool:
    """True if the shipped model file holds the scorecards the current code would fit."""
    if not os.path.exists(shipped_path):
        return False
    with open(shipped_path) as f:
        document = json.load(f)
    if abs(document["threshold"] - train_scoring.THRESHOLD) > 1e-9:
        return False
    shipped = document["scorecards"]
    for shape, card in fitted.items():
        features = shipped.get(shape, {}).get("features", [])
        if [feature["name"] for feature in features] != card["names"]:
            return False
        if not np.allclose([feature["weight"] for feature in features], card["weights"], atol=1e-4):
            return False
    return True


def fit_held_out_models(output_dir: str, workers: int) -> Tuple[Dict[str, str], float, bool]:
    """Fit the scorecards once without each fold of drawings.

    Returns (model file for every drawing, chance that a correct candidate
    outscores a wrong one on held-out drawings, whether the shipped model is
    what the current code would fit).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        images = train_scoring.build_dataset(os.path.join(output_dir, "synthetic"), workers)
    folds = train_scoring.fold_of_drawings(images)
    models = {
        fold: train_scoring.fit([image for image in images if folds[image["drawing"]] != fold])
        for fold in sorted(set(folds.values()))
    }
    held_out = [train_scoring.predict(models[folds[image["drawing"]]], image) for image in images]
    separation = train_scoring.separation(images, held_out)["auc"]

    model_dir = os.path.join(output_dir, "cv")
    for fold, model in models.items():
        train_scoring.write_json(
            os.path.join(model_dir, f"fold_{fold}.json"),
            train_scoring.model_document(model, train_scoring.THRESHOLD, {"held_out_fold": fold}),
        )
    model_of = {drawing: os.path.join(model_dir, f"fold_{fold}.json") for drawing, fold in folds.items()}
    return model_of, separation, _same_model(train_scoring.fit(images), MODEL_PATH)


def _gaps_quietly(*job) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_pair(*job)


def _psnr_quietly(*job) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate_single(*job)


def measure(model_of: Dict[str, str], output_dir: str, workers: int) -> Dict[str, Any]:
    """Restore every drawing with the model that has not seen it; gap benchmark and PSNR/SSIM."""
    pairs = [pair for pair in discover_pairs() if pair[0] in model_of]

    gap_jobs = [(name, original, damaged, os.path.join(output_dir, "gaps"), COVERAGE_THRESHOLD,
                 TOLERANCE_LINE_WIDTHS, model_of[name]) for name, original, damaged in pairs]
    rows = [row for row in map_in_processes(_gaps_quietly, gap_jobs, workers) if "skipped" not in row]
    summary = summarize(rows)

    psnr_jobs = [(name, original, damaged, os.path.join(output_dir, "psnr"), False, model_of[name])
                 for name, original, damaged in pairs]
    metrics = [m for m in map_in_processes(_psnr_quietly, psnr_jobs, workers) if not m.error]

    psnr_of = {m.name: m.psnr_after for m in metrics}
    return {
        "summary": {
            "gap_recall": round(100.0 * summary["gap_recall"], 2),
            "off_drawing_widths": summary["off_drawing_widths_total"],
            "off_drawing_widths_damaged": summary["damaged"]["off_drawing_widths"],
            "off_drawing_widths_undamaged": summary["undamaged"]["off_drawing_widths"],
            "wrong_strokes": summary["damaged"]["wrong_strokes"] + summary["undamaged"]["wrong_strokes"],
            "strokes_on_undamaged": summary["undamaged"]["strokes"],
            "placement_median_widths": round(summary["placement_deviation_widths"]["median"] or 0.0, 3),
            "psnr": round(float(np.mean([m.psnr_after for m in metrics])), 3),
            "ssim": round(float(np.mean([m.ssim_after for m in metrics])), 4),
        },
        "images": {
            row["name"]: {
                "gaps": len(row["gaps"]),
                "repaired": sum(gap["stage"] == "repaired" for gap in row["gaps"]),
                "off_drawing_widths": round(sum(s["off_drawing_widths"]
                                                for s in row["strokes"] + row["undamaged_strokes"]), 1),
                "strokes_on_undamaged": len(row["undamaged_strokes"]),
                "psnr": round(psnr_of.get(row["name"], float("nan")), 2),
            }
            for row in rows
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Comparison with the baseline
# ═══════════════════════════════════════════════════════════════════════════

def verdict(current: Dict[str, float], baseline: Dict[str, float]) -> str:
    """Judge the change by line drawn off the drawing first, gaps repaired second."""
    recall = current["gap_recall"] - baseline["gap_recall"]
    off = current["off_drawing_widths"] - baseline["off_drawing_widths"]
    recall_same = abs(recall) < RECALL_TOLERANCE_POINTS
    off_same = abs(off) < OFF_DRAWING_TOLERANCE_WIDTHS

    if recall_same and off_same:
        return "UNCHANGED: same gaps repaired, same line off the drawing."
    if (off < 0 or off_same) and (recall > 0 or recall_same):
        return "BETTER: keep it."
    if (off > 0 or off_same) and (recall < 0 or recall_same):
        return "WORSE: do not keep it."
    if off < 0:
        drop = -off / max(baseline["off_drawing_widths"], 1e-9)
        if drop >= MIN_OFF_DRAWING_DROP and -recall <= MAX_RECALL_LOSS_POINTS:
            return (f"TRADE-OFF, KEEP: {100 * drop:.0f}% less line off the drawing "
                    f"for {-recall:.1f} points of gap recall.")
        return (f"TRADE-OFF, DO NOT KEEP: {100 * drop:.0f}% less line off the drawing does not "
                f"pay for {-recall:.1f} points of gap recall (needs at least {100 * MIN_OFF_DRAWING_DROP:.0f}% "
                f"for at most {MAX_RECALL_LOSS_POINTS:g} points).")
    return (f"TRADE-OFF, DO NOT KEEP: {recall:.1f} more points of gap recall, but {off:.1f} more line widths "
            "off the drawing. Wrong strokes come first.")


ROWS = (
    ("gap_recall", "Gaps repaired (%)", "{:.1f}"),
    ("off_drawing_widths", "Line off the drawing (line widths)", "{:.1f}"),
    ("wrong_strokes", "Wrong strokes", "{:.0f}"),
    ("strokes_on_undamaged", "Strokes on undamaged originals", "{:.0f}"),
    ("placement_median_widths", "Placement, median (line widths)", "{:.2f}"),
    ("psnr", "Mean PSNR (dB)", "{:.2f}"),
    ("ssim", "Mean SSIM", "{:.4f}"),
)


def print_comparison(current: Dict[str, Any], baseline: Optional[Dict[str, Any]]) -> None:
    sep = "=" * 78
    print(f"\n{sep}\n  RESULTS ON DRAWINGS THE MODEL HAD NOT SEEN\n{sep}")
    header = f"  {'':<36s} {'now':>10s}"
    print(header + (f" {'baseline':>10s} {'change':>10s}" if baseline else ""))
    for key, label, fmt in ROWS:
        line = f"  {label:<36s} {fmt.format(current['summary'][key]):>10s}"
        if baseline:
            before = baseline["summary"][key]
            change = current["summary"][key] - before
            line += f" {fmt.format(before):>10s} {('+' if change > 0 else '') + fmt.format(change):>10s}"
        print(line)

    if not baseline:
        print(f"\n  No baseline at {BASELINE_PATH}; run with --save-baseline to record one.")
        print(sep)
        return

    changed = []
    for name, now in sorted(current["images"].items()):
        before = baseline["images"].get(name)
        if before is None:
            changed.append(f"    {name:<24s} new drawing: {now['repaired']} of {now['gaps']} gaps repaired, "
                           f"{now['off_drawing_widths']:.1f} line widths off the drawing")
        elif (now["repaired"] != before["repaired"]
              or abs(now["off_drawing_widths"] - before["off_drawing_widths"]) >= 0.1
              or now["strokes_on_undamaged"] != before["strokes_on_undamaged"]):
            changed.append(
                f"    {name:<24s} gaps repaired {before['repaired']:>3d} -> {now['repaired']:>3d} of {now['gaps']:<3d} | "
                f"off the drawing {before['off_drawing_widths']:6.1f} -> {now['off_drawing_widths']:6.1f} | "
                f"strokes on its original {before['strokes_on_undamaged']} -> {now['strokes_on_undamaged']}"
            )
    missing = sorted(set(baseline["images"]) - set(current["images"]))
    print(f"\n  Drawings whose result changed: {len(changed)} of {len(current['images'])}")
    print("\n".join(changed) if changed else "    none")
    if missing:
        print(f"  In the baseline but not tested now: {', '.join(missing)}")
    print(f"\n  {verdict(current['summary'], baseline['summary'])}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Test a change to the pipeline against the recorded baseline.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Record the results of this run as the new baseline.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Where reports and intermediate files go (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Images processed in parallel (1 = one at a time).")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.perf_counter()

    print("  Step 1 of 3: unit tests")
    tests_passed = run_unit_tests()

    print("  Step 2 of 3: fitting the scorecards without each drawing (a minute or two)")
    model_of, separation, shipped_model_current = fit_held_out_models(args.output_dir, args.workers)
    print(f"    a correct candidate outscores a wrong one {100 * separation:.1f}% of the time")

    print("  Step 3 of 3: restoring every drawing with a model that has not seen it")
    current = measure(model_of, args.output_dir, args.workers)
    current["summary"]["candidate_separation"] = round(100 * separation, 2)

    baseline = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
    print_comparison(current, baseline)

    if not tests_passed:
        print("  UNIT TESTS FAILED: see above. The results are not to be trusted until they pass.")
    if not shipped_model_current:
        print("  The shipped restoration/scoring_model.json is not what this code would fit.\n"
              "  If you keep the change, run:  python train_scoring.py")

    train_scoring.write_json(os.path.join(args.output_dir, "check_results.json"), current)
    if args.save_baseline:
        if tests_passed:
            train_scoring.write_json(BASELINE_PATH, current)
            print(f"  Baseline saved -> {BASELINE_PATH}")
        else:
            print("  Baseline NOT saved, because the unit tests failed.")
    print(f"  Details -> {args.output_dir}   ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
