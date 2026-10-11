"""Quantitative evaluation of the heritage sketch restoration pipeline.

Runs the restoration pipeline on all matched damaged/original image pairs,
renders restored outputs as binary sketches, and computes PSNR and SSIM
metrics against the ground truth originals.

Usage:
    python evaluate_restoration.py                  # run all 32 pairs
    python evaluate_restoration.py --limit 5        # run first 5 pairs only
    python evaluate_restoration.py --filter ankh    # run only pairs matching 'ankh'
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from restoration.pipeline import restore
from restoration.render_restored_sketch import (
    estimate_line_width,
    render_bridges_on_damaged,
)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

ORIGINAL_DIR = os.path.join("test_images", "difficult_test_cases_original")
DAMAGED_DIR = os.path.join("test_images", "difficult_test_cases")
OUTPUT_DIR = os.path.join("restoration", "outputs")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# The six reported metrics: (ImageMetrics field, console label, console format).
METRICS = (
    ("psnr_before", "PSNR (before)", ".2f"),
    ("psnr_after", "PSNR (after)", ".2f"),
    ("delta_psnr", "d PSNR", ".2f"),
    ("ssim_before", "SSIM (before)", ".4f"),
    ("ssim_after", "SSIM (after)", ".4f"),
    ("delta_ssim", "d SSIM", ".4f"),
)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageMetrics:
    """PSNR/SSIM metrics for one image pair."""

    name: str
    original_path: str
    damaged_path: str
    estimated_line_width: float = 0.0

    # Baseline: damaged vs. original
    psnr_before: float = 0.0
    ssim_before: float = 0.0

    # After restoration: restored vs. original
    psnr_after: float = 0.0
    ssim_after: float = 0.0

    # Improvement
    delta_psnr: float = 0.0
    delta_ssim: float = 0.0

    # Processing time
    restoration_time_s: float = 0.0

    # Error info (if processing failed)
    error: Optional[str] = None

    def as_row(self) -> Dict[str, Any]:
        """Rounded values as written to the CSV and JSON exports."""
        return {
            "name": self.name,
            "original_path": self.original_path,
            "damaged_path": self.damaged_path,
            "estimated_line_width": round(self.estimated_line_width, 1),
            "psnr_before": round(self.psnr_before, 4),
            "psnr_after": round(self.psnr_after, 4),
            "delta_psnr": round(self.delta_psnr, 4),
            "ssim_before": round(self.ssim_before, 6),
            "ssim_after": round(self.ssim_after, 6),
            "delta_ssim": round(self.delta_ssim, 6),
            "restoration_time_s": round(self.restoration_time_s, 3),
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Pair discovery
# ═══════════════════════════════════════════════════════════════════════════

def _images_by_basename(directory: str, kind: str) -> Dict[str, str]:
    """Map basename (no extension) -> path for the images in a dataset directory."""
    if not os.path.isdir(directory):
        print(f"\nError: {kind} dataset directory not found at '{directory}'.")
        print("Please download the test dataset from the link in the README and extract it correctly.")
        sys.exit(1)
    return {
        os.path.splitext(f)[0]: os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    }


def discover_pairs(
    original_dir: str = ORIGINAL_DIR,
    damaged_dir: str = DAMAGED_DIR,
    name_filter: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Discover matched original/damaged image pairs.

    Naming convention: original is '{name}.png', damaged is '{name}_damaged.png'.

    Args:
        original_dir: Directory containing original (ground truth) images.
        damaged_dir: Directory containing damaged images.
        name_filter: Optional substring filter for image names.

    Returns:
        List of (name, original_path, damaged_path) tuples, sorted by name.
    """
    originals = _images_by_basename(original_dir, "Original")
    damaged = _images_by_basename(damaged_dir, "Damaged")

    return [
        (name, orig_path, damaged[name + "_damaged"])
        for name, orig_path in sorted(originals.items())
        if name + "_damaged" in damaged
        and (not name_filter or name_filter.lower() in name.lower())
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    original: np.ndarray,
    comparison: np.ndarray,
) -> Tuple[float, float]:
    """Compute (PSNR, SSIM) between two grayscale images.

    The comparison image is resized to the original's shape if they differ.
    """
    if original.shape != comparison.shape:
        comparison = cv2.resize(
            comparison, (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    psnr = float(peak_signal_noise_ratio(original, comparison, data_range=255))
    ssim = float(structural_similarity(original, comparison, data_range=255))
    return psnr, ssim


# ═══════════════════════════════════════════════════════════════════════════
# Single image evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single(
    name: str,
    original_path: str,
    damaged_path: str,
    output_dir: str = OUTPUT_DIR,
) -> ImageMetrics:
    """Run restoration on one image pair and compute metrics.

    Args:
        name: Base name of the image (e.g. 'ankh').
        original_path: Path to the ground truth original image.
        damaged_path: Path to the damaged image.
        output_dir: Directory to save outputs.

    Returns:
        ImageMetrics with before/after PSNR and SSIM.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = ImageMetrics(name=name, original_path=original_path, damaged_path=damaged_path)

    # Load images as grayscale
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    damaged = cv2.imread(damaged_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        metrics.error = f"Cannot read original image: {original_path}"
        return metrics
    if damaged is None:
        metrics.error = f"Cannot read damaged image: {damaged_path}"
        return metrics

    metrics.estimated_line_width = estimate_line_width(original)

    # Baseline metrics: damaged vs. original
    metrics.psnr_before, metrics.ssim_before = compute_metrics(original, damaged)

    # Run restoration
    t0 = time.perf_counter()
    try:
        result = restore(damaged_path, output_dir=output_dir)
    except Exception as e:
        metrics.restoration_time_s = time.perf_counter() - t0
        metrics.error = f"Restoration failed: {e}"
        return metrics
    metrics.restoration_time_s = time.perf_counter() - t0

    # Render bridges on top of the damaged image
    restored_image = render_bridges_on_damaged(
        restored_paths=result.restored_paths,
        damaged_image=damaged,
        original_image=original,
    )
    cv2.imwrite(os.path.join(output_dir, f"{name}_restored.png"), restored_image)

    # Restored metrics: damaged + bridges vs. original
    metrics.psnr_after, metrics.ssim_after = compute_metrics(original, restored_image)
    metrics.delta_psnr = metrics.psnr_after - metrics.psnr_before
    metrics.delta_ssim = metrics.ssim_after - metrics.ssim_before
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Console output
# ═══════════════════════════════════════════════════════════════════════════

def print_table_header() -> None:
    """Print the metrics table header."""
    sep = "-" * 120
    print(f"\n{sep}")
    print(
        f"{'Image':<28s} | {'LineW':>5s} | "
        f"{'PSNR_bef':>9s} {'PSNR_aft':>9s} {'dPSNR':>8s} | "
        f"{'SSIM_bef':>9s} {'SSIM_aft':>9s} {'dSSIM':>8s} | "
        f"{'Time':>6s}"
    )
    print(sep)


def print_table_row(m: ImageMetrics) -> None:
    """Print one row of the metrics table."""
    if m.error:
        print(f"  {m.name:<26s} | {'ERR':>5s} | {m.error}")
        return

    delta_psnr_str = f"{m.delta_psnr:+.2f}"
    delta_ssim_str = f"{m.delta_ssim:+.4f}"

    print(
        f"  {m.name:<26s} | {m.estimated_line_width:5.1f} | "
        f"{m.psnr_before:9.2f} {m.psnr_after:9.2f} {delta_psnr_str:>8s} | "
        f"{m.ssim_before:9.4f} {m.ssim_after:9.4f} {delta_ssim_str:>8s} | "
        f"{m.restoration_time_s:6.1f}s"
    )


def _improved_degraded(valid: List[ImageMetrics], field: str) -> Tuple[int, int]:
    """Number of images whose *field* delta is positive / negative."""
    deltas = [getattr(m, field) for m in valid]
    return sum(1 for d in deltas if d > 0), sum(1 for d in deltas if d < 0)


def print_aggregate_summary(results: List[ImageMetrics]) -> None:
    """Print aggregate statistics."""
    valid = [m for m in results if m.error is None]
    if not valid:
        print("\n  No valid results to summarize.")
        return

    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  AGGREGATE SUMMARY ({len(valid)} images)")
    print(sep)

    for field, label, fmt in METRICS:
        if field == "ssim_before":
            print()
        arr = np.array([getattr(m, field) for m in valid])
        print(
            f"  {label:<22s} | "
            f"Mean={np.mean(arr):{fmt}}  "
            f"Median={np.median(arr):{fmt}}  "
            f"Min={np.min(arr):{fmt}}  "
            f"Max={np.max(arr):{fmt}}"
        )

    print()
    for metric in ("psnr", "ssim"):
        improved, degraded = _improved_degraded(valid, f"delta_{metric}")
        print(f"  {metric.upper()}: {improved} improved, {degraded} degraded, "
              f"{len(valid) - improved - degraded} unchanged")

    total_time = sum(m.restoration_time_s for m in valid)
    print(f"\n  Total processing time: {total_time:.1f}s "
          f"(avg {total_time / len(valid):.1f}s per image)")

    failed = [m for m in results if m.error is not None]
    if failed:
        print(f"\n  WARNING: {len(failed)} image(s) failed:")
        for m in failed:
            print(f"    - {m.name}: {m.error}")

    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(results: List[ImageMetrics], output_path: str) -> None:
    """Export per-image metrics to CSV."""
    fieldnames = [
        "name", "estimated_line_width",
        "psnr_before", "psnr_after", "delta_psnr",
        "ssim_before", "ssim_after", "delta_ssim",
        "restoration_time_s", "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for m in results:
            writer.writerow(m.as_row())

    print(f"\n  CSV saved -> {output_path}")


def export_json(results: List[ImageMetrics], output_path: str) -> None:
    """Export full evaluation report to JSON."""
    valid = [m for m in results if m.error is None]

    def stats(field: str) -> Dict[str, float]:
        if not valid:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array([getattr(m, field) for m in valid])
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    psnr_improved, psnr_degraded = _improved_degraded(valid, "delta_psnr")
    ssim_improved, ssim_degraded = _improved_degraded(valid, "delta_ssim")
    report = {
        "evaluation_summary": {
            "total_pairs": len(results),
            "successful": len(valid),
            "failed": len(results) - len(valid),
            **{field: stats(field) for field, _, _ in METRICS},
            "total_time_s": round(sum(m.restoration_time_s for m in valid), 3),
            "psnr_improved_count": psnr_improved,
            "psnr_degraded_count": psnr_degraded,
            "ssim_improved_count": ssim_improved,
            "ssim_degraded_count": ssim_degraded,
        },
        "per_image": [m.as_row() for m in results],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  JSON saved -> {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Graph generation
# ═══════════════════════════════════════════════════════════════════════════

def export_graphs(results: List[ImageMetrics], output_dir: str) -> None:
    """Generate and save evaluation result graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [m for m in results if m.error is None]
    if not valid:
        print("  No valid results to graph.")
        return

    names = [m.name for m in valid]
    n = len(valid)
    x = np.arange(n)
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n * 0.55), 10))
    fig.suptitle("Restoration Evaluation Results", fontsize=16, fontweight="bold", y=0.98)

    def label_images(ax) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)

    # One column per metric: before/after bars on top, the change below.
    for col, (metric, unit) in enumerate((("PSNR", " (dB)"), ("SSIM", ""))):
        before = [getattr(m, f"{metric.lower()}_before") for m in valid]
        after = [getattr(m, f"{metric.lower()}_after") for m in valid]
        delta = [getattr(m, f"delta_{metric.lower()}") for m in valid]

        ax = axes[0, col]
        ax.bar(x - bar_w / 2, before, bar_w, label="Damaged", color="#e74c3c", alpha=0.85)
        ax.bar(x + bar_w / 2, after, bar_w, label="Restored", color="#2ecc71", alpha=0.85)
        ax.set_ylabel(f"{metric}{unit}")
        ax.set_title(f"{metric}: Damaged vs Restored")
        label_images(ax)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, col]
        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in delta]
        ax.bar(x, delta, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"Delta {metric}{unit}")
        ax.set_title(f"{metric} Change (Restored - Damaged)")
        label_images(ax)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    graph_path = os.path.join(output_dir, "evaluation_results.png")
    fig.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Graph saved -> {graph_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate restoration pipeline with PSNR and SSIM metrics.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of image pairs to evaluate.",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only evaluate pairs whose name contains this substring.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help="Directory for output files.",
    )
    args = parser.parse_args()

    # Discover pairs
    pairs = discover_pairs(name_filter=args.filter)
    if args.limit:
        pairs = pairs[:args.limit]

    if not pairs:
        print("No matched image pairs found.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  RESTORATION EVALUATION")
    print(f"  {len(pairs)} image pair(s) to evaluate")
    print(f"{'=' * 60}")

    # Evaluate each pair
    results: List[ImageMetrics] = []
    print_table_header()

    for i, (name, orig_path, dmg_path) in enumerate(pairs, 1):
        print(f"\n  [{i}/{len(pairs)}] Processing: {name}")
        metrics = evaluate_single(name, orig_path, dmg_path, output_dir=args.output_dir)
        results.append(metrics)
        print_table_row(metrics)

    # Aggregate summary
    print_aggregate_summary(results)

    # Export results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    json_path = os.path.join(args.output_dir, "evaluation_report.json")
    export_csv(results, csv_path)
    export_json(results, json_path)
    export_graphs(results, args.output_dir)


if __name__ == "__main__":
    main()
