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
from dataclasses import dataclass, field
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


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageMetrics:
    """PSNR/SSIM metrics for one image pair."""

    name: str
    original_path: str
    damaged_path: str
    estimated_line_width: float

    # Baseline: damaged vs. original
    psnr_before: float
    ssim_before: float

    # After restoration: restored vs. original
    psnr_after: float
    ssim_after: float

    # Improvement
    delta_psnr: float = 0.0
    delta_ssim: float = 0.0

    # Processing time
    restoration_time_s: float = 0.0

    # Error info (if processing failed)
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Pair discovery
# ═══════════════════════════════════════════════════════════════════════════

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
    if not os.path.isdir(original_dir):
        print(f"\nError: Original dataset directory not found at '{original_dir}'.")
        print("Please download the test dataset from the link in the README and extract it correctly.")
        sys.exit(1)
    if not os.path.isdir(damaged_dir):
        print(f"\nError: Damaged dataset directory not found at '{damaged_dir}'.")
        print("Please download the test dataset from the link in the README and extract it correctly.")
        sys.exit(1)

    originals = {}
    for f in os.listdir(original_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            originals[base] = os.path.join(original_dir, f)

    damaged = {}
    for f in os.listdir(damaged_dir):
        base, ext = os.path.splitext(f)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            damaged[base] = os.path.join(damaged_dir, f)

    pairs = []
    for name, orig_path in sorted(originals.items()):
        damaged_name = name + "_damaged"
        if damaged_name in damaged:
            if name_filter and name_filter.lower() not in name.lower():
                continue
            pairs.append((name, orig_path, damaged[damaged_name]))

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    original: np.ndarray,
    comparison: np.ndarray,
) -> Tuple[float, float]:
    """Compute PSNR and SSIM between two grayscale images.

    Both images must have the same shape.

    Returns:
        (psnr, ssim) tuple.
    """
    # Ensure same shape
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

    # Load images as grayscale
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    damaged = cv2.imread(damaged_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        return ImageMetrics(
            name=name, original_path=original_path, damaged_path=damaged_path,
            estimated_line_width=0.0,
            psnr_before=0.0, ssim_before=0.0,
            psnr_after=0.0, ssim_after=0.0,
            error=f"Cannot read original image: {original_path}",
        )
    if damaged is None:
        return ImageMetrics(
            name=name, original_path=original_path, damaged_path=damaged_path,
            estimated_line_width=0.0,
            psnr_before=0.0, ssim_before=0.0,
            psnr_after=0.0, ssim_after=0.0,
            error=f"Cannot read damaged image: {damaged_path}",
        )

    # Estimate line width from original
    line_width = estimate_line_width(original)

    # Baseline metrics: damaged vs. original
    psnr_before, ssim_before = compute_metrics(original, damaged)

    # Run restoration
    t0 = time.perf_counter()
    try:
        result = restore(damaged_path, output_dir=output_dir)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return ImageMetrics(
            name=name, original_path=original_path, damaged_path=damaged_path,
            estimated_line_width=line_width,
            psnr_before=psnr_before, ssim_before=ssim_before,
            psnr_after=0.0, ssim_after=0.0,
            restoration_time_s=elapsed,
            error=f"Restoration failed: {e}",
        )
    elapsed = time.perf_counter() - t0

    # Render bridges on top of the damaged image
    restored_image = render_bridges_on_damaged(
        restored_paths=result.restored_paths,
        damaged_image=damaged,
        original_image=original,
    )

    # Save restored image
    restored_path = os.path.join(output_dir, f"{name}_restored.png")
    cv2.imwrite(restored_path, restored_image)

    # Restored metrics: damaged + bridges vs. original
    psnr_after, ssim_after = compute_metrics(original, restored_image)

    metrics = ImageMetrics(
        name=name,
        original_path=original_path,
        damaged_path=damaged_path,
        estimated_line_width=line_width,
        psnr_before=psnr_before,
        ssim_before=ssim_before,
        psnr_after=psnr_after,
        ssim_after=ssim_after,
        delta_psnr=psnr_after - psnr_before,
        delta_ssim=ssim_after - ssim_before,
        restoration_time_s=elapsed,
    )

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

    psnr_before = [m.psnr_before for m in valid]
    psnr_after = [m.psnr_after for m in valid]
    ssim_before = [m.ssim_before for m in valid]
    ssim_after = [m.ssim_after for m in valid]
    delta_psnr = [m.delta_psnr for m in valid]
    delta_ssim = [m.delta_ssim for m in valid]

    def _stat_line(label: str, values: List[float], fmt: str = ".2f") -> str:
        arr = np.array(values)
        return (
            f"  {label:<22s} | "
            f"Mean={np.mean(arr):{fmt}}  "
            f"Median={np.median(arr):{fmt}}  "
            f"Min={np.min(arr):{fmt}}  "
            f"Max={np.max(arr):{fmt}}"
        )

    print(_stat_line("PSNR (before)", psnr_before))
    print(_stat_line("PSNR (after)", psnr_after))
    print(_stat_line("d PSNR", delta_psnr))
    print()
    print(_stat_line("SSIM (before)", ssim_before, fmt=".4f"))
    print(_stat_line("SSIM (after)", ssim_after, fmt=".4f"))
    print(_stat_line("d SSIM", delta_ssim, fmt=".4f"))

    improved_psnr = sum(1 for d in delta_psnr if d > 0)
    degraded_psnr = sum(1 for d in delta_psnr if d < 0)
    improved_ssim = sum(1 for d in delta_ssim if d > 0)
    degraded_ssim = sum(1 for d in delta_ssim if d < 0)

    print(f"\n  PSNR: {improved_psnr} improved, {degraded_psnr} degraded, "
          f"{len(valid) - improved_psnr - degraded_psnr} unchanged")
    print(f"  SSIM: {improved_ssim} improved, {degraded_ssim} degraded, "
          f"{len(valid) - improved_ssim - degraded_ssim} unchanged")

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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in results:
            writer.writerow({
                "name": m.name,
                "estimated_line_width": round(m.estimated_line_width, 1),
                "psnr_before": round(m.psnr_before, 4),
                "psnr_after": round(m.psnr_after, 4),
                "delta_psnr": round(m.delta_psnr, 4),
                "ssim_before": round(m.ssim_before, 6),
                "ssim_after": round(m.ssim_after, 6),
                "delta_ssim": round(m.delta_ssim, 6),
                "restoration_time_s": round(m.restoration_time_s, 3),
                "error": m.error or "",
            })

    print(f"\n  CSV saved -> {output_path}")


def export_json(results: List[ImageMetrics], output_path: str) -> None:
    """Export full evaluation report to JSON."""
    valid = [m for m in results if m.error is None]

    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(values)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    report = {
        "evaluation_summary": {
            "total_pairs": len(results),
            "successful": len(valid),
            "failed": len(results) - len(valid),
            "psnr_before": _stats([m.psnr_before for m in valid]),
            "psnr_after": _stats([m.psnr_after for m in valid]),
            "delta_psnr": _stats([m.delta_psnr for m in valid]),
            "ssim_before": _stats([m.ssim_before for m in valid]),
            "ssim_after": _stats([m.ssim_after for m in valid]),
            "delta_ssim": _stats([m.delta_ssim for m in valid]),
            "total_time_s": round(sum(m.restoration_time_s for m in valid), 3),
            "psnr_improved_count": sum(1 for m in valid if m.delta_psnr > 0),
            "psnr_degraded_count": sum(1 for m in valid if m.delta_psnr < 0),
            "ssim_improved_count": sum(1 for m in valid if m.delta_ssim > 0),
            "ssim_degraded_count": sum(1 for m in valid if m.delta_ssim < 0),
        },
        "per_image": [
            {
                "name": m.name,
                "original_path": m.original_path,
                "damaged_path": m.damaged_path,
                "estimated_line_width": round(m.estimated_line_width, 1),
                "psnr_before": round(m.psnr_before, 4),
                "psnr_after": round(m.psnr_after, 4),
                "delta_psnr": round(m.delta_psnr, 4),
                "ssim_before": round(m.ssim_before, 6),
                "ssim_after": round(m.ssim_after, 6),
                "delta_ssim": round(m.delta_ssim, 6),
                "restoration_time_s": round(m.restoration_time_s, 3),
                "error": m.error,
            }
            for m in results
        ],
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
    psnr_before = [m.psnr_before for m in valid]
    psnr_after = [m.psnr_after for m in valid]
    ssim_before = [m.ssim_before for m in valid]
    ssim_after = [m.ssim_after for m in valid]
    d_psnr = [m.delta_psnr for m in valid]
    d_ssim = [m.delta_ssim for m in valid]

    n = len(valid)
    x = np.arange(n)
    bar_w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n * 0.55), 10))
    fig.suptitle("Restoration Evaluation Results", fontsize=16, fontweight="bold", y=0.98)

    # --- Panel 1: PSNR before / after ---
    ax = axes[0, 0]
    ax.bar(x - bar_w / 2, psnr_before, bar_w, label="Damaged", color="#e74c3c", alpha=0.85)
    ax.bar(x + bar_w / 2, psnr_after, bar_w, label="Restored", color="#2ecc71", alpha=0.85)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR: Damaged vs Restored")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: SSIM before / after ---
    ax = axes[0, 1]
    ax.bar(x - bar_w / 2, ssim_before, bar_w, label="Damaged", color="#e74c3c", alpha=0.85)
    ax.bar(x + bar_w / 2, ssim_after, bar_w, label="Restored", color="#2ecc71", alpha=0.85)
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM: Damaged vs Restored")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: Delta PSNR ---
    ax = axes[1, 0]
    colors_psnr = ["#2ecc71" if d >= 0 else "#e74c3c" for d in d_psnr]
    ax.bar(x, d_psnr, color=colors_psnr, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta PSNR (dB)")
    ax.set_title("PSNR Change (Restored - Damaged)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 4: Delta SSIM ---
    ax = axes[1, 1]
    colors_ssim = ["#2ecc71" if d >= 0 else "#e74c3c" for d in d_ssim]
    ax.bar(x, d_ssim, color=colors_ssim, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta SSIM")
    ax.set_title("SSIM Change (Restored - Damaged)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
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
