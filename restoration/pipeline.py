"""Phase 7 — Pipeline Orchestrator.

Simple API:
    result = restore("path/to/damaged_image.png")
    results = restore_batch(["img1.png", "img2.png"])
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment, visualize_paths
from restoration.extraction import ExtractionResult, extract_paths
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.scoring import score_candidates
from restoration.asp_engine import encode_facts, solve, decode_solution, RULES_PATH
from restoration.synthesis import synthesize_bridges, merge_restored_paths
from restoration.efd_closure import close_single_gaps

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Result data class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RestorationResult:
    """Outcome of a single image restoration."""

    image_path: str
    original_paths: List[BezierPath]
    restored_paths: List[BezierPath]
    bridges: List[BezierSegment]
    report: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _save_visualization(
    image_path: str,
    original_paths: List[BezierPath],
    restored_paths: List[BezierPath],
    output_dir: str,
) -> str:
    """Save a side-by-side comparison: original vs. restored."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return ""

    h, w = img_bgr.shape[:2]

    # Draw original paths (red channel)
    canvas_orig = np.zeros((h, w, 3), dtype=np.uint8)
    for path in original_paths:
        pts = path.sample(pts_per_segment=60)
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        if len(pts_int) >= 2:
            cv2.polylines(canvas_orig, [pts_int], False, (0, 0, 255), 2,
                          lineType=cv2.LINE_AA)

    # Draw restored paths (green channel)
    canvas_restore = np.zeros((h, w, 3), dtype=np.uint8)
    for path in restored_paths:
        pts = path.sample(pts_per_segment=60)
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        if len(pts_int) >= 2:
            color = (0, 255, 0) if path.source_type == "restored" else (0, 200, 200)
            cv2.polylines(canvas_restore, [pts_int], False, color, 2,
                          lineType=cv2.LINE_AA)

    # Side-by-side
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("black")

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", color="white")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(canvas_orig, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Extracted Paths ({len(original_paths)})", color="white")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(canvas_restore, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Restored Paths ({len(restored_paths)})", color="white")
    axes[2].axis("off")

    plt.tight_layout()
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_restoration.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  -> Saved restoration visualization -> {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _build_report(
    result: ExtractionResult,
    candidates: List[ConnectionCandidate],
    accepted: List[ConnectionCandidate],
    final_paths: List[BezierPath],
    elapsed: float,
) -> Dict[str, Any]:
    """Build a structured restoration report."""
    return {
        "image_shape": list(result.image_shape),
        "diagonal": round(result.diagonal, 1),
        "extraction": {
            "total_paths": len(result.paths),
            "open_paths": sum(1 for p in result.paths if not p.is_closed),
            "closed_paths": sum(1 for p in result.paths if p.is_closed),
            "endpoints": len(result.endpoints),
            "efd_contours": len(result.efd_contours),
        },
        "candidates": {
            "total_generated": len(candidates),
            "tier1": sum(1 for c in candidates if c.tier == 1),
            "tier2": sum(1 for c in candidates if c.tier == 2),
            "accepted": len(accepted),
            "acceptance_rate": (
                round(len(accepted) / max(len(candidates), 1) * 100, 1)
            ),
        },
        "restoration": {
            "final_paths": len(final_paths),
            "final_open": sum(1 for p in final_paths if not p.is_closed),
            "final_closed": sum(1 for p in final_paths if p.is_closed),
            "bridges_created": len(accepted),
        },
        "timing_seconds": round(elapsed, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def restore(
    image_path: str,
    lookahead_fraction: float = 0.15,
    max_candidates_per_endpoint: int = 5,
    efd_gap_threshold: float = 0.30,
    output_dir: str = OUTPUT_DIR,
) -> RestorationResult:
    """Restore a single damaged sketch image.

    Args:
        image_path: path to the damaged sketch
        lookahead_fraction: endpoint search radius as fraction of image diagonal
        max_candidates_per_endpoint: ASP candidate cap per endpoint
        efd_gap_threshold: max gap/perimeter ratio for EFD closure
        output_dir: where to save visualization outputs

    Returns:
        RestorationResult with original paths, restored paths, bridges, and report
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'=' * 60}")
    print(f"  Restoring: {image_path}")
    print(f"{'=' * 60}")

    # Phase 1: Extraction
    print("  Phase 1: Extracting paths and endpoints...")
    extraction = extract_paths(image_path)
    print(f"    {len(extraction.paths)} paths, "
          f"{len(extraction.endpoints)} endpoints, "
          f"diagonal = {extraction.diagonal:.0f}px")

    # Phase 2: Candidate Generation
    print("  Phase 2: Generating connection candidates...")
    candidates = generate_candidates(
        extraction, lookahead_fraction, max_candidates_per_endpoint,
    )
    t1_count = sum(1 for c in candidates if c.tier == 1)
    t2_count = sum(1 for c in candidates if c.tier == 2)
    print(f"    {len(candidates)} candidates (Tier1={t1_count}, Tier2={t2_count})")

    # Phase 3: Scoring
    print("  Phase 3: Scoring candidates...")
    scored = score_candidates(candidates, extraction)
    if scored:
        print(f"    Best score: {scored[0].score:.3f}, "
              f"Worst score: {scored[-1].score:.3f}")

    # Phase 4: ASP Decision
    print("  Phase 4: Solving with ASP...")
    if scored:
        facts = encode_facts(scored, extraction.endpoints)
        accepted_ids = solve(facts, RULES_PATH)
        accepted = decode_solution(accepted_ids, scored)
    else:
        accepted = []
    print(f"    {len(accepted)} connections accepted")

    # Phase 5: Synthesis
    print("  Phase 5: Synthesizing bridges...")
    bridges = synthesize_bridges(accepted)
    restored_paths = merge_restored_paths(
        extraction.paths, bridges, accepted,
    )
    print(f"    {len(bridges)} bridge segment(s) created")

    # Phase 6: EFD Gap Closure
    print("  Phase 6: EFD single-gap closure...")
    final_paths = close_single_gaps(
        restored_paths, extraction.efd_contours, efd_gap_threshold,
    )
    closed_by_efd = (
        sum(1 for p in final_paths if p.is_closed)
        - sum(1 for p in restored_paths if p.is_closed)
    )
    print(f"    {max(0, closed_by_efd)} path(s) closed by EFD")

    # Phase 7: Output
    elapsed = time.time() - t0
    report = _build_report(extraction, candidates, accepted, final_paths, elapsed)

    _save_visualization(image_path, extraction.paths, final_paths, output_dir)

    # Save report JSON
    report_path = os.path.join(output_dir, f"{name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  -> Report saved -> {report_path}")

    print(f"\n  Done in {elapsed:.2f}s")
    print(f"  Original: {len(extraction.paths)} paths "
          f"({sum(1 for p in extraction.paths if not p.is_closed)} open)")
    print(f"  Restored: {len(final_paths)} paths "
          f"({sum(1 for p in final_paths if not p.is_closed)} open)")
    print(f"{'=' * 60}\n")

    return RestorationResult(
        image_path=image_path,
        original_paths=extraction.paths,
        restored_paths=final_paths,
        bridges=bridges,
        report=report,
    )


def restore_batch(
    image_paths: List[str],
    **kwargs,
) -> List[RestorationResult]:
    """Restore multiple images. Each is processed independently."""
    results = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]")
        results.append(restore(path, **kwargs))
    return results
