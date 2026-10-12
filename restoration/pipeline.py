"""Phase 7 — Pipeline Orchestrator.

Simple API:
    result = restore("path/to/damaged_image.png")
    results = restore_batch(["img1.png", "img2.png"])
"""

from __future__ import annotations

import functools
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.extraction import ExtractionResult, extract_paths
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.scoring import score_candidates
from restoration.asp_engine import decode_solution, solve_candidates, RULES_PATH
from restoration.synthesis import synthesize_bridges, merge_restored_paths
from restoration.efd_closure import close_single_gaps
from restoration.geometry import polyline_length, safe_normalize, visual_order_key

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Segment source types that mark a restoration change (anything not original).
CHANGE_SOURCES = ("bridge", "efd_closure")


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


def _sanitize_accepted_candidates(
    accepted: List[ConnectionCandidate],
) -> Tuple[List[ConnectionCandidate], int]:
    """Drop accepted candidates that reuse an endpoint already consumed."""
    used: Set[int] = set()
    sanitized: List[ConnectionCandidate] = []
    for c in sorted(accepted, key=lambda c: (-c.score, c.distance, c.id)):
        pair = {c.ep_a.endpoint_id, c.ep_b.endpoint_id}
        if len(pair) == 2 and not (pair & used):
            used |= pair
            sanitized.append(c)
    return sanitized, len(accepted) - len(sanitized)


def _bridge_crosses_image_mask(
    bridge_points: np.ndarray,
    dist_map: Optional[np.ndarray],
    ep_a: Any,
    ep_b: Any,
    proximity_px: float = 4.0,
    exclusion_radius: float = 12.0,
) -> bool:
    """Return True if the bridge passes too close to original image lines.

    Ignores bridge points that are within *exclusion_radius* of either endpoint.
    If dist_map is None, falls back to returning False.
    """
    if dist_map is None or len(bridge_points) < 2:
        return False

    h, w = dist_map.shape

    for pt in bridge_points:
        x, y = pt
        if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
            continue
        if float(np.linalg.norm(pt - ep_a.position)) < exclusion_radius:
            continue
        if float(np.linalg.norm(pt - ep_b.position)) < exclusion_radius:
            continue
        if dist_map[int(round(y)), int(round(x))] < proximity_px:
            return True

    return False


@contextmanager
def _timed(timings: Dict[str, float], key: str) -> Iterator[None]:
    """Record the wall-clock duration of the enclosed block under *key*."""
    start = time.perf_counter()
    yield
    timings[key] = time.perf_counter() - start


def _pyplot():
    """Matplotlib's pyplot on the non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ═══════════════════════════════════════════════════════════════════════════
# Restoration changes (bridges and closures) and their report entries
# ═══════════════════════════════════════════════════════════════════════════

def _iter_changes(
    final_paths: List[BezierPath],
) -> Iterator[Tuple[str, List[BezierSegment], np.ndarray]]:
    """Yield (source_type, segments, sampled points) for every contiguous run
    of restoration segments, in path order. Change number N is labelled "RN"."""
    for path in final_paths:
        for source, run in groupby(path.segments, key=lambda seg: seg.source_type):
            if source in CHANGE_SOURCES:
                segments = list(run)
                yield source, segments, np.vstack([s.sample(n=30) for s in segments])


def _build_bridge_event_explanation(
    label_id: str,
    candidate: ConnectionCandidate,
    segment_count: int,
    approx_length_px: float,
) -> str:
    """Create a detailed, human-readable explanation for an ASP bridge event."""
    quality = max(0.0, min(1.0, (float(candidate.bilateral_alignment) + max(0.0, 1.0 - float(candidate.misalignment_deg) / 180.0)) / 2.0))
    scenario_desc = f"'{candidate.scenario}' scenario"
    if candidate.scenario == "extension_intersection" and candidate.same_path_closure:
        scenario_desc = "extension-based same-path closure"

    extension_suffix = ""
    if candidate.scenario == "extension_intersection":
        extension_suffix = f" Extension convergence quality is {float(candidate.extension_quality):.2f}."

    return (
        f"{label_id} is an ASP-selected bridge from candidate C{candidate.id}. "
        f"It links path {candidate.ep_a.path_index} ({candidate.ep_a.end}) to "
        f"path {candidate.ep_b.path_index} ({candidate.ep_b.end}) using the "
        f"{scenario_desc}. Gap distance is {candidate.distance:.1f}px, "
        f"directional alignment is {candidate.bilateral_alignment:.3f}, and endpoint "
        f"misalignment is {candidate.misalignment_deg:.1f} deg. The accepted score is "
        f"{candidate.score:.3f} (tier {candidate.tier}), producing {segment_count} bridge "
        f"segment(s) with an estimated repaired length of {approx_length_px:.1f}px. "
        f"Estimated geometric confidence is {quality:.2f}.{extension_suffix}"
    )


def _build_efd_event_explanation(
    label_id: str,
    segment_count: int,
    approx_length_px: float,
) -> str:
    """Create a detailed explanation for an EFD closure event."""
    return (
        f"{label_id} is an EFD single-gap closure applied after bridge synthesis. "
        f"The contour remained open with one recoverable gap, so the EFD closure stage "
        f"inserted {segment_count} closure segment(s) covering approximately "
        f"{approx_length_px:.1f}px to complete the shape boundary while preserving local "
        f"curvature continuity."
    )


def _candidate_event_details(c: ConnectionCandidate) -> Dict[str, Any]:
    """Report metadata of the candidate behind a bridge event."""
    def endpoint(ep) -> Dict[str, Any]:
        return {"path_index": int(ep.path_index), "end": str(ep.end), "endpoint_id": int(ep.endpoint_id)}

    details: Dict[str, Any] = {
        "id": int(c.id),
        "scenario": str(c.scenario),
        "tier": int(c.tier),
        "distance_px": round(float(c.distance), 2),
        "score": round(float(c.score), 4),
        "extension_quality": round(float(c.extension_quality), 4),
        "bilateral_alignment": round(float(c.bilateral_alignment), 4),
        "misalignment_deg": round(float(c.misalignment_deg), 2),
        "same_path_closure": bool(c.same_path_closure),
        "spur_involved": bool(c.spur_involved),
        "endpoint_a": endpoint(c.ep_a),
        "endpoint_b": endpoint(c.ep_b),
    }
    if c.intersection_point is not None:
        ip = np.asarray(c.intersection_point, dtype=np.float64).reshape(2)
        details["intersection_point"] = [round(float(ip[0]), 2), round(float(ip[1]), 2)]
    return details


def _change_events(
    final_paths: List[BezierPath],
    accepted: List[ConnectionCandidate],
    efd_phase_diagnostics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Describe every restoration change for the report's ``detailed_events``."""
    # Bridge segments are matched back to their candidate by object identity,
    # so a bridge whose path was reversed during merging (copied segments)
    # has no candidate link.
    bridge_owner = {id(seg): c for c in accepted for seg in c.bridge_bezier}
    efd_accepted = iter([e for e in efd_phase_diagnostics if e["accepted"]])

    events: List[Dict[str, Any]] = []
    for number, (source, segments, pts) in enumerate(_iter_changes(final_paths), 1):
        label_id = f"R{number}"
        is_bridge = source == "bridge"
        pos = pts[len(pts) // 2]
        approx_length = polyline_length(pts)

        event: Dict[str, Any] = {
            "id": label_id,
            "source": "asp_bridge" if is_bridge else "efd_gap_closure",
            "type": "ASP Junction Bridge" if is_bridge else "EFD Gap Closure",
            "coordinates": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
            "segment_count": len(segments),
            "geometry": {
                "approx_length_px": round(approx_length, 2),
                "sample_points": int(len(pts)),
            },
        }

        if is_bridge:
            candidate = next((bridge_owner[id(s)] for s in segments if id(s) in bridge_owner), None)
            if candidate is not None:
                event["candidate"] = _candidate_event_details(candidate)
                event["explanation"] = _build_bridge_event_explanation(
                    label_id, candidate, len(segments), approx_length,
                )
            else:
                event["explanation"] = (
                    f"{label_id} is an ASP-generated bridge segment group, but no direct "
                    "candidate metadata link was available after path merging."
                )
        else:
            event["explanation"] = _build_efd_event_explanation(label_id, len(segments), approx_length)
            efd_diag = next(efd_accepted, None)
            if efd_diag:
                event["efd_closure_validity"] = {
                    key: efd_diag[key]
                    for key in (
                        "accepted", "closure_method", "reason", "path_index",
                        "gap_distance_px", "gap_ratio", "plausibility_threshold", "metrics",
                    )
                }

        events.append(event)
    return events


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _draw_paths(shape: Tuple[int, int], paths: List[BezierPath], color_of) -> np.ndarray:
    """Draw *paths* on a black BGR canvas, coloured per path by *color_of*."""
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for path in paths:
        pts = np.round(path.sample(pts_per_segment=60)).astype(np.int32).reshape((-1, 1, 2))
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color_of(path), 2, lineType=cv2.LINE_AA)
    return canvas


def _save_visualization(
    image_path: str,
    original_paths: List[BezierPath],
    restored_paths: List[BezierPath],
    output_dir: str,
) -> None:
    """Save a side-by-side comparison: original image, extracted paths, restored paths."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    shape = img_bgr.shape[:2]
    panels = [
        ("Original Image", img_bgr),
        (f"Extracted Paths ({len(original_paths)})",
         _draw_paths(shape, original_paths, lambda p: (0, 0, 255))),
        (f"Restored Paths ({len(restored_paths)})",
         _draw_paths(shape, restored_paths,
                     lambda p: (0, 255, 0) if p.source_type == "restored" else (0, 200, 200))),
    ]

    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("black")
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="white")
        ax.axis("off")

    plt.tight_layout()
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_restoration.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  -> Saved restoration visualization -> {out_path}")


# Label offsets tried in order, as (along normal, along tangent) multiples of the base offset.
_LABEL_OFFSETS = [
    (1.0, 0.0), (-1.0, 0.0), (1.8, 0.4), (-1.8, 0.4),
    (2.6, -0.3), (-2.6, -0.3), (0.9, 1.0), (-0.9, 1.0),
]


def _place_label(
    pts: np.ndarray,
    text: str,
    placed_boxes: List[Tuple[float, float, float, float]],
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Pick a label centre beside the midpoint of *pts* that avoids earlier labels."""
    mid = len(pts) // 2
    pos = pts[mid]
    tangent = safe_normalize(pts[min(len(pts) - 1, mid + 1)] - pts[max(0, mid - 1)])
    normal = safe_normalize(np.array([-tangent[1], tangent[0]], dtype=np.float64))
    base_offset = float(np.clip(float(np.hypot(height, width)) * 0.007, 8.0, 18.0))
    box_w, box_h = max(18.0, 5.5 * len(text) + 8.0), 12.0
    margin, pad = 8.0, 2.0

    def centred_box(point: np.ndarray):
        cx = float(np.clip(point[0], margin, width - margin))
        cy = float(np.clip(point[1], margin, height - margin))
        return cx, cy, (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)

    def overlaps(a, b) -> bool:
        return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])

    for n_scale, t_scale in _LABEL_OFFSETS:
        cx, cy, box = centred_box(pos + normal * (base_offset * n_scale) + tangent * (base_offset * t_scale))
        if not any(overlaps(box, existing) for existing in placed_boxes):
            break
    else:
        cx, cy, box = centred_box(pos)

    placed_boxes.append(box)
    return cx, cy


def _save_overlay(
    image_path: str,
    final_paths: List[BezierPath],
    output_dir: str,
    labeled: bool,
) -> None:
    """Save the original image with restoration changes drawn in green,
    optionally labelled R1, R2, ... to match the report's events."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    plt = _pyplot()
    h, w = img_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    placed_boxes: List[Tuple[float, float, float, float]] = []
    change_count = 0
    for change_count, (_, _, pts) in enumerate(_iter_changes(final_paths), 1):
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#2ecc40",
            linewidth=2.1,
            alpha=1.0,
            zorder=5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        if labeled:
            label_id = f"R{change_count}"
            x, y = _place_label(pts, label_id, placed_boxes, w, h)
            ax.text(
                x,
                y,
                label_id,
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10,
                bbox=dict(
                    facecolor="#1f7a1f",
                    alpha=0.82,
                    edgecolor="none",
                    boxstyle="round,pad=0.12",
                ),
            )

    kind = "Labeled Overlay" if labeled else "Change Overlay"
    ax.set_title(f"Heritage Restoration: {kind} ({change_count} changes)",
                 color="white", fontsize=14, pad=10)
    ax.axis("off")
    fig.patch.set_facecolor("#0a0a0a")

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_{'labeled_overlay' if labeled else 'overlay'}.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  -> Saved {'labeled' if labeled else 'unlabeled'} restoration overlay -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _numeric_summary(values: List[float], digits: int = 3) -> Dict[str, float]:
    """Summarize numeric lists for reporting."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), digits),
        "max": round(float(np.max(arr)), digits),
        "mean": round(float(np.mean(arr)), digits),
    }


def _candidate_endpoint_pair_key(candidate: ConnectionCandidate) -> str:
    """Canonical order-independent endpoint pair key for exactness checks."""
    key_a, key_b = sorted((f"id:{candidate.ep_a.endpoint_id}", f"id:{candidate.ep_b.endpoint_id}"))
    return f"{key_a}|{key_b}"


def _build_efd_phase_summary(
    efd_phase_diagnostics: List[Dict[str, Any]],
    validity_enabled: bool,
    plausibility_threshold: float,
) -> Dict[str, Any]:
    """Summarize Phase 6 EFD validity diagnostics for report analysis."""
    accepted = [e for e in efd_phase_diagnostics if e["accepted"]]
    rejected = [e for e in efd_phase_diagnostics if not e["accepted"]]
    return {
        "validity_check_enabled": bool(validity_enabled),
        "plausibility_threshold": round(float(plausibility_threshold), 4),
        "attempted": len(efd_phase_diagnostics),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_reasons": dict(Counter(e["reason"] for e in rejected)),
        "plausibility_score_summary": _numeric_summary(
            [e["metrics"]["plausibility_score"] for e in efd_phase_diagnostics], digits=4,
        ),
        "accepted_path_indices": [e["path_index"] for e in accepted],
        "rejected_path_indices": [e["path_index"] for e in rejected],
    }


def _build_report(
    image_path: str,
    result: ExtractionResult,
    candidates: List[ConnectionCandidate],
    candidate_diagnostics: Dict[str, Any],
    accepted: List[ConnectionCandidate],
    final_paths: List[BezierPath],
    events: List[Dict[str, Any]],
    efd_phase_summary: Dict[str, Any],
    dropped_after_sanitize: int,
    stage_timings: Dict[str, float],
    asp_solver_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a detailed structured restoration report."""
    extraction_open = sum(1 for p in result.paths if not p.is_closed)
    extraction_closed = len(result.paths) - extraction_open
    final_open = sum(1 for p in final_paths if not p.is_closed)
    final_closed = len(final_paths) - final_open

    return {
        "schema_version": "2.1",
        "image": {
            "name": os.path.basename(image_path),
            "shape": list(result.image_shape),
            "diagonal_px": round(result.diagonal, 1),
        },
        "summary": {
            "processing_time_s": round(stage_timings["total_pipeline_s"], 3),
            "stage_timings_s": {k: round(float(v), 6) for k, v in stage_timings.items()},
            "total_restoration_events": len(events),
            "open_paths_before": extraction_open,
            "open_paths_after": final_open,
            "closed_paths_before": extraction_closed,
            "closed_paths_after": final_closed,
            "net_open_path_change": final_open - extraction_open,
        },
        "analysis": {
            "extraction": {
                "total_paths": len(result.paths),
                "open_paths": extraction_open,
                "closed_paths": extraction_closed,
                "endpoints": len(result.endpoints),
            },
            "candidate_generation": {
                "total": len(candidates),
                "tier_distribution": {
                    "tier1": sum(1 for c in candidates if c.tier == 1),
                    "tier2": sum(1 for c in candidates if c.tier == 2),
                },
                "scenario_distribution": dict(Counter(c.scenario for c in candidates)),
                "score_summary": _numeric_summary([c.score for c in candidates], digits=4),
                "alignment_summary": _numeric_summary([c.bilateral_alignment for c in candidates], digits=4),
                "misalignment_deg_summary": _numeric_summary([c.misalignment_deg for c in candidates], digits=2),
                "diagnostics": candidate_diagnostics,
            },
            "selection": {
                "accepted_total": len(accepted),
                "dropped_after_endpoint_sanitize": dropped_after_sanitize,
                "acceptance_rate_percent": round(len(accepted) / max(len(candidates), 1) * 100, 1),
                "accepted_candidate_ids": [int(c.id) for c in accepted],
                "accepted_endpoint_pairs": sorted(_candidate_endpoint_pair_key(c) for c in accepted),
                "accepted_scenarios": dict(Counter(c.scenario for c in accepted)),
                "accepted_score_summary": _numeric_summary([c.score for c in accepted], digits=4),
                "accepted_alignment_summary": _numeric_summary([c.bilateral_alignment for c in accepted], digits=4),
                "accepted_misalignment_deg_summary": _numeric_summary([c.misalignment_deg for c in accepted], digits=2),
            },
            "asp_solver": asp_solver_summary,
            "restoration_outcome": {
                "final_paths": len(final_paths),
                "final_open_paths": final_open,
                "final_closed_paths": final_closed,
                "bridges_created": len(accepted),
                "bridge_events_logged": sum(1 for e in events if e["source"] == "asp_bridge"),
                "efd_closure_events_logged": sum(1 for e in events if e["source"] == "efd_gap_closure"),
            },
            "efd_closure_phase": efd_phase_summary,
        },
        "detailed_events": events,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def restore(
    image_path: str,
    lookahead_fraction: float = 0.15,
    max_candidates_per_endpoint: int = 5,
    efd_gap_threshold: float = 0.30,
    efd_validity_check_enabled: bool = True,
    efd_plausibility_threshold: float = 0.50,
    efd_min_gap_for_validity_check: float = 3.0,
    output_dir: str = OUTPUT_DIR,
    asp_timeout_s: float = 30.0,
    save_images: bool = True,
) -> RestorationResult:
    """Restore a single damaged sketch image.

    Args:
        image_path: path to the damaged sketch
        lookahead_fraction: endpoint search radius as fraction of image diagonal
        max_candidates_per_endpoint: ASP candidate cap per endpoint
        efd_gap_threshold: max gap/perimeter ratio for EFD closure
        efd_validity_check_enabled: enable semantic plausibility gate in Phase 6
        efd_plausibility_threshold: minimum plausibility score required for Phase 6 closure
        efd_min_gap_for_validity_check: bypass plausibility check for tiny gaps (pixels)
        output_dir: where to save visualization outputs
        asp_timeout_s: max time in seconds for the ASP solver
        save_images: write the two overlays and the comparison figure; the
            JSON report is written either way

    Returns:
        RestorationResult with original paths, restored paths, bridges, and report
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.perf_counter()
    stage_timings: Dict[str, float] = {}
    asp_solver_summary: Dict[str, Any] = {
        "fact_lines": 0,
        "fact_bytes": 0,
        "accepted_id_count_before_decode": 0,
    }
    name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\n{'=' * 60}")
    print(f"  Restoring: {image_path}")
    print(f"{'=' * 60}")

    # Phase 1: Extraction
    print("  Phase 1: Extracting paths and endpoints...")
    with _timed(stage_timings, "phase1_extraction_s"):
        extraction = extract_paths(image_path)
    print(f"    {len(extraction.paths)} paths, "
          f"{len(extraction.endpoints)} endpoints, "
          f"diagonal = {extraction.diagonal:.0f}px")

    # Phase 2: Candidate Generation
    print("  Phase 2: Generating connection candidates...")
    candidate_diagnostics: Dict[str, Any] = {}
    with _timed(stage_timings, "phase2_candidate_generation_s"):
        candidates = generate_candidates(
            extraction,
            lookahead_fraction,
            max_candidates_per_endpoint,
            diagnostics=candidate_diagnostics,
        )
    t1_count = sum(1 for c in candidates if c.tier == 1)
    t2_count = sum(1 for c in candidates if c.tier == 2)
    print(f"    {len(candidates)} candidates (Tier1={t1_count}, Tier2={t2_count})")

    # Phase 3: Scoring
    print("  Phase 3: Scoring candidates...")
    with _timed(stage_timings, "phase3_scoring_s"):
        scored = score_candidates(candidates, extraction)
    if scored:
        print(f"    Best score: {scored[0].score:.3f}, "
              f"Worst score: {scored[-1].score:.3f}")

    # Phase 4: ASP Decision
    print("  Phase 4: Solving with ASP...")
    accepted: List[ConnectionCandidate] = []
    stage_timings.update(
        phase4_asp_fact_encoding_s=0.0,
        phase4_asp_solving_s=0.0,
        phase4_asp_decode_s=0.0,
    )
    if scored:
        accepted_ids, solver_summary = solve_candidates(
            scored,
            RULES_PATH,
            timeout_s=asp_timeout_s,
        )
        stage_timings["phase4_asp_fact_encoding_s"] = solver_summary["encoding_time_s"]
        stage_timings["phase4_asp_solving_s"] = solver_summary["solving_time_s"]
        asp_solver_summary.update(solver_summary)
        with _timed(stage_timings, "phase4_asp_decode_s"):
            accepted = decode_solution(accepted_ids, scored)
    stage_timings["phase4_asp_total_s"] = (
        stage_timings["phase4_asp_fact_encoding_s"]
        + stage_timings["phase4_asp_solving_s"]
        + stage_timings["phase4_asp_decode_s"]
    )

    accepted, dropped_after_sanitize = _sanitize_accepted_candidates(accepted)
    print(f"    {len(accepted)} connections accepted")
    if dropped_after_sanitize:
        print(f"    {dropped_after_sanitize} conflicting connection(s) dropped")

    # Phase 4.5: Post-processing Image Mask Filtering
    exclusion_radius = max(12.0, extraction.diagonal * 0.015)
    filtered_accepted = [
        c for c in accepted
        if not _bridge_crosses_image_mask(
            c.bridge_points, extraction.dist_map, c.ep_a, c.ep_b,
            proximity_px=4.0, exclusion_radius=exclusion_radius,
        )
    ]
    if len(filtered_accepted) < len(accepted):
        print(f"    {len(accepted) - len(filtered_accepted)} connection(s) dropped "
              f"by strict image mask post-processing")
        accepted = filtered_accepted

    # Phase 5: Synthesis
    print("  Phase 5: Synthesizing bridges...")
    with _timed(stage_timings, "phase5_synthesis_s"):
        bridges = synthesize_bridges(accepted)
        restored_paths = merge_restored_paths(extraction.paths, accepted)
    print(f"    {len(bridges)} bridge segment(s) created")

    # Phase 6: EFD Gap Closure
    print("  Phase 6: EFD single-gap closure...")
    with _timed(stage_timings, "phase6_efd_closure_s"):
        final_paths, efd_phase_diagnostics = close_single_gaps(
            restored_paths, efd_gap_threshold,
            validity_check_enabled=efd_validity_check_enabled,
            plausibility_threshold=efd_plausibility_threshold,
            min_gap_for_validity_check=efd_min_gap_for_validity_check,
        )
    closed_by_efd = (
        sum(1 for p in final_paths if p.is_closed)
        - sum(1 for p in restored_paths if p.is_closed)
    )
    print(f"    {max(0, closed_by_efd)} path(s) closed by EFD")

    # Phase 7: Output
    with _timed(stage_timings, "phase7_output_s"):
        # Ensure visual order for labeling (top-down, left-right)
        final_paths.sort(key=visual_order_key)

        events = _change_events(final_paths, accepted, efd_phase_diagnostics)
        if save_images:
            _save_overlay(image_path, final_paths, output_dir, labeled=True)
            _save_overlay(image_path, final_paths, output_dir, labeled=False)
            _save_visualization(image_path, extraction.paths, final_paths, output_dir)

    elapsed = time.perf_counter() - t0
    stage_timings["total_pipeline_s"] = elapsed

    report = _build_report(
        image_path,
        extraction,
        candidates,
        candidate_diagnostics,
        accepted,
        final_paths,
        events,
        _build_efd_phase_summary(
            efd_phase_diagnostics, efd_validity_check_enabled, efd_plausibility_threshold,
        ),
        dropped_after_sanitize,
        stage_timings,
        asp_solver_summary,
    )

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


def default_workers() -> int:
    """Worker processes for batch runs: leave two cores free, use at most eight."""
    return max(1, min(8, (os.cpu_count() or 1) - 2))


def _run_captured(fn: Callable, job: Sequence[Any]) -> Tuple[Any, str]:
    """Call fn(*job) and return its result with everything it printed."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = fn(*job)
    return result, buffer.getvalue()


def map_in_processes(
    fn: Callable,
    jobs: Sequence[Sequence[Any]],
    workers: int = 1,
    announce: Optional[Callable[[int], None]] = None,
) -> Iterator[Any]:
    """Yield fn(*job) for every job, in order.

    With more than one worker the calls run in separate processes; what each
    call prints is held back and shown as one block when its result is
    yielded. *announce(i)* is called just before the output of job i appears.
    *fn* must be a module-level function so that it can be sent to a worker.
    """
    if workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            if announce:
                announce(i)
            yield fn(*job)
        return

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(_run_captured, fn, job) for job in jobs]
        for i, future in enumerate(futures):
            result, output = future.result()
            if announce:
                announce(i)
            sys.stdout.write(output)
            yield result


def restore_batch(
    image_paths: List[str],
    workers: int = 1,
    **kwargs,
) -> List[RestorationResult]:
    """Restore multiple images. Each is processed independently, in *workers*
    parallel processes if more than one is asked for; results keep the input order."""
    total = len(image_paths)
    return list(map_in_processes(
        functools.partial(restore, **kwargs),
        [(path,) for path in image_paths],
        workers,
        announce=lambda i: print(f"\n[{i + 1}/{total}]"),
    ))
