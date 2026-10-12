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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.extraction import EndpointInfo, ExtractionResult, extract_paths
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.scoring import FEATURE_SCALES, FEATURES, MODEL_PATH, ScoringModel, load_model, score_candidates
from restoration.asp_engine import decode_solution, solve_candidates, utility, RULES_PATH
from restoration.synthesis import synthesize_bridges, merge_restored_paths
from restoration.efd_closure import close_single_gaps
from restoration.geometry import polyline_length, safe_normalize, visual_order_key

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Segment source types that mark a restoration change (anything not original).
CHANGE_SOURCES = ("bridge", "efd_closure")

# A measurement is named among a bridge's reasons when it moves the log-odds by at least this much.
MIN_REPORTED_LOG_ODDS = 0.25


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


def _endpoint_ref(ep: EndpointInfo) -> Dict[str, Any]:
    return {"path_index": int(ep.path_index), "end": str(ep.end), "endpoint_id": int(ep.endpoint_id)}


def _endpoint_name(ep: EndpointInfo) -> str:
    return f"the {ep.end} of path {ep.path_index}"


def _candidate_ref(c: ConnectionCandidate) -> Dict[str, Any]:
    """Short description of a candidate, for lists of alternatives."""
    return {
        "id": int(c.id),
        "shape": str(c.shape),
        "endpoint_a": _endpoint_ref(c.ep_a),
        "endpoint_b": _endpoint_ref(c.ep_b),
        "probability": round(float(c.probability), 4),
    }


def _reasons(c: ConnectionCandidate, sign: int) -> List[Dict[str, Any]]:
    """Up to three measurements that argue most for (sign = 1) or against (sign = -1) a candidate."""
    ranked = sorted(c.contributions.items(), key=lambda item: -sign * item[1])
    return [
        {
            "feature": name,
            "meaning": FEATURES[name],
            "value": float(f"{c.features[name]:.4g}"),
            "log_odds": round(float(value), 2),
        }
        for name, value in ranked[:3]
        if sign * value >= MIN_REPORTED_LOG_ODDS
    ]


def _reasons_text(reasons: List[Dict[str, Any]]) -> str:
    if not reasons:
        return "no single measurement stands out"
    return "; ".join(f"{r['meaning']} = {r['value']:.3g} ({r['log_odds']:+.1f})" for r in reasons)


def _candidates_by_endpoint(candidates: List[ConnectionCandidate]) -> Dict[int, List[ConnectionCandidate]]:
    by_endpoint: Dict[int, List[ConnectionCandidate]] = {}
    for c in candidates:
        for ep in (c.ep_a, c.ep_b):
            by_endpoint.setdefault(ep.endpoint_id, []).append(c)
    return by_endpoint


def _build_bridge_event_explanation(
    label_id: str,
    candidate: ConnectionCandidate,
    threshold: float,
    approx_length_px: float,
    rivals: List[ConnectionCandidate],
) -> str:
    """Create a detailed, human-readable explanation for an ASP bridge event."""
    widths = candidate.distance / max(candidate.ep_a.radius + candidate.ep_b.radius, 1e-6)
    text = (
        f"{label_id} joins {_endpoint_name(candidate.ep_a)} to {_endpoint_name(candidate.ep_b)} with a "
        f"{candidate.shape} bridge (candidate C{candidate.id}) across a gap of {candidate.distance:.1f}px, "
        f"{widths:.1f} stroke widths, drawing about {approx_length_px:.1f}px of line. "
        f"The estimated probability that it redraws a missing stroke is {candidate.probability:.2f}; "
        f"bridges are drawn above {threshold:.2f}. "
        f"In its favour: {_reasons_text(_reasons(candidate, 1))}. "
        f"Against it: {_reasons_text(_reasons(candidate, -1))}."
    )
    if rivals:
        text += (
            f" It was preferred to {len(rivals)} other candidate(s) for these two stroke ends; "
            f"the most likely of them, C{rivals[0].id}, has probability {rivals[0].probability:.2f}."
        )
    return text


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


def _candidate_event_details(
    c: ConnectionCandidate,
    threshold: float,
    rivals: List[ConnectionCandidate],
) -> Dict[str, Any]:
    """Report metadata of the candidate behind a bridge event."""
    details: Dict[str, Any] = {
        **_candidate_ref(c),
        "threshold": round(float(threshold), 4),
        "log_odds_above_threshold": round(float(c.score), 3),
        "distance_px": round(float(c.distance), 2),
        "same_path_closure": bool(c.same_path_closure),
        "in_favour": _reasons(c, 1),
        "against": _reasons(c, -1),
        # Every measurement with the log-odds it adds, relative to an average candidate.
        "measurements": {
            name: {"value": float(f"{c.features[name]:.4g}"), "log_odds": round(float(value), 3)}
            for name, value in c.contributions.items()
        },
        "alternatives": [_candidate_ref(rival) for rival in rivals[:3]],
    }
    if c.corner_point is not None:
        corner = np.asarray(c.corner_point, dtype=np.float64).reshape(2)
        details["corner_point"] = [round(float(corner[0]), 2), round(float(corner[1]), 2)]
    return details


def _segment_key(segment: BezierSegment) -> frozenset:
    """Identity of a segment by where it starts and ends, whichever way round it is stored."""
    cp = segment.control_points
    return frozenset((tuple(np.round(cp[0], 3)), tuple(np.round(cp[3], 3))))


def _change_events(
    final_paths: List[BezierPath],
    candidates: List[ConnectionCandidate],
    accepted: List[ConnectionCandidate],
    threshold: float,
    efd_phase_diagnostics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Describe every restoration change for the report's ``detailed_events``."""
    bridge_owner = {_segment_key(seg): c for c in accepted for seg in c.bridge_bezier}
    by_endpoint = _candidates_by_endpoint(candidates)
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
            candidate = next(
                (bridge_owner[key] for key in map(_segment_key, segments) if key in bridge_owner), None,
            )
            if candidate is not None:
                # The other candidates for the same two stroke ends, most likely first.
                rivals = {
                    other.id: other
                    for ep in (candidate.ep_a, candidate.ep_b)
                    for other in by_endpoint[ep.endpoint_id]
                    if other.id != candidate.id
                }
                ranked = sorted(rivals.values(), key=lambda c: (-c.probability, c.id))
                event["candidate"] = _candidate_event_details(candidate, threshold, ranked)
                event["explanation"] = _build_bridge_event_explanation(
                    label_id, candidate, threshold, approx_length, ranked,
                )
            else:
                event["explanation"] = f"{label_id} is a bridge whose candidate could not be identified."
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


def _open_ends(
    extraction: ExtractionResult,
    candidates: List[ConnectionCandidate],
    accepted: List[ConnectionCandidate],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Every free stroke end that was left open, with its best candidate and why it was not drawn."""
    by_endpoint = _candidates_by_endpoint(candidates)
    used_by = {ep.endpoint_id: c for c in accepted for ep in (c.ep_a, c.ep_b)}

    open_ends: List[Dict[str, Any]] = []
    for ep in extraction.endpoints:
        if ep.endpoint_id in used_by:
            continue
        entry: Dict[str, Any] = {
            "endpoint": _endpoint_ref(ep),
            "coordinates": [round(float(ep.position[0]), 1), round(float(ep.position[1]), 1)],
        }
        options = by_endpoint.get(ep.endpoint_id, [])
        if not options:
            entry["reason"] = "No other free stroke end lies within reach."
        else:
            best = max(options, key=lambda c: (c.probability, -c.id))
            partner = best.ep_b if best.ep_a.endpoint_id == ep.endpoint_id else best.ep_a
            entry["best_candidate"] = _candidate_ref(best)
            described = f"Its most likely bridge, C{best.id} to {_endpoint_name(partner)}, has probability {best.probability:.2f}"
            if utility(best.score) <= 0:
                entry["against"] = _reasons(best, -1)
                entry["reason"] = (
                    f"{described}, not above the {threshold:.2f} needed to draw it. "
                    f"Against it: {_reasons_text(entry['against'])}."
                )
            elif partner.endpoint_id in used_by:
                winner = used_by[partner.endpoint_id]
                entry["reason"] = (
                    f"{described}, but that end is joined by the more plausible bridge C{winner.id} "
                    f"(probability {winner.probability:.2f})."
                )
            else:
                entry["reason"] = f"{described}, but the solver found a better overall set of bridges without it."
        open_ends.append(entry)
    return open_ends


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
    accepted: List[ConnectionCandidate],
    final_paths: List[BezierPath],
    events: List[Dict[str, Any]],
    open_ends: List[Dict[str, Any]],
    efd_phase_summary: Dict[str, Any],
    model: ScoringModel,
    stage_timings: Dict[str, float],
    asp_solver_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a detailed structured restoration report."""
    extraction_open = sum(1 for p in result.paths if not p.is_closed)
    extraction_closed = len(result.paths) - extraction_open
    final_open = sum(1 for p in final_paths if not p.is_closed)
    final_closed = len(final_paths) - final_open

    return {
        "schema_version": "3.0",
        "image": {
            "name": os.path.basename(image_path),
            "shape": list(result.image_shape),
            "diagonal_px": round(result.diagonal, 1),
            "line_width_px": round(result.line_width, 2),
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
            "free_ends_left_open": len(open_ends),
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
                "shape_distribution": dict(Counter(c.shape for c in candidates)),
                "probability_summary": _numeric_summary([c.probability for c in candidates], digits=4),
                "above_threshold": sum(1 for c in candidates if utility(c.score) > 0),
            },
            "scoring_model": {
                "threshold": round(float(model.threshold), 4),
                # log-odds = intercept + sum of weight * (transform(value) - mean) / scale, on the scorecard of the bridge's shape
                "scorecards": {
                    shape: {
                        "intercept": round(float(card.intercept), 4),
                        "features": {
                            name: {"meaning": FEATURES[name], "transform": FEATURE_SCALES.get(name, "x"),
                                   "weight": round(float(weight), 4),
                                   "mean": round(float(mean), 4), "scale": round(float(scale), 4)}
                            for name, weight, mean, scale in zip(card.names, card.weights, card.mean, card.scale)
                        },
                    }
                    for shape, card in model.scorecards.items()
                },
            },
            "selection": {
                "accepted_total": len(accepted),
                "acceptance_rate_percent": round(len(accepted) / max(len(candidates), 1) * 100, 1),
                "accepted_candidate_ids": [int(c.id) for c in accepted],
                "accepted_endpoint_pairs": sorted(_candidate_endpoint_pair_key(c) for c in accepted),
                "accepted_shapes": dict(Counter(c.shape for c in accepted)),
                "accepted_probability_summary": _numeric_summary([c.probability for c in accepted], digits=4),
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
        "open_ends": open_ends,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main API
# ═══════════════════════════════════════════════════════════════════════════

def restore(
    image_path: str,
    model_path: str = MODEL_PATH,
    efd_closure: bool = False,
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
        model_path: the fitted scoring model (see train_scoring.py)
        efd_closure: also run Phase 6, which closes single-gap contours by
            symmetry. Off by default: on the test drawings it repaired a few
            more gaps but drew more line off the drawing than it was worth.
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
    model = load_model(model_path)

    print(f"\n{'=' * 60}")
    print(f"  Restoring: {image_path}")
    print(f"{'=' * 60}")

    # Phase 1: Extraction
    print("  Phase 1: Extracting paths and endpoints...")
    with _timed(stage_timings, "phase1_extraction_s"):
        extraction = extract_paths(image_path)
    print(f"    {len(extraction.paths)} paths, "
          f"{len(extraction.endpoints)} free ends, "
          f"line width = {extraction.line_width:.1f}px")

    # Phase 2: Candidate Generation
    print("  Phase 2: Generating connection candidates...")
    with _timed(stage_timings, "phase2_candidate_generation_s"):
        candidates = generate_candidates(extraction)
    print(f"    {len(candidates)} candidates")

    # Phase 3: Scoring
    print("  Phase 3: Scoring candidates...")
    with _timed(stage_timings, "phase3_scoring_s"):
        scored = score_candidates(candidates, extraction, model)
    if scored:
        print(f"    {sum(1 for c in scored if utility(c.score) > 0)} above the "
              f"{model.threshold:.2f} probability needed to draw a bridge")

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
    print(f"    {len(accepted)} connections accepted")

    # Phase 5: Synthesis
    print("  Phase 5: Synthesizing bridges...")
    with _timed(stage_timings, "phase5_synthesis_s"):
        bridges = synthesize_bridges(accepted)
        restored_paths = merge_restored_paths(extraction.paths, accepted)
    print(f"    {len(bridges)} bridge segment(s) created")

    # Phase 6: EFD Gap Closure (only if asked for)
    final_paths: List[BezierPath] = list(restored_paths)
    efd_phase_diagnostics: List[Dict[str, Any]] = []
    with _timed(stage_timings, "phase6_efd_closure_s"):
        if efd_closure:
            print("  Phase 6: EFD single-gap closure...")
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

        events = _change_events(final_paths, scored, accepted, model.threshold, efd_phase_diagnostics)
        open_ends = _open_ends(extraction, scored, accepted, model.threshold)
        if save_images:
            _save_overlay(image_path, final_paths, output_dir, labeled=True)
            _save_overlay(image_path, final_paths, output_dir, labeled=False)
            _save_visualization(image_path, extraction.paths, final_paths, output_dir)

    elapsed = time.perf_counter() - t0
    stage_timings["total_pipeline_s"] = elapsed

    report = _build_report(
        image_path,
        extraction,
        scored,
        accepted,
        final_paths,
        events,
        open_ends,
        {"enabled": bool(efd_closure), **_build_efd_phase_summary(
            efd_phase_diagnostics, efd_validity_check_enabled, efd_plausibility_threshold,
        )},
        model,
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
