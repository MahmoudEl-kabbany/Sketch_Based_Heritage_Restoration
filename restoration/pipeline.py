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
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment, visualize_paths
from restoration.extraction import ExtractionResult, extract_paths
from restoration.candidates import ConnectionCandidate, generate_candidates
from restoration.scoring import score_candidates
from restoration.asp_engine import decode_solution, solve_partitioned, RULES_PATH
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


def _endpoint_token(ep) -> Tuple[str, int]:
    """Canonical endpoint token used for occupancy checks."""
    if getattr(ep, "endpoint_id", -1) >= 0:
        return ("id", int(ep.endpoint_id))
    # Fallback for synthetic endpoints.
    return ("path_end", int(ep.path_index) * 2 + (1 if ep.end == "end" else 0))


def _sanitize_accepted_candidates(
    accepted: List[ConnectionCandidate],
) -> Tuple[List[ConnectionCandidate], int]:
    """Drop accepted candidates that reuse an endpoint already consumed."""
    if not accepted:
        return [], 0

    ordered = sorted(accepted, key=lambda c: (-c.score, c.distance, c.id))
    used_tokens: Set[Tuple[str, int]] = set()
    sanitized: List[ConnectionCandidate] = []
    dropped = 0

    for c in ordered:
        token_a = _endpoint_token(c.ep_a)
        token_b = _endpoint_token(c.ep_b)
        if token_a == token_b or token_a in used_tokens or token_b in used_tokens:
            dropped += 1
            continue
        used_tokens.add(token_a)
        used_tokens.add(token_b)
        sanitized.append(c)

    return sanitized, dropped


def _safe_unit(v: np.ndarray) -> np.ndarray:
    """Return a unit vector with robust fallback for degenerate vectors."""
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


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


def _count_by(items: List[Any], key_fn) -> Dict[str, int]:
    """Count occurrences of computed keys in a list."""
    counts: Dict[str, int] = {}
    for item in items:
        key = str(key_fn(item))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _endpoint_token_key(token: Tuple[str, int]) -> str:
    """Serialize endpoint token tuples into stable text keys."""
    return f"{token[0]}:{token[1]}"


def _candidate_endpoint_pair_key(candidate: ConnectionCandidate) -> str:
    """Canonical order-independent endpoint pair key for exactness checks."""
    token_a = _endpoint_token(candidate.ep_a)
    token_b = _endpoint_token(candidate.ep_b)
    key_a = _endpoint_token_key(token_a)
    key_b = _endpoint_token_key(token_b)
    if key_a <= key_b:
        return f"{key_a}|{key_b}"
    return f"{key_b}|{key_a}"


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
        ext_quality = float(getattr(candidate, "extension_quality", 0.0))
        extension_suffix = f" Extension convergence quality is {ext_quality:.2f}."

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


def _build_efd_phase_summary(
    efd_phase_diagnostics: Optional[List[Dict[str, Any]]],
    validity_enabled: bool,
    plausibility_threshold: float,
) -> Dict[str, Any]:
    """Summarize Phase 6 EFD validity diagnostics for report analysis."""
    entries = list(efd_phase_diagnostics or [])
    attempted = len(entries)
    accepted_entries = [e for e in entries if bool(e.get("accepted", False))]
    rejected_entries = [e for e in entries if not bool(e.get("accepted", False))]

    reason_counts: Dict[str, int] = {}
    for entry in rejected_entries:
        reason = str(entry.get("reason", "unknown"))
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    plausibility_values: List[float] = []
    for entry in entries:
        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        val = metrics.get("plausibility_score") if isinstance(metrics, dict) else None
        if isinstance(val, (float, int)):
            plausibility_values.append(float(val))

    return {
        "validity_check_enabled": bool(validity_enabled),
        "plausibility_threshold": round(float(plausibility_threshold), 4),
        "attempted": int(attempted),
        "accepted": int(len(accepted_entries)),
        "rejected": int(len(rejected_entries)),
        "rejection_reasons": reason_counts,
        "plausibility_score_summary": _numeric_summary(plausibility_values, digits=4),
        "accepted_path_indices": [int(e.get("path_index", -1)) for e in accepted_entries],
        "rejected_path_indices": [int(e.get("path_index", -1)) for e in rejected_entries],
    }


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


def _save_labeled_restoration(
    image_path: str,
    final_paths: List[BezierPath],
    accepted: Optional[List[ConnectionCandidate]],
    efd_phase_diagnostics: Optional[List[Dict[str, Any]]],
    output_dir: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Save an overlay of the original image with labeled restoration bridges."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return "", []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img_bgr.shape[:2]
    # Use a consistent figure size
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    bridge_owner_by_segment_id: Dict[int, ConnectionCandidate] = {}
    if accepted:
        for c in accepted:
            for bridge_seg in getattr(c, "bridge_bezier", []):
                bridge_owner_by_segment_id[id(bridge_seg)] = c

    changes = []
    change_count = 0
    efd_accepted_diags = [e for e in (efd_phase_diagnostics or []) if bool(e.get("accepted", False))]
    efd_diag_idx = 0
    placed_boxes: List[Tuple[float, float, float, float]] = []
    img_diag = float(np.hypot(h, w))

    def _estimate_label_box(text: str) -> Tuple[float, float]:
        width = max(18.0, 5.5 * len(text) + 8.0)
        height = 12.0
        return width, height

    def _overlaps(box_a: Tuple[float, float, float, float],
                  box_b: Tuple[float, float, float, float],
                  pad: float = 2.0) -> bool:
        return not (
            box_a[2] + pad < box_b[0]
            or box_b[2] + pad < box_a[0]
            or box_a[3] + pad < box_b[1]
            or box_b[3] + pad < box_a[1]
        )

    for path in final_paths:
        i = 0
        while i < len(path.segments):
            seg = path.segments[i]
            # Restoration changes are non-original segments
            if seg.source_type in ["bridge", "efd_closure"]:
                start_i = i
                source = seg.source_type
                # Group contiguous segments of the same source type
                while i < len(path.segments) and path.segments[i].source_type == source:
                    i += 1
                end_i = i

                change_count += 1
                label_id = f"R{change_count}"

                # Sample points for the change accurately
                change_pts_list = []
                event_segments = path.segments[start_i:end_i]
                for s in event_segments:
                    change_pts_list.append(s.sample(n=30))
                change_pts = np.vstack(change_pts_list)

                # Draw a single clean green line (no glow).
                ax.plot(
                    change_pts[:, 0],
                    change_pts[:, 1],
                    color="#2ecc40",
                    linewidth=2.1,
                    alpha=1.0,
                    zorder=5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

                # Label position - midpoint, offset along normal with collision avoidance.
                mid_idx = len(change_pts) // 2
                pos = change_pts[mid_idx]
                prev_pt = change_pts[max(0, mid_idx - 1)]
                next_pt = change_pts[min(len(change_pts) - 1, mid_idx + 1)]
                tangent = _safe_unit(next_pt - prev_pt)
                normal = _safe_unit(np.array([-tangent[1], tangent[0]], dtype=np.float64))
                base_offset = float(np.clip(img_diag * 0.007, 8.0, 18.0))
                box_w, box_h = _estimate_label_box(label_id)

                anchor = pos.copy()
                attempts = [
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (1.8, 0.4),
                    (-1.8, 0.4),
                    (2.6, -0.3),
                    (-2.6, -0.3),
                    (0.9, 1.0),
                    (-0.9, 1.0),
                ]
                margin = 8.0
                for n_scale, t_scale in attempts:
                    candidate_pos = pos + normal * (base_offset * n_scale) + tangent * (base_offset * t_scale)
                    cx = float(np.clip(candidate_pos[0], margin, w - margin))
                    cy = float(np.clip(candidate_pos[1], margin, h - margin))
                    box = (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)
                    if not any(_overlaps(box, existing) for existing in placed_boxes):
                        anchor = np.array([cx, cy], dtype=np.float64)
                        placed_boxes.append(box)
                        break
                else:
                    cx = float(np.clip(pos[0], margin, w - margin))
                    cy = float(np.clip(pos[1], margin, h - margin))
                    box = (cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0)
                    anchor = np.array([cx, cy], dtype=np.float64)
                    placed_boxes.append(box)

                ax.text(
                    anchor[0],
                    anchor[1],
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

                approx_length = float(np.sum(np.linalg.norm(np.diff(change_pts, axis=0), axis=1))) if len(change_pts) >= 2 else 0.0

                owning_candidate = None
                if source == "bridge":
                    for bridge_seg in event_segments:
                        owning_candidate = bridge_owner_by_segment_id.get(id(bridge_seg))
                        if owning_candidate is not None:
                            break

                # Add to changes list for report explanation
                desc = "ASP Junction Bridge" if source == "bridge" else "EFD Gap Closure"
                event_entry: Dict[str, Any] = {
                    "id": label_id,
                    "source": "asp_bridge" if source == "bridge" else "efd_gap_closure",
                    "type": desc,
                    "coordinates": [round(float(pos[0]), 1), round(float(pos[1]), 1)],
                    "segment_count": end_i - start_i,
                    "geometry": {
                        "approx_length_px": round(approx_length, 2),
                        "sample_points": int(len(change_pts)),
                    },
                }

                if source == "bridge" and owning_candidate is not None:
                    event_entry["candidate"] = {
                        "id": int(owning_candidate.id),
                        "scenario": str(owning_candidate.scenario),
                        "tier": int(owning_candidate.tier),
                        "distance_px": round(float(owning_candidate.distance), 2),
                        "score": round(float(owning_candidate.score), 4),
                        "extension_quality": round(float(getattr(owning_candidate, "extension_quality", 0.0)), 4),
                        "bilateral_alignment": round(float(owning_candidate.bilateral_alignment), 4),
                        "misalignment_deg": round(float(owning_candidate.misalignment_deg), 2),
                        "same_path_closure": bool(owning_candidate.same_path_closure),
                        "spur_involved": bool(owning_candidate.spur_involved),
                        "endpoint_a": {
                            "path_index": int(owning_candidate.ep_a.path_index),
                            "end": str(owning_candidate.ep_a.end),
                            "endpoint_id": int(getattr(owning_candidate.ep_a, "endpoint_id", -1)),
                        },
                        "endpoint_b": {
                            "path_index": int(owning_candidate.ep_b.path_index),
                            "end": str(owning_candidate.ep_b.end),
                            "endpoint_id": int(getattr(owning_candidate.ep_b, "endpoint_id", -1)),
                        },
                    }
                    if getattr(owning_candidate, "intersection_point", None) is not None:
                        ip = np.asarray(owning_candidate.intersection_point, dtype=np.float64).reshape(2)
                        event_entry["candidate"]["intersection_point"] = [
                            round(float(ip[0]), 2),
                            round(float(ip[1]), 2),
                        ]
                    event_entry["explanation"] = _build_bridge_event_explanation(
                        label_id, owning_candidate, end_i - start_i, approx_length,
                    )
                elif source == "bridge":
                    event_entry["explanation"] = (
                        f"{label_id} is an ASP-generated bridge segment group, but no direct "
                        "candidate metadata link was available after path merging."
                    )
                else:
                    efd_diag: Dict[str, Any] = {}
                    if efd_diag_idx < len(efd_accepted_diags):
                        efd_diag = dict(efd_accepted_diags[efd_diag_idx])
                        efd_diag_idx += 1

                    event_entry["explanation"] = _build_efd_event_explanation(
                        label_id, end_i - start_i, approx_length,
                    )
                    if efd_diag:
                        event_entry["efd_closure_validity"] = {
                            "accepted": bool(efd_diag.get("accepted", True)),
                            "closure_method": str(efd_diag.get("closure_method", "unknown")),
                            "reason": str(efd_diag.get("reason", "accepted")),
                            "path_index": int(efd_diag.get("path_index", -1)),
                            "gap_distance_px": float(efd_diag.get("gap_distance_px", 0.0)),
                            "gap_ratio": float(efd_diag.get("gap_ratio", 0.0)),
                            "plausibility_threshold": float(efd_diag.get("plausibility_threshold", 0.0)),
                            "metrics": efd_diag.get("metrics", {}),
                        }

                changes.append(event_entry)
            else:
                i += 1

    ax.set_title(f"Heritage Restoration: Labeled Overlay ({change_count} changes)",
                 color="white", fontsize=14, pad=10)
    ax.axis("off")
    fig.patch.set_facecolor("#0a0a0a")

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_labeled_overlay.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  -> Saved labeled restoration overlay -> {out_path}")

    return out_path, changes


def _save_unlabeled_restoration_overlay(
    image_path: str,
    final_paths: List[BezierPath],
    output_dir: str,
) -> str:
    """Save an overlay of restoration changes on the original image (no labels)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return ""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    change_count = 0

    for path in final_paths:
        i = 0
        while i < len(path.segments):
            seg = path.segments[i]
            if seg.source_type in ["bridge", "efd_closure"]:
                source = seg.source_type
                event_segments: List[BezierSegment] = []
                while i < len(path.segments) and path.segments[i].source_type == source:
                    event_segments.append(path.segments[i])
                    i += 1

                if event_segments:
                    change_count += 1
                    change_pts = np.vstack([s.sample(n=30) for s in event_segments])
                    ax.plot(
                        change_pts[:, 0],
                        change_pts[:, 1],
                        color="#2ecc40",
                        linewidth=2.1,
                        alpha=1.0,
                        zorder=5,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                    )
            else:
                i += 1

    ax.set_title(
        f"Heritage Restoration: Change Overlay ({change_count} changes)",
        color="white",
        fontsize=14,
        pad=10,
    )
    ax.axis("off")
    fig.patch.set_facecolor("#0a0a0a")

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_overlay.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"  -> Saved unlabeled restoration overlay -> {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _build_report(
    image_path: str,
    result: ExtractionResult,
    candidates: List[ConnectionCandidate],
    candidate_diagnostics: Optional[Dict[str, Any]],
    accepted: List[ConnectionCandidate],
    final_paths: List[BezierPath],
    elapsed: float,
    restoration_history: List[Dict[str, Any]] = None,
    efd_phase_diagnostics: Optional[List[Dict[str, Any]]] = None,
    efd_validity_check_enabled: bool = True,
    efd_plausibility_threshold: float = 0.50,
    dropped_after_sanitize: int = 0,
    stage_timings: Optional[Dict[str, float]] = None,
    asp_solver_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a detailed structured restoration report."""
    extraction_open = sum(1 for p in result.paths if not p.is_closed)
    extraction_closed = sum(1 for p in result.paths if p.is_closed)
    final_open = sum(1 for p in final_paths if not p.is_closed)
    final_closed = sum(1 for p in final_paths if p.is_closed)

    accepted_ids = [int(c.id) for c in accepted]
    accepted_endpoint_pairs = sorted(_candidate_endpoint_pair_key(c) for c in accepted)
    accepted_scenarios = _count_by(accepted, lambda c: c.scenario)
    scenario_counts = _count_by(candidates, lambda c: c.scenario)

    bridge_events = [e for e in (restoration_history or []) if e.get("source") == "asp_bridge"]
    efd_events = [e for e in (restoration_history or []) if e.get("source") == "efd_gap_closure"]
    efd_phase_summary = _build_efd_phase_summary(
        efd_phase_diagnostics,
        validity_enabled=efd_validity_check_enabled,
        plausibility_threshold=efd_plausibility_threshold,
    )

    all_scores = [float(c.score) for c in candidates]
    accepted_scores = [float(c.score) for c in accepted]
    all_alignment = [float(c.bilateral_alignment) for c in candidates]
    accepted_alignment = [float(c.bilateral_alignment) for c in accepted]
    all_misalignment = [float(c.misalignment_deg) for c in candidates]
    accepted_misalignment = [float(c.misalignment_deg) for c in accepted]

    return {
        "schema_version": "2.0",
        "image": {
            "name": os.path.basename(image_path),
            "shape": list(result.image_shape),
            "diagonal_px": round(result.diagonal, 1),
        },
        "summary": {
            "processing_time_s": round(elapsed, 3),
            "stage_timings_s": {
                str(k): round(float(v), 6)
                for k, v in (stage_timings or {}).items()
            },
            "total_restoration_events": len(restoration_history or []),
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
                "efd_contours": len(result.efd_contours),
            },
            "candidate_generation": {
                "total": len(candidates),
                "tier_distribution": {
                    "tier1": sum(1 for c in candidates if c.tier == 1),
                    "tier2": sum(1 for c in candidates if c.tier == 2),
                },
                "scenario_distribution": scenario_counts,
                "score_summary": _numeric_summary(all_scores, digits=4),
                "alignment_summary": _numeric_summary(all_alignment, digits=4),
                "misalignment_deg_summary": _numeric_summary(all_misalignment, digits=2),
                "diagnostics": candidate_diagnostics or {},
            },
            "selection": {
                "accepted_total": len(accepted),
                "dropped_after_endpoint_sanitize": int(dropped_after_sanitize),
                "acceptance_rate_percent": round(len(accepted) / max(len(candidates), 1) * 100, 1),
                "accepted_candidate_ids": accepted_ids,
                "accepted_endpoint_pairs": accepted_endpoint_pairs,
                "accepted_scenarios": accepted_scenarios,
                "accepted_score_summary": _numeric_summary(accepted_scores, digits=4),
                "accepted_alignment_summary": _numeric_summary(accepted_alignment, digits=4),
                "accepted_misalignment_deg_summary": _numeric_summary(accepted_misalignment, digits=2),
            },
            "asp_solver": asp_solver_summary or {},
            "restoration_outcome": {
                "final_paths": len(final_paths),
                "final_open_paths": final_open,
                "final_closed_paths": final_closed,
                "bridges_created": len(accepted),
                "bridge_events_logged": len(bridge_events),
                "efd_closure_events_logged": len(efd_events),
            },
            "efd_closure_phase": efd_phase_summary,
        },
        "detailed_events": restoration_history or [],
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
        asp_timeout_s: max time in seconds for the ASP solver to run per component

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
    t_phase = time.perf_counter()
    extraction = extract_paths(image_path)
    stage_timings["phase1_extraction_s"] = time.perf_counter() - t_phase
    print(f"    {len(extraction.paths)} paths, "
          f"{len(extraction.endpoints)} endpoints, "
          f"diagonal = {extraction.diagonal:.0f}px")

    # Phase 2: Candidate Generation
    print("  Phase 2: Generating connection candidates...")
    candidate_diagnostics: Dict[str, Any] = {}
    t_phase = time.perf_counter()
    candidates = generate_candidates(
        extraction,
        lookahead_fraction,
        max_candidates_per_endpoint,
        diagnostics=candidate_diagnostics,
    )
    stage_timings["phase2_candidate_generation_s"] = time.perf_counter() - t_phase
    t1_count = sum(1 for c in candidates if c.tier == 1)
    t2_count = sum(1 for c in candidates if c.tier == 2)
    print(f"    {len(candidates)} candidates (Tier1={t1_count}, Tier2={t2_count})")

    # Phase 3: Scoring
    print("  Phase 3: Scoring candidates...")
    t_phase = time.perf_counter()
    scored = score_candidates(candidates, extraction)
    stage_timings["phase3_scoring_s"] = time.perf_counter() - t_phase
    if scored:
        print(f"    Best score: {scored[0].score:.3f}, "
              f"Worst score: {scored[-1].score:.3f}")

    # Phase 4: ASP Decision
    print("  Phase 4: Solving with ASP...")
    if scored:
        accepted_ids, partition_summary = solve_partitioned(
            scored,
            extraction.endpoints,
            RULES_PATH,
            timeout_s=asp_timeout_s,
        )
        stage_timings["phase4_asp_fact_encoding_s"] = float(partition_summary.get("encoding_time_s", 0.0))
        stage_timings["phase4_asp_solving_s"] = float(partition_summary.get("solving_time_s", 0.0))
        asp_solver_summary.update(partition_summary)

        t_decode = time.perf_counter()
        accepted = decode_solution(accepted_ids, scored)
        stage_timings["phase4_asp_decode_s"] = time.perf_counter() - t_decode
    else:
        accepted = []
        stage_timings["phase4_asp_fact_encoding_s"] = 0.0
        stage_timings["phase4_asp_solving_s"] = 0.0
        stage_timings["phase4_asp_decode_s"] = 0.0

    stage_timings["phase4_asp_total_s"] = (
        stage_timings.get("phase4_asp_fact_encoding_s", 0.0)
        + stage_timings.get("phase4_asp_solving_s", 0.0)
        + stage_timings.get("phase4_asp_decode_s", 0.0)
    )

    accepted, dropped_after_sanitize = _sanitize_accepted_candidates(accepted)
    print(f"    {len(accepted)} connections accepted")
    if dropped_after_sanitize:
        print(f"    {dropped_after_sanitize} conflicting connection(s) dropped")

    # Phase 5: Synthesis
    print("  Phase 5: Synthesizing bridges...")
    t_phase = time.perf_counter()
    bridges = synthesize_bridges(accepted)
    restored_paths = merge_restored_paths(
        extraction.paths, bridges, accepted,
    )
    stage_timings["phase5_synthesis_s"] = time.perf_counter() - t_phase
    print(f"    {len(bridges)} bridge segment(s) created")

    # Phase 6: EFD Gap Closure
    print("  Phase 6: EFD single-gap closure...")
    t_phase = time.perf_counter()
    final_paths, efd_phase_diagnostics = close_single_gaps(
        restored_paths, extraction.efd_contours, efd_gap_threshold,
        validity_check_enabled=efd_validity_check_enabled,
        plausibility_threshold=efd_plausibility_threshold,
        min_gap_for_validity_check=efd_min_gap_for_validity_check,
        return_diagnostics=True,
    )
    stage_timings["phase6_efd_closure_s"] = time.perf_counter() - t_phase
    closed_by_efd = (
        sum(1 for p in final_paths if p.is_closed)
        - sum(1 for p in restored_paths if p.is_closed)
    )
    print(f"    {max(0, closed_by_efd)} path(s) closed by EFD")

    # Phase 7: Output
    t_phase = time.perf_counter()

    # Labeled overlay visualization + logs
    _, change_logs = _save_labeled_restoration(
        image_path,
        final_paths,
        accepted,
        efd_phase_diagnostics,
        output_dir,
    )
    _save_unlabeled_restoration_overlay(image_path, final_paths, output_dir)
    _save_visualization(image_path, extraction.paths, final_paths, output_dir)

    stage_timings["phase7_output_s"] = time.perf_counter() - t_phase
    elapsed = time.perf_counter() - t0
    stage_timings["total_pipeline_s"] = elapsed

    report = _build_report(
        image_path,
        extraction,
        candidates,
        candidate_diagnostics,
        accepted,
        final_paths,
        elapsed,
        restoration_history=change_logs,
        efd_phase_diagnostics=efd_phase_diagnostics,
        efd_validity_check_enabled=efd_validity_check_enabled,
        efd_plausibility_threshold=efd_plausibility_threshold,
        dropped_after_sanitize=dropped_after_sanitize,
        stage_timings=stage_timings,
        asp_solver_summary=asp_solver_summary,
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
