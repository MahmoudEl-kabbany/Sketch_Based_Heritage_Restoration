"""
R-5: XAI Explanation Module
=============================
Generates human-readable explanations for each restoration decision.

Each action performed by the pipeline is traced back to the geometric
features and Gestalt principles that motivated it.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from restoration.feature_bridge import FeatureBundle, GapCandidate
from restoration.restoration import RestorationResult


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExplanationEntry:
    """One human-readable explanation for a restoration action."""
    action_index: int
    action_type: str
    summary: str
    gestalt_principles: List[str]
    supporting_facts: List[str]
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Generate explanations
# ═══════════════════════════════════════════════════════════════════════

def generate_explanation(
    result: RestorationResult,
    bundle: FeatureBundle,
) -> List[ExplanationEntry]:
    """Generate an explanation entry for each applied action."""
    entries: List[ExplanationEntry] = []

    for idx, action_dict in enumerate(result.actions_applied):
        atype = action_dict.get("type", "unknown")

        if atype == "extend_curve":
            entry = _explain_extend(idx, action_dict, bundle)
        elif atype == "complete_contour":
            entry = _explain_complete(idx, action_dict, bundle)
        elif atype == "mirror_element":
            entry = _explain_mirror(idx, action_dict, bundle)
        elif atype == "action_skipped":
            entry = _explain_skipped(idx, action_dict)
        elif atype == "action_failed":
            entry = _explain_failed(idx, action_dict)
        else:
            entry = ExplanationEntry(
                action_index=idx,
                action_type=atype,
                summary=f"Unknown action type: {atype}",
                gestalt_principles=[],
                supporting_facts=[],
            )
        entries.append(entry)

    return entries


def _explain_extend(
    idx: int, action: dict, bundle: FeatureBundle,
) -> ExplanationEntry:
    """Explain an extend_curve action."""
    pa = action.get("path_a", "?")
    pb = action.get("path_b", "?")
    ea = action.get("endpoint_a", "?")
    eb = action.get("endpoint_b", "?")
    conf = action.get("confidence", 0.0)

    # Find the matching gap
    matching_gap = None
    for gap in bundle.gaps:
        if (gap.path_id_a == pa and gap.path_id_b == pb):
            matching_gap = gap
            break
        if (gap.path_id_a == pb and gap.path_id_b == pa):
            matching_gap = gap
            break

    principles = ["Good Continuation"]
    facts = [
        f"Path {pa} ({ea}) → Path {pb} ({eb})",
    ]

    if matching_gap:
        facts.append(f"Gap distance: {matching_gap.gap_dist:.1f} px")
        facts.append(f"Continuation angle: {matching_gap.continuation_angle:.1f}°")
        facts.append(f"Continuation confidence: {matching_gap.confidence:.2f}")
        principles.append("Proximity")

    summary = (
        f"Connected path {pa} ({ea}) to path {pb} ({eb}) via Bézier bridge. "
        f"The tangent directions at both endpoints align well "
        f"(confidence={conf:.0%}), satisfying the Gestalt principle of "
        f"Good Continuation."
    )

    if matching_gap and matching_gap.gap_dist < 30:
        summary += " The short gap distance also satisfies Proximity."

    return ExplanationEntry(
        action_index=idx,
        action_type="extend_curve",
        summary=summary,
        gestalt_principles=principles,
        supporting_facts=facts,
        confidence=conf,
    )


def _explain_complete(
    idx: int, action: dict, bundle: FeatureBundle,
) -> ExplanationEntry:
    """Explain a complete_contour action."""
    pid = action.get("path_id", "?")
    method = action.get("method", "unknown")
    conf = action.get("confidence", 0.0)

    principles = ["Closure"]
    facts = [f"Path {pid} — method: {method}"]

    # Find matching closure candidate
    for cc in bundle.closure_candidates:
        if cc["path_id"] == pid:
            facts.append(f"Gap distance: {cc['gap_dist']:.1f} px")
            facts.append(f"Gap fraction: {cc['gap_fraction']:.1%} of contour")
            facts.append(f"Closure confidence: {cc['confidence']:.2f}")
            break

    if method == "efd":
        summary = (
            f"Closed contour {pid} using Elliptic Fourier Descriptor reconstruction. "
            f"The EFD captures the global shape frequency content and produces "
            f"a smooth completion that maintains the curvature profile of the "
            f"existing contour (Gestalt Closure principle)."
        )
    else:
        summary = (
            f"Closed contour {pid} using a G1-continuous Bézier curve. "
            f"The closing segment maintains tangent continuity at both "
            f"join points (Gestalt Closure principle)."
        )

    return ExplanationEntry(
        action_index=idx,
        action_type="complete_contour",
        summary=summary,
        gestalt_principles=principles,
        supporting_facts=facts,
        confidence=conf,
    )


def _explain_mirror(
    idx: int, action: dict, bundle: FeatureBundle,
) -> ExplanationEntry:
    """Explain a mirror_element action."""
    pid = action.get("path_id", "?")
    axis = action.get("axis", "?")

    return ExplanationEntry(
        action_index=idx,
        action_type="mirror_element",
        summary=(
            f"Reflected path {pid} across the {axis} symmetry axis. "
            f"This leverages the Gestalt principle of Symmetry to "
            f"reconstruct missing geometry."
        ),
        gestalt_principles=["Symmetry"],
        supporting_facts=[
            f"Path {pid}, axis: {axis}",
            f"Symmetry axis detected in image",
        ],
        confidence=action.get("confidence", 0.5),
    )


def _explain_skipped(idx: int, action: dict) -> ExplanationEntry:
    reason = action.get("reason", "unknown")
    return ExplanationEntry(
        action_index=idx,
        action_type="action_skipped",
        summary=f"Action skipped: {reason}",
        gestalt_principles=[],
        supporting_facts=[f"Reason: {reason}"],
    )


def _explain_failed(idx: int, action: dict) -> ExplanationEntry:
    error = action.get("error", "unknown")
    return ExplanationEntry(
        action_index=idx,
        action_type="action_failed",
        summary=f"Action failed: {error}",
        gestalt_principles=[],
        supporting_facts=[f"Error: {error}"],
    )


# ═══════════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════════

def format_explanation_text(entries: List[ExplanationEntry]) -> str:
    """Format explanation entries as a human-readable report."""
    lines = [
        "=" * 70,
        "  RESTORATION EXPLANATION REPORT",
        "=" * 70,
        "",
    ]

    if not entries:
        lines.append("  No restoration actions were applied.")
        lines.append("")
        return "\n".join(lines)

    for entry in entries:
        lines.append(f"  Action {entry.action_index + 1}: {entry.action_type}")
        lines.append(f"  {'─' * 50}")
        # Wrap summary text
        wrapped = textwrap.fill(entry.summary, width=66, initial_indent="  ",
                                subsequent_indent="  ")
        lines.append(wrapped)
        lines.append("")

        if entry.gestalt_principles:
            principles_str = ", ".join(entry.gestalt_principles)
            lines.append(f"  Gestalt Principles: {principles_str}")

        if entry.confidence > 0:
            lines.append(f"  Confidence: {entry.confidence:.0%}")

        if entry.supporting_facts:
            lines.append("  Supporting Evidence:")
            for fact in entry.supporting_facts:
                lines.append(f"    • {fact}")

        lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_explanation(
    entries: List[ExplanationEntry],
    output_path: str,
) -> None:
    """Save the explanation report to a text file."""
    text = format_explanation_text(entries)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> Explanation report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Image annotation
# ═══════════════════════════════════════════════════════════════════════

def annotate_image(
    image: np.ndarray,
    entries: List[ExplanationEntry],
    result: RestorationResult,
) -> np.ndarray:
    """Draw numbered labels next to each restored segment on the image."""
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 200, 0)  # cyan-ish for annotation labels

    action_idx = 0
    for entry in entries:
        if entry.action_type in ("action_skipped", "action_failed"):
            continue

        label = f"[{entry.action_index + 1}]"

        # Find a position for the label near the restored geometry
        pos = _find_label_position(entry, result, action_idx)
        if pos is not None:
            x, y = int(pos[0]), int(pos[1])
            cv2.putText(annotated, label, (x, y), font, font_scale,
                        color, thickness, cv2.LINE_AA)

        action_idx += 1

    return annotated


def _find_label_position(
    entry: ExplanationEntry,
    result: RestorationResult,
    seg_index: int,
) -> Optional[np.ndarray]:
    """Find an (x, y) position to place an annotation label."""
    if entry.action_type == "extend_curve":
        if seg_index < len(result.new_segments):
            seg = result.new_segments[seg_index]
            mid = seg.evaluate(0.5)
            return mid + np.array([5, -10])

    elif entry.action_type == "complete_contour":
        # Could be a new_segment or efd_arc
        if seg_index < len(result.new_segments):
            seg = result.new_segments[seg_index]
            return seg.evaluate(0.5) + np.array([5, -10])
        # Check efd_arcs
        efd_idx = seg_index - len(result.new_segments)
        if 0 <= efd_idx < len(result.efd_arcs):
            arc = result.efd_arcs[efd_idx]
            mid_idx = len(arc) // 2
            return arc[mid_idx] + np.array([5, -10])

    return None
