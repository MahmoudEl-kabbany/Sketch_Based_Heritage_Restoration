from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from restoration.asp.asp_inference import RankedHypothesis
from restoration.restoration import RestorationResult, ShapeVocab


@dataclass
class XAIConfig:
    """Configuration for explanation generation."""

    include_surrogate_rules: bool = True
    include_rejected_summary: bool = True
    include_confidence_bands: bool = True


@dataclass
class ExplanationReport:
    """Human-readable explanation output."""

    summary: str
    explanations: List[str] = field(default_factory=list)
    surrogate_rules: List[str] = field(default_factory=list)


def _confidence_band(conf: float) -> str:
    if conf >= 0.75:
        return "high"
    if conf >= 0.45:
        return "medium"
    return "low"


def generate_report(
    hypotheses: Sequence[RankedHypothesis],
    result: RestorationResult,
    vocab: Optional[ShapeVocab],
    config: Optional[XAIConfig] = None,
) -> ExplanationReport:
    """Generate a technical report connecting actions to geometric rationale."""
    cfg = config or XAIConfig()

    explanations: List[str] = []
    used_pairs = {(a.endpoint_a_id, a.endpoint_b_id) for a in result.additions}

    if result.additions:
        for add in result.additions:
            band = _confidence_band(add.confidence)
            line = (
                f"Segment #{add.segment_id}: method={add.method}, confidence={add.confidence:.3f} ({band}), "
                f"paths=({add.path_a},{add.path_b}), endpoints=(e{add.endpoint_a_id},e{add.endpoint_b_id}), "
                f"forced={add.is_forced}. Rationale: {add.reason}."
            )
            explanations.append(line)
    else:
        explanations.append("No restoration segment was synthesized from the current hypotheses.")

    if cfg.include_rejected_summary and hypotheses:
        rejected = [
            h for h in hypotheses
            if (h.endpoint_a_id, h.endpoint_b_id) not in used_pairs
            and (h.endpoint_b_id, h.endpoint_a_id) not in used_pairs
        ]
        if rejected:
            top = sorted(rejected, key=lambda h: (-h.confidence, h.score))[:5]
            for h in top:
                explanations.append(
                    "Rejected candidate "
                    f"{h.hypothesis_id}: method_hint={h.method}, conf={h.confidence:.3f}, "
                    f"endpoints=(e{h.endpoint_a_id},e{h.endpoint_b_id}), reason=endpoint_conflict_or_lower_rank."
                )

    summary = (
        f"Applied {len(result.additions)} restoration segments from {len(hypotheses)} hypotheses. "
        f"Forced low-confidence links: {int(sum(1 for a in result.additions if a.is_forced))}."
    )

    surrogate_rules: List[str] = []
    if cfg.include_surrogate_rules:
        surrogate_rules.append(
            "Rule R1: choose EFD for one-gap contours when confidence >= threshold and EFD features exist."
        )
        surrogate_rules.append(
            "Rule R2: choose symmetry completion for one-gap contours when symmetry evidence dominates and EFD is weak."
        )
        surrogate_rules.append(
            "Rule R3: use Bezier bridging for multi-gap or line-network continuation with directional compatibility."
        )
        surrogate_rules.append(
            "Rule R4: enforce one-endpoint-one-connection unless forced fallback is required to avoid skipped endpoints."
        )

    if vocab is not None and vocab.entries:
        surrogate_rules.append(
            f"Vocabulary active: {len(vocab.entries)} reference descriptors loaded for optional prior guidance."
        )

    return ExplanationReport(
        summary=summary,
        explanations=explanations,
        surrogate_rules=surrogate_rules,
    )
