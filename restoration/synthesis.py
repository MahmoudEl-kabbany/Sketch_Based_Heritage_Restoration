"""Phase 5 — Synthesis.

Merge the bridges of the accepted candidates into the restored path set.
A candidate's bridge is built when the candidate is generated, so that what
is scored is exactly what is drawn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bezier_curves.bezier import BezierPath, BezierSegment
from restoration.candidates import ConnectionCandidate
from restoration.geometry import polyline_length


def synthesize_bridges(
    accepted: List[ConnectionCandidate],
) -> List[BezierSegment]:
    """The bridge segments of the accepted candidates, in order."""
    return [segment for c in accepted for segment in c.bridge_bezier]


# ═══════════════════════════════════════════════════════════════════════════
# Path merging
# ═══════════════════════════════════════════════════════════════════════════

def merge_restored_paths(
    original_paths: List[BezierPath],
    accepted: List[ConnectionCandidate],
) -> List[BezierPath]:
    """Merge the accepted candidates' bridge segments into the original path set.

    For each accepted connection:
      - If the bridge connects path A's end to path B's start,
        merge them into one path: A.segments + bridge + B.segments
      - Detect if the merged path forms a closed loop

    Returns the new set of paths (some merged, some untouched).
    """
    # A component is a chain of segments plus the tokens of its two free ends.
    # A token is the (path_index, end) of an original free path end; it is None
    # for an end on a junction, for closed paths, and once a bridge has used it.
    active: Dict[int, Dict[str, Any]] = {}
    for i, path in enumerate(original_paths):
        is_open = not path.is_closed and bool(path.segments)
        active[i] = {
            "segments": list(path.segments),
            "start_token": (i, "start") if is_open and path.start_free else None,
            "end_token": (i, "end") if is_open and path.end_free else None,
        }

    def component_with(token: Tuple[int, str]) -> Optional[int]:
        return next(
            (cid for cid, comp in active.items()
             if token in (comp["start_token"], comp["end_token"])),
            None,
        )

    def orient(comp: Dict[str, Any], side: str, token: Tuple[int, str]) -> None:
        """Reverse *comp* if needed so that *token* is its *side* end."""
        if comp[side] != token:
            comp["segments"] = _reverse_segments(comp["segments"])
            comp["start_token"], comp["end_token"] = comp["end_token"], comp["start_token"]

    for c in sorted(accepted, key=lambda c: (-c.score, c.distance, c.id)):
        token_a = (c.ep_a.path_index, c.ep_a.end)
        token_b = (c.ep_b.path_index, c.ep_b.end)
        id_a, id_b = component_with(token_a), component_with(token_b)
        if id_a is None or id_b is None or token_a == token_b:
            continue

        comp_a = active[id_a]
        orient(comp_a, "end_token", token_a)
        if id_a == id_b:
            # The bridge closes this component onto itself.
            comp_a["segments"] = comp_a["segments"] + list(c.bridge_bezier)
            comp_a["start_token"] = comp_a["end_token"] = None
            continue

        comp_b = active.pop(id_b)
        orient(comp_b, "start_token", token_b)
        active[id_a] = {
            "segments": comp_a["segments"] + list(c.bridge_bezier) + comp_b["segments"],
            "start_token": comp_a["start_token"],
            "end_token": comp_b["end_token"],
        }

    result: List[BezierPath] = []
    for comp in active.values():
        segs: List[BezierSegment] = comp["segments"]
        if not segs:
            continue

        start_pt = segs[0].control_points[0]
        end_pt = segs[-1].control_points[3]
        path_len = polyline_length(np.vstack([seg.sample(12) for seg in segs]))
        closure_tolerance = max(5.0, 0.01 * path_len)
        is_closed = float(np.linalg.norm(start_pt - end_pt)) < closure_tolerance

        result.append(BezierPath(
            segments=segs,
            is_closed=is_closed,
            source_type="restored",
            start_free=comp["start_token"] is not None,
            end_free=comp["end_token"] is not None,
        ))

    return result


def _reverse_segments(segments: List[BezierSegment]) -> List[BezierSegment]:
    """Reversed copy: segment order and each segment's control points flipped."""
    return [
        BezierSegment(control_points=seg.control_points[::-1].copy(), source_type=seg.source_type)
        for seg in reversed(segments)
    ]
