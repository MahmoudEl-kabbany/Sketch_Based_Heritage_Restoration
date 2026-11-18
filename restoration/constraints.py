"""User constraints on the restoration: forbid, force or hint bridges between stroke ends.

A user (through scripts/refine.py, or a JSON file) corrects the pipeline by
naming pairs of free stroke ends by position:

    {
      "image": "ankh_damaged.png",
      "forbid": [ {"a": [412.0, 96.0], "b": [450.5, 98.0]} ],
      "force":  [ {"a": [100.0, 210.0], "b": [130.0, 214.0]} ],
      "hint":   [ {"a": [300.0, 50.0],  "b": [360.0, 40.0]} ],
      "tolerance_px": 8.0
    }

Positions, not candidate ids, because ids depend on the order candidates are
generated in and are not stable across runs or code changes. Each position is
resolved to the nearest free end within tolerance_px; if there is none, that is
an error, not a silent no-op.

    forbid  no bridge may join the two ends (an ASP constraint);
    force   a bridge must join them (an ASP constraint; a candidate is
            synthesized when the pipeline proposed none);
    hint    a bridge between them is worth at least HINT_BONUS to the solver,
            however poorly it scores: it is drawn unless a force needs one of
            its ends, or the bridges it displaces are together worth more.

The solver re-optimizes globally: forbidding a bridge frees its two ends for
other partners, and forcing one takes its ends away from their old partners.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from restoration.candidates import ConnectionCandidate, sample_bridge, smooth_bridge
from restoration.extraction import EndpointInfo

KINDS = ("forbid", "force", "hint")
DEFAULT_TOLERANCE_PX = 8.0
# Utility a hinted candidate is worth at least (its own utility, if positive, plus
# this): 100 units = 10 log-odds above the threshold, more than nearly any
# candidate scores on its own, so a hint wins unless it conflicts with a force.
HINT_BONUS = 100


class ConstraintError(ValueError):
    """A constraint that cannot be resolved or satisfied."""


@dataclass
class ResolvedConstraints:
    """Constraints as pairs of endpoint ids, ready for the solver."""

    forbid: List[Tuple[int, int]] = field(default_factory=list)
    force: List[Tuple[int, int]] = field(default_factory=list)
    hint: List[Tuple[int, int]] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)   # for the report
    synthesized: List[int] = field(default_factory=list)          # ids of candidates made for force/hint

    def __bool__(self) -> bool:
        return bool(self.forbid or self.force or self.hint)

    def pairs(self, kind: str) -> set:
        return {frozenset(pair) for pair in getattr(self, kind)}


def load_constraints(source: Union[None, str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A constraints dict from a dict, a JSON file path, or None."""
    if source is None:
        return None
    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            source = json.load(f)
    unknown = set(source) - set(KINDS) - {"image", "tolerance_px"}
    if unknown:
        raise ConstraintError(f"unknown constraint keys: {sorted(unknown)}")
    return source


def _nearest(endpoints: Sequence[EndpointInfo], xy: Sequence[float], tolerance: float, what: str) -> EndpointInfo:
    if not endpoints:
        raise ConstraintError(f"{what}: the drawing has no free stroke ends")
    positions = np.array([ep.position for ep in endpoints])
    distances = np.linalg.norm(positions - np.asarray(xy, dtype=float), axis=1)
    best = int(np.argmin(distances))
    if distances[best] > tolerance:
        raise ConstraintError(
            f"{what}: no free stroke end within {tolerance:g} px of {list(map(float, xy))} "
            f"(nearest is {distances[best]:.1f} px away at {positions[best].round(1).tolist()})")
    return endpoints[best]


def resolve(constraints: Optional[Dict[str, Any]], endpoints: Sequence[EndpointInfo]) -> ResolvedConstraints:
    """Map every constrained position to its free stroke end; check the constraints are consistent."""
    resolved = ResolvedConstraints()
    if not constraints:
        return resolved
    tolerance = float(constraints.get("tolerance_px", DEFAULT_TOLERANCE_PX))
    for kind in KINDS:
        for n, item in enumerate(constraints.get(kind, []) or []):
            what = f"{kind}[{n}]"
            ep_a = _nearest(endpoints, item["a"], tolerance, what + ".a")
            ep_b = _nearest(endpoints, item["b"], tolerance, what + ".b")
            if ep_a.endpoint_id == ep_b.endpoint_id:
                raise ConstraintError(f"{what}: both positions resolve to the same stroke end {ep_a.endpoint_id}")
            getattr(resolved, kind).append((ep_a.endpoint_id, ep_b.endpoint_id))
            resolved.records.append({
                "kind": kind,
                "a": [float(v) for v in item["a"]], "b": [float(v) for v in item["b"]],
                "endpoint_a": int(ep_a.endpoint_id), "endpoint_b": int(ep_b.endpoint_id),
            })

    forced_ends = [e for pair in resolved.force for e in pair]
    if len(forced_ends) != len(set(forced_ends)):
        raise ConstraintError("two forced bridges share a stroke end; a stroke end can take only one bridge")
    clash = resolved.pairs("force") & resolved.pairs("forbid")
    if clash:
        raise ConstraintError(f"pairs both forced and forbidden: {[sorted(p) for p in clash]}")
    return resolved


def add_missing_candidates(
    candidates: List[ConnectionCandidate], resolved: ResolvedConstraints, endpoints: Sequence[EndpointInfo],
) -> List[ConnectionCandidate]:
    """Append a smooth candidate for every forced or hinted pair the pipeline did not propose."""
    by_id = {ep.endpoint_id: ep for ep in endpoints}
    existing = {frozenset((c.ep_a.endpoint_id, c.ep_b.endpoint_id)) for c in candidates}
    next_id = max((c.id for c in candidates), default=-1) + 1
    for a, b in resolved.force + resolved.hint:
        if frozenset((a, b)) in existing:
            continue
        ep_a, ep_b = by_id[a], by_id[b]
        segments = smooth_bridge(ep_a, ep_b)
        candidates.append(ConnectionCandidate(
            id=next_id, ep_a=ep_a, ep_b=ep_b, shape="smooth",
            bridge_points=sample_bridge(segments), bridge_bezier=segments,
            distance=float(np.linalg.norm(ep_b.position - ep_a.position)),
            same_path_closure=ep_a.path_index == ep_b.path_index,
        ))
        resolved.synthesized.append(next_id)
        existing.add(frozenset((a, b)))
        next_id += 1
    for record in resolved.records:
        pair = frozenset((record["endpoint_a"], record["endpoint_b"]))
        record["candidates"] = sorted(c.id for c in candidates
                                      if frozenset((c.ep_a.endpoint_id, c.ep_b.endpoint_id)) == pair)
        record["synthesized"] = any(i in resolved.synthesized for i in record["candidates"])
    return candidates


def user_role(candidate: ConnectionCandidate, resolved: Optional[ResolvedConstraints]) -> Optional[str]:
    """"forced", "hinted" or None: how the user constrained this candidate's pair."""
    if not resolved:
        return None
    pair = frozenset((candidate.ep_a.endpoint_id, candidate.ep_b.endpoint_id))
    if pair in resolved.pairs("force"):
        return "forced"
    if pair in resolved.pairs("hint"):
        return "hinted"
    return None
