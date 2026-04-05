from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ASPConfig:
    """Configuration for ASP solving."""

    use_asp: bool = True
    use_clingo_package: bool = True
    max_hypotheses: int = 64
    fallback_on_solver_failure: bool = True


@dataclass
class RankedHypothesis:
    """A ranked restoration decision emitted by ASP or fallback logic."""

    rank: int
    hypothesis_id: str
    gap_id: int
    endpoint_a_id: int
    endpoint_b_id: int
    path_a: int
    path_b: int
    action: str
    method: str
    confidence: float
    score: float
    is_forced: bool = False
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class _CandidateFact:
    gap_id: int
    endpoint_a_id: int
    endpoint_b_id: int
    path_a: int = -1
    path_b: int = -1
    kind: str = "bridge"
    method_hint: str = "bezier"
    confidence: float = 0.0
    weight: int = 0
    is_forced: bool = False


def _parse_int_token(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise ValueError(f"Expected token prefix {prefix!r}, got {token!r}")
    return int(token[len(prefix):])


def _parse_fact_args(line: str) -> List[str]:
    start = line.find("(")
    end = line.rfind(")")
    if start < 0 or end < 0 or end <= start:
        return []
    payload = line[start + 1:end]
    return [x.strip() for x in payload.split(",")]


def _parse_candidates(facts_str: str) -> Dict[int, _CandidateFact]:
    candidates: Dict[int, _CandidateFact] = {}

    for raw in facts_str.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue

        if line.startswith("candidate("):
            args = _parse_fact_args(line)
            if len(args) != 3:
                continue
            gid = _parse_int_token(args[0], "g")
            e1 = _parse_int_token(args[1], "e")
            e2 = _parse_int_token(args[2], "e")
            candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=e1, endpoint_b_id=e2))
            continue

        if line.startswith("candidate_paths("):
            args = _parse_fact_args(line)
            if len(args) != 3:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            cand.path_a = int(args[1])
            cand.path_b = int(args[2])
            continue

        if line.startswith("candidate_kind("):
            args = _parse_fact_args(line)
            if len(args) != 2:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            cand.kind = args[1]
            continue

        if line.startswith("candidate_method_hint("):
            args = _parse_fact_args(line)
            if len(args) != 2:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            cand.method_hint = args[1]
            continue

        if line.startswith("candidate_confidence("):
            args = _parse_fact_args(line)
            if len(args) != 2:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            conf_val = float(args[1])
            cand.confidence = conf_val / 1000.0 if conf_val > 1.0 else conf_val
            continue

        if line.startswith("candidate_weight("):
            args = _parse_fact_args(line)
            if len(args) != 2:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            cand.weight = int(args[1])
            if cand.confidence <= 0.0 and cand.weight > 0:
                cand.confidence = min(1.0, cand.weight / 1000.0)
            continue

        if line.startswith("candidate_forced("):
            args = _parse_fact_args(line)
            if len(args) != 1:
                continue
            gid = _parse_int_token(args[0], "g")
            cand = candidates.setdefault(gid, _CandidateFact(gap_id=gid, endpoint_a_id=-1, endpoint_b_id=-1))
            cand.is_forced = True

    # Drop partial entries.
    return {
        gid: c
        for gid, c in candidates.items()
        if c.endpoint_a_id >= 0 and c.endpoint_b_id >= 0
    }


def _heuristic_select(candidates: Dict[int, _CandidateFact]) -> List[int]:
    ranked = sorted(
        candidates.values(),
        key=lambda c: (
            -c.confidence,
            -c.weight,
            c.is_forced,
            c.gap_id,
        ),
    )

    selected: List[int] = []
    occupied: set = set()
    for cand in ranked:
        if cand.endpoint_a_id in occupied or cand.endpoint_b_id in occupied:
            continue
        occupied.add(cand.endpoint_a_id)
        occupied.add(cand.endpoint_b_id)
        selected.append(cand.gap_id)

    return selected


def _solve_with_clingo(facts_str: str, cfg: ASPConfig) -> Optional[List[int]]:
    try:
        import clingo  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("clingo package unavailable: %s", exc)
        return None

    logic = r"""
        ep_in(G,E) :- candidate(G,E,_).
        ep_in(G,E) :- candidate(G,_,E).

        { choose(G) : candidate(G,_,_) }.

        :- choose(G1), choose(G2), G1 < G2, ep_in(G1,E), ep_in(G2,E).

        #maximize { W,G : choose(G), candidate_weight(G,W) }.

        #show choose/1.
    """

    selected: List[int] = []
    # Enumerate optimization models and keep the last (optimal) model.
    control = clingo.Control(["-n", "0", "--opt-mode=optN"])
    control.add("base", [], facts_str + "\n" + logic)
    control.ground([("base", [])])

    with control.solve(yield_=True) as handle:
        model = None
        for m in handle:
            model = m
        if model is None:
            return []

        atoms = model.symbols(shown=True)
        for atom in atoms:
            if atom.name != "choose" or len(atom.arguments) != 1:
                continue
            raw = str(atom.arguments[0])
            if raw.startswith("g"):
                try:
                    selected.append(int(raw[1:]))
                except ValueError:
                    continue

    selected.sort()
    return selected


def run_asp(facts_str: str, config: Optional[ASPConfig] = None) -> List[RankedHypothesis]:
    """Run ASP-based inference to select non-conflicting restoration actions."""
    cfg = config or ASPConfig()
    candidates = _parse_candidates(facts_str)
    if not candidates:
        return []

    selected_ids: Optional[List[int]]
    if cfg.use_asp and cfg.use_clingo_package:
        selected_ids = _solve_with_clingo(facts_str, cfg)
    else:
        selected_ids = None

    if selected_ids is None:
        if not cfg.fallback_on_solver_failure:
            return []
        selected_ids = _heuristic_select(candidates)

    selected_candidates = [candidates[g] for g in selected_ids if g in candidates]
    selected_candidates = sorted(
        selected_candidates,
        key=lambda c: (
            -c.confidence,
            -c.weight,
            c.gap_id,
        ),
    )

    hypotheses: List[RankedHypothesis] = []
    for rank, cand in enumerate(selected_candidates, start=1):
        score = 1.0 - cand.confidence
        hypotheses.append(
            RankedHypothesis(
                rank=rank,
                hypothesis_id=f"h{rank}",
                gap_id=cand.gap_id,
                endpoint_a_id=cand.endpoint_a_id,
                endpoint_b_id=cand.endpoint_b_id,
                path_a=cand.path_a,
                path_b=cand.path_b,
                action="connect_endpoints",
                method=cand.method_hint,
                confidence=float(cand.confidence),
                score=float(score),
                is_forced=cand.is_forced,
                metadata={
                    "weight": float(cand.weight),
                    "is_single_gap": 1.0 if cand.kind == "single_gap" else 0.0,
                },
            )
        )

    return hypotheses[: cfg.max_hypotheses]
