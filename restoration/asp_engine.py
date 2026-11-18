"""Phase 4 — ASP Decision Engine.

Encodes scored candidates as ASP facts, solves with clingo,
and decodes the optimal connection set. The solve is deterministic:
the same candidates always give the same connections.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from restoration.candidates import ConnectionCandidate
from restoration.constraints import HINT_BONUS, ConstraintError, ResolvedConstraints

RULES_PATH = os.path.join(os.path.dirname(__file__), "asp_rules", "restoration.lp")

# Integer utility units per unit of log-odds above the acceptance threshold.
UTILITY_SCALE = 10


def _control_args(max_solutions: int) -> List[str]:
    """clingo arguments: enumerate optimal models on a single thread.

    Core-guided optimization (usc) proves the optimum in milliseconds on these
    programs, and one thread makes the choice among equally good answers
    repeatable; with several threads it depends on which one finishes first.
    """
    return [f"--models={max_solutions}", "--opt-mode=optN", "--opt-strategy=usc"]


# ═══════════════════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════════════════

def utility(score: float) -> int:
    """Integer utility of a candidate whose log-odds exceed the threshold by *score*."""
    return int(round(score * UTILITY_SCALE))


def encode_facts(
    candidates: Sequence[ConnectionCandidate], constraints: Optional[ResolvedConstraints] = None,
) -> str:
    """Convert scored candidates (and the user's constraints, if any) into ASP fact strings."""
    hinted = constraints.pairs("hint") if constraints else set()

    def value(c: ConnectionCandidate) -> int:
        if frozenset((c.ep_a.endpoint_id, c.ep_b.endpoint_id)) in hinted:
            return max(utility(c.score), 0) + HINT_BONUS
        return utility(c.score)

    lines = [f"candidate({c.id},{c.ep_a.endpoint_id},{c.ep_b.endpoint_id},{value(c)})." for c in candidates]
    if constraints:
        lines += [f"forbid_pair({a},{b})." for a, b in constraints.forbid]
        lines += [f"force_pair({a},{b})." for a, b in constraints.force]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Solving
# ═══════════════════════════════════════════════════════════════════════════

def _solve_facts(
    facts: str,
    rules_path: str,
    max_solutions: int,
    timeout_s: float = 30.0,
    found: Optional[List[bool]] = None,
) -> List[int]:
    """Run clingo on one fact bundle and return accepted candidate IDs.

    *found*, if given, gets True appended when the solver found any answer
    (an empty answer and no answer both return []).
    """
    import clingo

    ctl = clingo.Control(_control_args(max_solutions))

    ctl.load(rules_path)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    accepted_ids: List[int] = []

    def on_model(model):
        nonlocal accepted_ids
        if found is not None and not found:
            found.append(True)
        accepted_ids = []
        for atom in model.symbols(shown=True):
            if atom.name == "accept" and len(atom.arguments) == 1:
                accepted_ids.append(atom.arguments[0].number)

    if timeout_s > 0:
        with ctl.solve(on_model=on_model, async_=True) as handle:
            finished = handle.wait(timeout_s)
            if not finished:
                handle.cancel()
    else:
        ctl.solve(on_model=on_model)

    return sorted(accepted_ids)


def solve_candidates(
    candidates: List[ConnectionCandidate],
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
    timeout_s: float = 30.0,
    constraints: Optional[ResolvedConstraints] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Solve for the optimal set of connections; return accepted IDs + stats.

    A candidate at or below the acceptance threshold has no positive utility
    and cannot be part of an optimal answer, so only the others are encoded,
    plus the candidates of pairs the user forced or hinted (*constraints*).
    """
    start_total = time.perf_counter()

    wanted = (constraints.pairs("force") | constraints.pairs("hint")) if constraints else set()
    working = [c for c in candidates
               if utility(c.score) > 0 or frozenset((c.ep_a.endpoint_id, c.ep_b.endpoint_id)) in wanted]

    t_encode = time.perf_counter()
    facts = encode_facts(working, constraints)
    encoding_time_s = time.perf_counter() - t_encode

    t_solve = time.perf_counter()
    found: List[bool] = []
    accepted_ids = _solve_facts(facts, rules_path, max_solutions, timeout_s, found) if working else []
    solving_time_s = time.perf_counter() - t_solve
    if constraints and constraints.force and not found:
        raise ConstraintError("the forced bridges cannot all be drawn together (no answer satisfies them)")

    summary: Dict[str, Any] = {
        "candidate_count_input": len(candidates),
        "candidate_count_above_threshold": len(working),
        "fact_lines": len(facts.splitlines()),
        "fact_bytes": len(facts.encode("utf-8")),
        "accepted_id_count_before_decode": len(accepted_ids),
        "encoding_time_s": encoding_time_s,
        "solving_time_s": solving_time_s,
        "total_solve_s": time.perf_counter() - start_total,
        "control_args": _control_args(max_solutions),
    }
    return accepted_ids, summary


# ═══════════════════════════════════════════════════════════════════════════
# Decoding
# ═══════════════════════════════════════════════════════════════════════════

def decode_solution(
    accepted_ids: List[int],
    candidates: List[ConnectionCandidate],
) -> List[ConnectionCandidate]:
    """Map accepted IDs back to ConnectionCandidate objects."""
    id_set = set(accepted_ids)
    return [c for c in candidates if c.id in id_set]
