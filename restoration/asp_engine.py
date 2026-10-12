"""Phase 4 — ASP Decision Engine.

Encodes scored candidates as ASP facts, solves with clingo,
and decodes the optimal connection set. The solve is deterministic:
the same candidates always give the same connections.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from restoration.candidates import ConnectionCandidate

RULES_PATH = os.path.join(os.path.dirname(__file__), "asp_rules", "restoration.lp")


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

def _to_int_score(value: float, scale: int = 100) -> int:
    """Scale a float score to an integer in [0, scale]."""
    return max(0, min(scale, int(round(value * scale))))


def encode_facts(
    candidates: List[ConnectionCandidate],
) -> str:
    """Convert scored candidates into ASP fact strings."""
    lines: List[str] = []
    end_map = {"start": 0, "end": 1}
    path_strength = _path_self_closure_strength_map(candidates)

    # Candidate facts
    for c in candidates:
        ea = end_map[c.ep_a.end]
        eb = end_map[c.ep_b.end]
        score_int = _to_int_score(c.score)
        lines.append(
            f"candidate({c.id},{c.ep_a.path_index},{ea},"
            f"{c.ep_b.path_index},{eb},{score_int})."
        )
        lines.append(f"candidate_endpoint({c.id},{c.ep_a.endpoint_id},{c.ep_b.endpoint_id}).")
        if c.same_path_closure:
            lines.append(f"candidate_self_closure({c.id}).")

        utility = _candidate_effective_utility(c, path_strength)
        lines.append(f"candidate_utility({c.id},{utility}).")

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Solving
# ═══════════════════════════════════════════════════════════════════════════

def _path_self_closure_strength_map(
    candidates: List[ConnectionCandidate],
) -> Dict[int, int]:
    """Compute per-path self-closure support buckets used by ASP soft suppression."""
    best_self_closure_by_path: Dict[int, float] = {}
    for c in candidates:
        if not c.same_path_closure:
            continue
        pidx = int(c.ep_a.path_index)
        best_self_closure_by_path[pidx] = max(
            best_self_closure_by_path.get(pidx, -1e9),
            float(c.score),
        )

    buckets: Dict[int, int] = {}
    for pidx, score in best_self_closure_by_path.items():
        if score >= 0.60:
            buckets[pidx] = 2
        elif score >= 0.45:
            buckets[pidx] = 1
    return buckets


def _candidate_effective_utility(
    candidate: ConnectionCandidate,
    path_strength: Dict[int, int],
) -> int:
    """Compute candidate utility equivalent to current ASP objective terms."""
    utility = int(_to_int_score(float(candidate.score)))
    relaxed_tier2_extension = bool(candidate.relaxed_tier2_extension)

    # Soft rewards from weak constraints with negative weights.
    closure_flag = bool(
        candidate.same_path_closure
        or (float(candidate.score) > 0.45 and float(candidate.distance) < 30.0)
    )
    if closure_flag:
        utility += 3
    if candidate.same_path_closure:
        utility += 2

    # Cross-shape suppression from path-level self-closure support.
    if int(candidate.ep_b.path_index) != int(candidate.ep_a.path_index):
        touched = [int(candidate.ep_a.path_index), int(candidate.ep_b.path_index)]
        for pidx in touched:
            bucket = int(path_strength.get(pidx, 0))
            if bucket == 1:
                utility -= 2
            elif bucket == 2:
                utility -= 4

    direction_ab = candidate.ep_b.position - candidate.ep_a.position
    n = float(np.linalg.norm(direction_ab))
    if n > 1e-12:
        cont_raw = float(np.dot(candidate.ep_a.tangent, direction_ab / n))
    else:
        cont_raw = 0.0
    cont_int = int(_to_int_score(max(0.0, cont_raw)))
    if cont_int < 40 and not relaxed_tier2_extension:
        utility -= 2

    ext_quality = int(_to_int_score(float(candidate.extension_quality)))
    if (
        str(candidate.scenario) == "extension_intersection"
        and bool(candidate.same_path_closure)
        and ext_quality < 45
    ):
        utility -= 3

    if int(candidate.tier) == 2:
        utility -= 1
        if relaxed_tier2_extension:
            utility += 1

    return utility


def _prune_strictly_dominated_candidates(
    candidates: List[ConnectionCandidate],
) -> Tuple[List[ConnectionCandidate], int]:
    """Drop strictly dominated alternatives sharing the same endpoint pair.

    Conservative rule: never prune groups containing same-path closure candidates,
    because those facts influence path-level soft suppression for other candidates.
    """
    if len(candidates) <= 1:
        return list(candidates), 0

    path_strength = _path_self_closure_strength_map(candidates)
    groups: Dict[Tuple[int, int], List[ConnectionCandidate]] = defaultdict(list)
    for c in candidates:
        ea, eb = c.ep_a.endpoint_id, c.ep_b.endpoint_id
        key = (ea, eb) if ea <= eb else (eb, ea)
        groups[key].append(c)

    keep_ids: Set[int] = set()
    for group in groups.values():
        if len(group) == 1:
            keep_ids.add(int(group[0].id))
            continue

        if any(bool(c.same_path_closure) for c in group):
            for c in group:
                keep_ids.add(int(c.id))
            continue

        best_utility = max(_candidate_effective_utility(c, path_strength) for c in group)
        for c in group:
            if _candidate_effective_utility(c, path_strength) >= best_utility:
                keep_ids.add(int(c.id))

    pruned = [c for c in candidates if int(c.id) not in keep_ids]
    kept = [c for c in candidates if int(c.id) in keep_ids]
    return kept, len(pruned)


def _solve_facts(
    facts: str,
    rules_path: str,
    max_solutions: int,
    timeout_s: float = 30.0,
) -> List[int]:
    """Run clingo on one fact bundle and return accepted candidate IDs."""
    import clingo

    ctl = clingo.Control(_control_args(max_solutions))

    ctl.load(rules_path)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    accepted_ids: List[int] = []

    def on_model(model):
        nonlocal accepted_ids
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
) -> Tuple[List[int], Dict[str, Any]]:
    """Solve for the optimal set of connections; return accepted IDs + stats.

    All candidates of an image are solved as one program: the reward for
    connecting both ends of a path spans endpoints that an endpoint-conflict
    partition would put into separate programs.
    """
    start_total = time.perf_counter()

    working, pruned_count = _prune_strictly_dominated_candidates(list(candidates))

    t_encode = time.perf_counter()
    facts = encode_facts(working)
    encoding_time_s = time.perf_counter() - t_encode

    t_solve = time.perf_counter()
    accepted_ids = _solve_facts(facts, rules_path, max_solutions, timeout_s)
    solving_time_s = time.perf_counter() - t_solve

    summary: Dict[str, Any] = {
        "candidate_count_input": len(candidates),
        "candidate_count_after_pruning": len(working),
        "candidate_count_pruned": pruned_count,
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
