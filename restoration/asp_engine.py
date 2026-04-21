"""Phase 4 — ASP Decision Engine.

Encodes scored candidates as ASP facts, solves with clingo,
and decodes the optimal connection set.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from restoration.extraction import EndpointInfo
from restoration.candidates import ConnectionCandidate

RULES_PATH = os.path.join(os.path.dirname(__file__), "asp_rules", "restoration.lp")
_CLINGO_OPTION_ARGS_CACHE: Optional[List[str]] = None


def _base_control_args(max_solutions: int) -> List[str]:
    """Build base clingo control args with stable optimization semantics."""
    return [
        f"--models={max_solutions}",
        "--opt-mode=optN",
    ]


def _discover_optional_control_args(base_args: List[str]) -> List[str]:
    """Probe optional clingo args for runtime speedup with compatibility guards."""
    import clingo

    conservative = os.environ.get("RESTORATION_CLINGO_CONSERVATIVE", "0") == "1"
    if conservative:
        return []

    workers = max(2, min(8, int(os.cpu_count() or 2)))
    thread_options = [
        f"--threads={workers}",
        f"--parallel-mode={workers}",
        f"-t{workers}",
    ]

    for opt in thread_options:
        try:
            clingo.Control(base_args + [opt])
            return [opt]
        except RuntimeError:
            continue

    return []


def _build_control_args(max_solutions: int) -> List[str]:
    """Return base + compatible optional clingo args (cached after first probe)."""
    global _CLINGO_OPTION_ARGS_CACHE
    base_args = _base_control_args(max_solutions)
    if _CLINGO_OPTION_ARGS_CACHE is None:
        _CLINGO_OPTION_ARGS_CACHE = _discover_optional_control_args(base_args)
    return base_args + list(_CLINGO_OPTION_ARGS_CACHE)


# ═══════════════════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════════════════

def _to_int_score(value: float, scale: int = 100) -> int:
    """Scale a float score to an integer in [0, scale]."""
    return max(0, min(scale, int(round(value * scale))))


def _endpoint_token_id(ep: EndpointInfo) -> int:
    """Return a stable endpoint token ID for ASP occupancy constraints."""
    if ep.endpoint_id >= 0:
        return int(ep.endpoint_id)
    # Fallback for synthetic endpoints not coming from extraction.
    end_map = {"start": 0, "end": 1}
    return int(ep.path_index * 2 + end_map.get(ep.end, 0))


def encode_facts(
    candidates: List[ConnectionCandidate],
    endpoints: List[EndpointInfo],
) -> str:
    """Convert scored candidates and endpoints into ASP fact strings."""
    lines: List[str] = []
    end_map = {"start": 0, "end": 1}
    endpoint_token_cache: Dict[Tuple[int, int, str], int] = {}
    for ep in endpoints:
        key = (int(getattr(ep, "endpoint_id", -1)), int(ep.path_index), str(ep.end))
        endpoint_token_cache[key] = _endpoint_token_id(ep)

    def _cached_endpoint_token_id(ep: EndpointInfo) -> int:
        key = (int(getattr(ep, "endpoint_id", -1)), int(ep.path_index), str(ep.end))
        cached = endpoint_token_cache.get(key)
        if cached is not None:
            return cached
        token_id = _endpoint_token_id(ep)
        endpoint_token_cache[key] = token_id
        return token_id

    path_strength = _path_self_closure_strength_map(candidates)

    # Candidate facts
    for c in candidates:
        ea = end_map[c.ep_a.end]
        eb = end_map[c.ep_b.end]
        ea_id = _cached_endpoint_token_id(c.ep_a)
        eb_id = _cached_endpoint_token_id(c.ep_b)
        score_int = _to_int_score(c.score)
        lines.append(
            f"candidate({c.id},{c.ep_a.path_index},{ea},"
            f"{c.ep_b.path_index},{eb},{score_int})."
        )
        lines.append(f"candidate_endpoint({c.id},{ea_id},{eb_id}).")
        if getattr(c, "same_path_closure", False):
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
        if not getattr(c, "same_path_closure", False):
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
    relaxed_tier2_extension = bool(getattr(candidate, "relaxed_tier2_extension", False))

    # Soft rewards from weak constraints with negative weights.
    closure_flag = bool(
        getattr(candidate, "same_path_closure", False)
        or (float(candidate.score) > 0.45 and float(candidate.distance) < 30.0)
    )
    if closure_flag:
        utility += 3
    if getattr(candidate, "same_path_closure", False):
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

    ext_quality = int(_to_int_score(float(getattr(candidate, "extension_quality", 0.0))))
    if (
        str(candidate.scenario) == "extension_intersection"
        and bool(getattr(candidate, "same_path_closure", False))
        and ext_quality < 45
    ):
        utility -= 3

    if int(getattr(candidate, "tier", 1)) == 2:
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
        ea = int(_endpoint_token_id(c.ep_a))
        eb = int(_endpoint_token_id(c.ep_b))
        key = (ea, eb) if ea <= eb else (eb, ea)
        groups[key].append(c)

    keep_ids: Set[int] = set()
    for group in groups.values():
        if len(group) == 1:
            keep_ids.add(int(group[0].id))
            continue

        if any(bool(getattr(c, "same_path_closure", False)) for c in group):
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


def _split_candidates_by_endpoint_component(
    candidates: List[ConnectionCandidate],
) -> List[List[ConnectionCandidate]]:
    """Split candidates by connected components in endpoint-conflict space."""
    if len(candidates) <= 1:
        return [list(candidates)] if candidates else []

    adjacency: Dict[int, Set[int]] = defaultdict(set)
    endpoint_to_candidate_indexes: Dict[int, List[int]] = defaultdict(list)

    for idx, c in enumerate(candidates):
        ea = int(_endpoint_token_id(c.ep_a))
        eb = int(_endpoint_token_id(c.ep_b))
        adjacency[ea].add(eb)
        adjacency[eb].add(ea)
        endpoint_to_candidate_indexes[ea].append(idx)
        endpoint_to_candidate_indexes[eb].append(idx)

    visited: Set[int] = set()
    components: List[List[ConnectionCandidate]] = []

    for start in list(adjacency.keys()):
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        candidate_indexes: Set[int] = set()

        while stack:
            node = stack.pop()
            for idx in endpoint_to_candidate_indexes.get(node, []):
                candidate_indexes.add(int(idx))
            for nxt in adjacency.get(node, set()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)

        component = [candidates[idx] for idx in sorted(candidate_indexes)]
        if component:
            components.append(component)

    return components


def _solve_facts(
    facts: str,
    rules_path: str,
    max_solutions: int,
) -> List[int]:
    """Run clingo on one fact bundle and return accepted candidate IDs."""
    import clingo

    ctl = clingo.Control(_build_control_args(max_solutions))

    if os.path.isfile(rules_path):
        ctl.load(rules_path)
    else:
        ctl.add("base", [], _fallback_rules())

    ctl.add("base", [], facts)
    ctl.ground([("base", [])])

    accepted_ids: List[int] = []

    def on_model(model):
        nonlocal accepted_ids
        accepted_ids = []
        for atom in model.symbols(shown=True):
            if atom.name == "accept" and len(atom.arguments) == 1:
                accepted_ids.append(atom.arguments[0].number)

    ctl.solve(on_model=on_model)
    return sorted(accepted_ids)


def solve_partitioned(
    candidates: List[ConnectionCandidate],
    endpoints: List[EndpointInfo],
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
    enable_component_partition: bool = True,
    enable_dominance_pruning: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """Solve ASP in independent endpoint components and return accepted IDs + stats."""
    start_total = time.perf_counter()

    working = list(candidates)
    pruned_count = 0
    if enable_dominance_pruning:
        working, pruned_count = _prune_strictly_dominated_candidates(working)

    components: List[List[ConnectionCandidate]]
    if enable_component_partition:
        components = _split_candidates_by_endpoint_component(working)
    else:
        components = [working] if working else []

    raw_component_count = int(len(components))
    raw_largest_component = int(max((len(c) for c in components), default=0))
    total_candidates = int(len(working))
    partition_fallback_reason = ""

    if enable_component_partition and len(components) > 1 and total_candidates > 0:
        dominant_ratio = float(raw_largest_component / max(total_candidates, 1))
        if dominant_ratio >= 0.80:
            # One dominant component means repeated grounding overhead can outweigh
            # any partitioning benefit; fall back to one monolithic solve.
            components = [working]
            partition_fallback_reason = "dominant_component"

    accepted_ids: List[int] = []
    encoding_time_s = 0.0
    solving_time_s = 0.0
    fact_lines = 0
    fact_bytes = 0

    for component_candidates in components:
        t_encode = time.perf_counter()
        facts = encode_facts(component_candidates, endpoints)
        encoding_time_s += (time.perf_counter() - t_encode)

        fact_lines += int(len(facts.splitlines()))
        fact_bytes += int(len(facts.encode("utf-8")))

        t_solve = time.perf_counter()
        accepted_ids.extend(_solve_facts(facts, rules_path, max_solutions))
        solving_time_s += (time.perf_counter() - t_solve)

    summary: Dict[str, Any] = {
        "candidate_count_input": int(len(candidates)),
        "candidate_count_after_pruning": int(len(working)),
        "candidate_count_pruned": int(pruned_count),
        "component_count": int(len(components)),
        "component_count_raw": int(raw_component_count),
        "largest_component_candidates": int(max((len(c) for c in components), default=0)),
        "largest_component_candidates_raw": int(raw_largest_component),
        "fact_lines": int(fact_lines),
        "fact_bytes": int(fact_bytes),
        "accepted_id_count_before_decode": int(len(accepted_ids)),
        "encoding_time_s": float(encoding_time_s),
        "solving_time_s": float(solving_time_s),
        "total_partitioned_solve_s": float(time.perf_counter() - start_total),
        "component_partition_enabled": bool(enable_component_partition),
        "dominance_pruning_enabled": bool(enable_dominance_pruning),
        "partition_fallback_reason": str(partition_fallback_reason),
        "control_args": list(_build_control_args(max_solutions)),
    }
    return sorted(accepted_ids), summary

def solve(
    facts: str,
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
) -> List[int]:
    """Run clingo on the encoded facts + rules and return accepted candidate IDs."""
    return _solve_facts(facts, rules_path, max_solutions)


def _fallback_rules() -> str:
    """Minimal ASP rules in case the .lp file is missing."""
    return """
{ accept(Id) } :- candidate(Id, _, _, _, _, _).

uses_endpoint(Id, E) :- candidate_endpoint(Id, E, _).
uses_endpoint(Id, E) :- candidate_endpoint(Id, _, E).

candidate_self_closure(-1).

endpoint_key(E) :- uses_endpoint(_, E).
:- endpoint_key(E), 2 { accept(Id) : uses_endpoint(Id, E) }.

:- accept(Id), candidate(Id, P, _, P, _, _), not candidate_self_closure(Id).

#maximize { Utility, Id : accept(Id), candidate_utility(Id, Utility) }.
"""


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
