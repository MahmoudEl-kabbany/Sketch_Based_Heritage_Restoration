"""Phase 4 — ASP Decision Engine.

Encodes scored candidates as ASP facts, solves with clingo,
and decodes the optimal connection set.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from restoration.extraction import EndpointInfo
from restoration.candidates import ConnectionCandidate

RULES_PATH = os.path.join(os.path.dirname(__file__), "asp_rules", "restoration.lp")


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

    # Endpoint facts — encode "start"/"end" as 0/1
    end_map = {"start": 0, "end": 1}
    for i, ep in enumerate(endpoints):
        x = int(round(ep.position[0]))
        y = int(round(ep.position[1]))
        eid = _endpoint_token_id(ep)
        lines.append(
            f"endpoint({ep.path_index},{end_map[ep.end]},{x},{y})."
        )
        lines.append(
            f"endpoint_id({eid},{ep.path_index},{end_map[ep.end]})."
        )

    lines.append("")

    # Candidate facts
    for c in candidates:
        ea = end_map[c.ep_a.end]
        eb = end_map[c.ep_b.end]
        ea_id = _endpoint_token_id(c.ep_a)
        eb_id = _endpoint_token_id(c.ep_b)
        score_int = _to_int_score(c.score)
        lines.append(
            f"candidate({c.id},{c.ep_a.path_index},{ea},"
            f"{c.ep_b.path_index},{eb},{score_int})."
        )
        lines.append(f"candidate_endpoint({c.id},{ea_id},{eb_id}).")

        # Gestalt sub-scores (used by soft rules in the ASP program)
        direction_ab = c.ep_b.position - c.ep_a.position
        n = np.linalg.norm(direction_ab)
        if n > 1e-12:
            cont_raw = float(np.dot(c.ep_a.tangent, direction_ab / n))
        else:
            cont_raw = 0.0

        cont_int = _to_int_score(max(0.0, cont_raw))
        lines.append(f"gestalt_continuation({c.id},{cont_int}).")

        # Closure flag
        closure_flag = 1 if c.score > 0.5 and c.distance < 30.0 else 0
        lines.append(f"gestalt_closure({c.id},{closure_flag}).")

        # Tier
        lines.append(f"tier({c.id},{c.tier}).")

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Solving
# ═══════════════════════════════════════════════════════════════════════════

def solve(
    facts: str,
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
) -> List[int]:
    """Run clingo on the encoded facts + rules and return accepted candidate IDs."""
    import clingo

    ctl = clingo.Control([
        f"--models={max_solutions}",
        "--opt-mode=optN",
    ])

    # Load rules from file
    if os.path.isfile(rules_path):
        ctl.load(rules_path)
    else:
        # Fallback: embed minimal rules
        ctl.add("base", [], _fallback_rules())

    # Add facts
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


def _fallback_rules() -> str:
    """Minimal ASP rules in case the .lp file is missing."""
    return """
{ accept(Id) } :- candidate(Id, _, _, _, _, _).

uses_endpoint(Id, E) :- candidate_endpoint(Id, E, _).
uses_endpoint(Id, E) :- candidate_endpoint(Id, _, E).

:- accept(Id1), accept(Id2), Id1 != Id2,
   uses_endpoint(Id1, E), uses_endpoint(Id2, E).

:- accept(Id1), accept(Id2), Id1 != Id2,
   candidate(Id1, P, E, _, _, _),
   candidate(Id2, P, E, _, _, _).

:- accept(Id1), accept(Id2), Id1 != Id2,
   candidate(Id1, _, _, P, E, _),
   candidate(Id2, _, _, P, E, _).

:- accept(Id), candidate(Id, P, _, P, _, _).

#maximize { Score, Id : accept(Id), candidate(Id, _, _, _, _, Score) }.
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
