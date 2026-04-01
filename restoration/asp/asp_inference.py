"""
R-3: ASP Inference Engine
==========================
Runs clingo with the Gestalt rule program and feature facts,
extracts stable models, parses action atoms, and ranks hypotheses.

Falls back to a pure-Python greedy solver if clingo is unavailable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StableModel:
    """One answer set from the ASP solver."""
    atoms: List[str] = field(default_factory=list)
    cost: int = 0
    optimal: bool = False


@dataclass
class RestorationAction:
    """A single restoration action parsed from an ASP answer set."""
    action_type: str               # "extend_curve" | "complete_contour" | "mirror_element"
    arguments: Dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class RankedHypothesis:
    """A stable model with its parsed actions and aggregate score."""
    model: StableModel
    score: float = 0.0
    actions: List[RestorationAction] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# ASP Inference Engine
# ═══════════════════════════════════════════════════════════════════════

_RULES_PATH = os.path.join(os.path.dirname(__file__), "gestalt_rules.lp")

# Action atom patterns
_EXTEND_RE = re.compile(
    r"extend_curve\((\d+),(\d+),(\d+),(\d+),(\d+)\)"
)
_EXTEND_RE_2 = re.compile(
    r"extend_curve\((\d+),(\d+),(\d+)\)"     # legacy: (A,B,Conf)
)
_EXTEND_RE_1 = re.compile(
    r"extend_curve\((\d+),(\d+)\)"            # legacy: (A,B)
)
_COMPLETE_RE = re.compile(
    r"complete_contour\((\d+)(?:,(\d+))?\)"
)
_MIRROR_RE = re.compile(
    r"mirror_element\((\d+),(\w+)\)"
)


class ASPInferenceEngine:
    """Solve an ASP program with clingo and produce ranked hypotheses."""

    def __init__(self, rules_path: Optional[str] = None, max_models: int = 5):
        self.rules_path = rules_path or _RULES_PATH
        self.max_models = max_models

    # ── Solve ──────────────────────────────────────────────────────────

    def solve(self, asp_facts: str) -> List[StableModel]:
        """Run clingo on facts + rules, return stable models."""
        try:
            import clingo
        except ImportError:
            return self._fallback_solve(asp_facts)

        ctl = clingo.Control([
            f"--models={self.max_models}",
            "--opt-mode=optN",
        ])
        ctl.add("base", [], asp_facts)

        # Load rules file
        if os.path.isfile(self.rules_path):
            with open(self.rules_path, "r", encoding="utf-8") as f:
                ctl.add("rules", [], f.read())
        else:
            raise FileNotFoundError(
                f"ASP rules file not found: {self.rules_path}"
            )

        ctl.ground([("base", []), ("rules", [])])

        models: List[StableModel] = []

        def on_model(model):
            atoms = [str(a) for a in model.symbols(shown=True)]
            sm = StableModel(
                atoms=atoms,
                cost=sum(model.cost),
                optimal=model.optimality_proven,
            )
            models.append(sm)

        ctl.solve(on_model=on_model)
        return models

    # ── Parse actions from a model ─────────────────────────────────────

    @staticmethod
    def extract_restoration_actions(model: StableModel) -> List[RestorationAction]:
        """Parse action atoms from a stable model."""
        actions: List[RestorationAction] = []
        ep_map = {0: "start", 1: "end"}

        for atom in model.atoms:
            # extend_curve(A, B, EA, EB, Conf)
            m = _EXTEND_RE.match(atom)
            if m:
                a, b, ea, eb, conf = [int(x) for x in m.groups()]
                actions.append(RestorationAction(
                    action_type="extend_curve",
                    arguments={
                        "path_a": a, "path_b": b,
                        "endpoint_a": ep_map.get(ea, "end"),
                        "endpoint_b": ep_map.get(eb, "start"),
                    },
                    confidence=conf / 100.0,
                ))
                continue

            # extend_curve(A, B, Conf)  — legacy 3-arg
            m = _EXTEND_RE_2.match(atom)
            if m:
                a, b, conf = [int(x) for x in m.groups()]
                actions.append(RestorationAction(
                    action_type="extend_curve",
                    arguments={
                        "path_a": a, "path_b": b,
                        "endpoint_a": "end", "endpoint_b": "start",
                    },
                    confidence=conf / 100.0,
                ))
                continue

            # extend_curve(A, B)  — legacy 2-arg
            m = _EXTEND_RE_1.match(atom)
            if m:
                a, b = [int(x) for x in m.groups()]
                actions.append(RestorationAction(
                    action_type="extend_curve",
                    arguments={
                        "path_a": a, "path_b": b,
                        "endpoint_a": "end", "endpoint_b": "start",
                    },
                    confidence=0.0,
                ))
                continue

            # complete_contour(P [, Conf])
            m = _COMPLETE_RE.match(atom)
            if m:
                pid = int(m.group(1))
                conf = int(m.group(2)) if m.group(2) else 0
                actions.append(RestorationAction(
                    action_type="complete_contour",
                    arguments={"contour_id": pid},
                    confidence=conf / 100.0,
                ))
                continue

            # mirror_element(P, Axis)
            m = _MIRROR_RE.match(atom)
            if m:
                pid = int(m.group(1))
                axis = m.group(2)
                actions.append(RestorationAction(
                    action_type="mirror_element",
                    arguments={"element_id": pid, "axis": axis},
                    confidence=0.5,
                ))
                continue

        return actions

    # ── Rank hypotheses ────────────────────────────────────────────────

    @staticmethod
    def rank_hypotheses(models: List[StableModel]) -> List[RankedHypothesis]:
        """Parse and rank all stable models by aggregate confidence."""
        hypotheses: List[RankedHypothesis] = []
        for model in models:
            actions = ASPInferenceEngine.extract_restoration_actions(model)
            score = sum(a.confidence for a in actions)
            hypotheses.append(RankedHypothesis(
                model=model, score=score, actions=actions,
            ))
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses

    # ── Fallback solver (no clingo) ────────────────────────────────────

    def _fallback_solve(self, asp_facts: str) -> List[StableModel]:
        """Pure-Python greedy solver for when clingo is unavailable.

        Parses the fact string directly and applies a greedy strategy:
        - All continues() facts → extend_curve actions
        - All closure() facts → complete_contour actions
        - Mutual exclusion enforced greedily
        """
        atoms: List[str] = []
        used_endpoints: set = set()

        # Parse continues facts
        continues_facts = re.findall(
            r"continues\((\d+),(\d+),(\d+),(\d+),(\d+)\)\.", asp_facts
        )
        # Sort by confidence descending
        continues_facts.sort(key=lambda x: int(x[4]), reverse=True)

        for a, b, ea, eb, conf in continues_facts:
            a, b, ea, eb, conf_int = int(a), int(b), int(ea), int(eb), int(conf)
            key_a = (a, ea)
            key_b = (b, eb)
            if key_a in used_endpoints or key_b in used_endpoints:
                continue
            if a == b:
                continue
            atoms.append(f"extend_curve({a},{b},{ea},{eb},{conf_int})")
            used_endpoints.add(key_a)
            used_endpoints.add(key_b)

        # Parse closure facts
        closure_facts = re.findall(r"closure\((\d+),(\d+)\)\.", asp_facts)
        closure_facts.sort(key=lambda x: int(x[1]), reverse=True)

        closed_paths = set()
        # Check for paths already involved in extend actions
        extended_paths = set()
        for atom in atoms:
            m = _EXTEND_RE.match(atom)
            if m:
                extended_paths.add(int(m.group(1)))
                extended_paths.add(int(m.group(2)))

        for pid_str, conf_str in closure_facts:
            pid = int(pid_str)
            if pid in extended_paths or pid in closed_paths:
                continue
            atoms.append(f"complete_contour({pid},{conf_str})")
            closed_paths.add(pid)

        return [StableModel(atoms=atoms, cost=0, optimal=True)]
