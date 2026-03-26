"""
ASP Inference  (R-3)
====================
Clingo ASP solver wrapper — loads rules, grounds facts, and returns
ranked stable models with restoration action extraction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── optional dependency ──────────────────────────────────────────────────
try:
    import clingo
except ImportError:  # pragma: no cover
    clingo = None  # type: ignore[assignment]
    logger.warning("clingo not installed — ASP inference will not be available")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ASPConfig:
    """Tuneable parameters for the ASP solver."""

    max_models: int = 0
    """Maximum number of stable models (0 = all)."""
    solve_timeout: str = "umax,10"
    """Clingo solve limit string."""
    opt_mode: str = "optN"
    """Optimisation mode for Clingo."""


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RestorationAction:
    """A single restoration action parsed from an ASP atom."""

    action_type: str
    """One of: complete_contour, extend_curve, mirror_element, replicate_motif,
    group_elements, flag_similar_missing, simplify_interpretation."""
    arguments: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments extracted from the atom."""
    confidence: float = 0.0


@dataclass
class StableModel:
    """One answer set from the ASP solver."""

    atoms: List[str] = field(default_factory=list)
    cost: List[int] = field(default_factory=list)
    optimal: bool = False


@dataclass
class RankedHypothesis:
    """A stable model scored and ranked for downstream use."""

    model: StableModel
    score: float = 0.0
    actions: List[RestorationAction] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ASP Inference Engine
# ═══════════════════════════════════════════════════════════════════════════

class ASPInferenceEngine:
    """Wraps Clingo to load rules, add facts, solve, and extract actions."""

    _ACTION_ATOMS = {
        "complete_contour",
        "extend_curve",
        "mirror_element",
        "replicate_motif",
        "group_elements",
        "flag_similar_missing",
        "simplify_interpretation",
        "restore_frieze_unit",
        "missing_column_pair",
        "portal_symmetric",
        "keystone_expected",
        "arch_type",
    }

    def __init__(self, config: Optional[ASPConfig] = None) -> None:
        self.config = config or ASPConfig()
        self._ctl: Any = None
        self._rules_loaded: bool = False
        self._facts_added: bool = False

    # ── public API ────────────────────────────────────────────────────────

    def load_knowledge_base(
        self,
        rules_path: str,
        grammar_path: Optional[str] = None,
    ) -> None:
        """Initialise Clingo Control and load rule files.

        Parameters
        ----------
        rules_path : str
            Path to ``gestalt_rules.lp``.
        grammar_path : str, optional
            Path to ``architectural_grammar.lp``.  When *None* or the
            file does not exist, only the Gestalt rules are loaded.

        Examples
        --------
        >>> engine = ASPInferenceEngine()
        >>> # engine.load_knowledge_base("gestalt_rules.lp")
        """
        if clingo is None:
            raise RuntimeError("clingo is not installed")

        try:
            args = [
                f"--models={self.config.max_models}",
                f"--opt-mode={self.config.opt_mode}",
            ]
            self._ctl = clingo.Control(args)
            self._ctl.configuration.solve.solve_limit = self.config.solve_timeout

            if os.path.isfile(rules_path):
                self._ctl.load(rules_path)
                logger.info("Loaded rules from %s", rules_path)
            else:
                logger.warning("Rules file not found: %s", rules_path)

            if grammar_path and os.path.isfile(grammar_path):
                self._ctl.load(grammar_path)
                logger.info("Loaded grammar from %s", grammar_path)

            self._rules_loaded = True
        except Exception as exc:
            logger.error("Failed to load knowledge base: %s", exc)
            raise

    def add_facts(self, facts_string: str) -> None:
        """Add serialized Gestalt facts to the base program.

        Parameters
        ----------
        facts_string : str
            Multi-line ASP facts (from :func:`serialize_to_asp_facts`).

        Examples
        --------
        >>> engine = ASPInferenceEngine()
        """
        if self._ctl is None:
            raise RuntimeError("Call load_knowledge_base() first")

        try:
            self._ctl.add("base", [], facts_string)
            self._facts_added = True
            logger.info("Added %d bytes of facts", len(facts_string))
        except Exception as exc:
            logger.error("Failed to add facts: %s", exc)
            raise

    def solve(self) -> List[StableModel]:
        """Ground and solve; return all answer sets.

        Returns
        -------
        List[StableModel]

        Examples
        --------
        >>> engine = ASPInferenceEngine()
        """
        if self._ctl is None:
            raise RuntimeError("Call load_knowledge_base() first")

        try:
            self._ctl.ground([("base", [])])
        except Exception as exc:
            logger.error("Grounding failed: %s", exc)
            raise

        models: List[StableModel] = []

        try:
            with self._ctl.solve(yield_=True) as handle:
                for model in handle:
                    atoms = [str(a) for a in model.symbols(atoms=True)]
                    cost = list(model.cost) if hasattr(model, "cost") else []
                    optimal = model.optimality_proven if hasattr(model, "optimality_proven") else False
                    models.append(
                        StableModel(atoms=atoms, cost=cost, optimal=optimal)
                    )
        except Exception as exc:
            logger.error("Solve failed: %s", exc)
            raise

        logger.info("Solver returned %d stable models", len(models))
        return models

    @staticmethod
    def rank_hypotheses(models: List[StableModel]) -> List[RankedHypothesis]:
        """Score each stable model by total predicate confidence.

        Parameters
        ----------
        models : List[StableModel]

        Returns
        -------
        List[RankedHypothesis]
            Sorted descending by *score*.

        Examples
        --------
        >>> ASPInferenceEngine.rank_hypotheses([])
        []
        """
        ranked: List[RankedHypothesis] = []

        for model in models:
            actions = ASPInferenceEngine.extract_restoration_actions(model)
            score = sum(a.confidence for a in actions) / max(len(actions), 1)
            ranked.append(
                RankedHypothesis(model=model, score=score, actions=actions)
            )

        ranked.sort(key=lambda h: h.score, reverse=True)
        return ranked

    @staticmethod
    def extract_restoration_actions(model: StableModel) -> List[RestorationAction]:
        """Parse action atoms from a stable model.

        Parameters
        ----------
        model : StableModel

        Returns
        -------
        List[RestorationAction]

        Examples
        --------
        >>> m = StableModel(atoms=["complete_contour(0)"])
        >>> acts = ASPInferenceEngine.extract_restoration_actions(m)
        >>> len(acts)
        1
        """
        actions: List[RestorationAction] = []

        for atom_str in model.atoms:
            # Parse "name(arg1,arg2,...)"
            paren_idx = atom_str.find("(")
            if paren_idx < 0:
                name = atom_str.strip()
                args_str = ""
            else:
                name = atom_str[:paren_idx].strip()
                args_str = atom_str[paren_idx + 1 : -1].strip()

            if name not in ASPInferenceEngine._ACTION_ATOMS:
                continue

            # Parse arguments
            raw_args = [a.strip() for a in args_str.split(",")] if args_str else []
            arguments: Dict[str, Any] = {}
            confidence = 0.0

            if name == "complete_contour" and len(raw_args) >= 1:
                arguments["contour_id"] = _safe_int(raw_args[0])
                if len(raw_args) >= 2:
                    confidence = _safe_int(raw_args[1]) / 100.0
            elif name == "extend_curve" and len(raw_args) >= 2:
                arguments["path_a"] = _safe_int(raw_args[0])
                arguments["path_b"] = _safe_int(raw_args[1])
                # New arity: extend_curve(A, B, EA, EB, Conf)
                if len(raw_args) >= 5:
                    arguments["endpoint_a"] = "start" if _safe_int(raw_args[2]) == 0 else "end"
                    arguments["endpoint_b"] = "start" if _safe_int(raw_args[3]) == 0 else "end"
                    confidence = _safe_int(raw_args[4]) / 100.0
                elif len(raw_args) >= 3:
                    confidence = _safe_int(raw_args[2]) / 100.0
            elif name == "mirror_element" and len(raw_args) >= 2:
                arguments["element_id"] = _safe_int(raw_args[0])
                arguments["axis"] = raw_args[1]
            elif name == "replicate_motif" and len(raw_args) >= 3:
                arguments["motif_id"] = _safe_int(raw_args[0])
                arguments["position"] = _safe_int(raw_args[1])
                arguments["transform"] = raw_args[2]
            elif name == "group_elements" and len(raw_args) >= 2:
                arguments["element_a"] = _safe_int(raw_args[0])
                arguments["element_b"] = _safe_int(raw_args[1])
            elif name == "flag_similar_missing" and len(raw_args) >= 2:
                arguments["path_a"] = _safe_int(raw_args[0])
                arguments["path_b"] = _safe_int(raw_args[1])
            elif name == "simplify_interpretation" and len(raw_args) >= 1:
                arguments["loop_id"] = _safe_int(raw_args[0])
            elif name == "restore_frieze_unit" and len(raw_args) >= 2:
                arguments["motif_id"] = _safe_int(raw_args[0])
                arguments["position"] = _safe_int(raw_args[1])
            elif name == "arch_type" and len(raw_args) >= 2:
                arguments["contour_id"] = _safe_int(raw_args[0])
                arguments["type"] = raw_args[1]

            # Try to extract confidence from the last numeric argument
            if raw_args:
                try:
                    last_val = int(raw_args[-1])
                    confidence = last_val / 100.0
                except (ValueError, IndexError):
                    pass

            actions.append(
                RestorationAction(
                    action_type=name,
                    arguments=arguments,
                    confidence=confidence,
                )
            )

        return actions


def _safe_int(s: str) -> int:
    """Parse an integer from string, returning 0 on failure."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Convenience orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_asp(
    predicates_str: str,
    rules_path: Optional[str] = None,
    grammar_path: Optional[str] = None,
    config: Optional[ASPConfig] = None,
) -> List[RankedHypothesis]:
    """End-to-end ASP inference pipeline.

    Parameters
    ----------
    predicates_str : str
        Serialised feature facts (from :func:`serialize_features_to_asp`).
    rules_path : str, optional
        Path to ``gestalt_rules.lp``.  Defaults to the file shipped
        alongside this module.
    grammar_path : str, optional
        Path to an optional architectural grammar file.  When *None*,
        no grammar constraints are loaded.
    config : ASPConfig, optional

    Returns
    -------
    List[RankedHypothesis]

    Examples
    --------
    >>> # run_asp("closure(0,90,50).")
    """
    _dir = os.path.dirname(os.path.abspath(__file__))
    if rules_path is None:
        rules_path = os.path.join(_dir, "gestalt_rules.lp")

    engine = ASPInferenceEngine(config)
    engine.load_knowledge_base(rules_path, grammar_path)
    engine.add_facts(predicates_str)
    models = engine.solve()
    return engine.rank_hypotheses(models)
