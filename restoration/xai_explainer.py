"""
XAI Explainer  (R-5)
====================
Explainability layer for the restoration pipeline:
  • Proof traces mapping ASP atoms back to supporting Gestalt facts
  • SHAP TreeExplainer on XGBoost EFD classifier
  • Surrogate decision-tree extraction of IF-THEN rules
  • Human-readable report generation
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── optional heavy dependencies ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment,misc]
    logger.info("xgboost not installed — SHAP classifier unavailable")

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    logger.info("shap not installed — SHAP explanations unavailable")

try:
    from sklearn.tree import DecisionTreeClassifier
except ImportError:  # pragma: no cover
    DecisionTreeClassifier = None  # type: ignore[assignment,misc]


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class XAIConfig:
    """Hyperparameters for the XAI layer."""

    efd_order: int = 10
    """Number of EFD harmonics (must match vocabulary)."""
    surrogate_max_depth: int = 4
    """Max depth of surrogate decision tree."""
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProofTrace:
    """Backward mapping: derived atom → supporting facts + rule names."""

    atom: str
    rule_name: str
    supporting_facts: List[str] = field(default_factory=list)
    principle: str = ""
    confidence: float = 0.0


@dataclass
class SHAPExplanation:
    """Per-feature SHAP values for one EFD classification."""

    feature_names: List[str] = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None
    predicted_class: str = ""
    predicted_proba: float = 0.0


@dataclass
class ExplanationReport:
    """Full per-element report combining all XAI components."""

    proof_traces: List[ProofTrace] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    shap_explanations: List[SHAPExplanation] = field(default_factory=list)
    surrogate_rules: List[str] = field(default_factory=list)
    summary: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 1. Proof trace extraction
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from action atoms to the Gestalt principle they originate from
_ACTION_TO_PRINCIPLE = {
    "complete_contour": "Closure",
    "extend_curve": "Good Continuation",
    "mirror_element": "Symmetry",
    "replicate_motif": "Periodicity / Proximity",
    "group_elements": "Proximity",
    "flag_similar_missing": "Similarity",
    "simplify_interpretation": "Pragnanz",
    "restore_frieze_unit": "Periodicity",
    "arch_type": "Closure / Domain",
    "keystone_expected": "Domain Knowledge",
}

# Mapping from action atoms to the rule name that derived them
_ACTION_TO_RULE = {
    "complete_contour": "closure_rule",
    "extend_curve": "continuation_rule",
    "mirror_element": "symmetry_rule",
    "replicate_motif": "periodicity_rule",
    "group_elements": "proximity_grouping_rule",
    "flag_similar_missing": "similarity_flag_rule",
    "simplify_interpretation": "pragnanz_simplify_rule",
    "restore_frieze_unit": "frieze_restoration_rule",
    "arch_type": "arch_classification_rule",
    "keystone_expected": "keystone_rule",
}

# Mapping from action atoms to fact predicates they depend on
_ACTION_TO_FACTS = {
    "complete_contour": ["closure"],
    "extend_curve": ["continues"],
    "mirror_element": ["symmetric", "member"],
    "replicate_motif": ["periodic_pattern", "expected_position"],
    "group_elements": ["proximity_group", "member"],
    "flag_similar_missing": ["similar_shape"],
    "simplify_interpretation": ["pragnanz"],
    "restore_frieze_unit": ["frieze_element", "replicate_motif"],
    "arch_type": ["complete_contour", "curvature_class"],
    "keystone_expected": ["arch", "arch_type"],
}


def extract_proof_trace(
    clingo_model_atoms: List[str],
    predicate_dict: Optional[Dict[str, List[str]]] = None,
) -> List[ProofTrace]:
    """Map each derived action atom back to supporting Gestalt facts and rule names.

    Parameters
    ----------
    clingo_model_atoms : List[str]
        Atom strings from the stable model (e.g. ``["complete_contour(0)", ...]``).
    predicate_dict : dict, optional
        Optional ``{predicate_name: [atom_strings]}`` for richer lookups.

    Returns
    -------
    List[ProofTrace]

    Examples
    --------
    >>> traces = extract_proof_trace(["complete_contour(0)"])
    >>> len(traces) == 1
    True
    >>> traces[0].principle
    'Closure'
    """
    traces: List[ProofTrace] = []
    pred_dict = predicate_dict or {}

    # Index all atoms by predicate name
    atom_index: Dict[str, List[str]] = {}
    for atom_str in clingo_model_atoms:
        paren = atom_str.find("(")
        name = atom_str[:paren].strip() if paren >= 0 else atom_str.strip()
        atom_index.setdefault(name, []).append(atom_str)

    # Merge with predicate_dict
    for k, v in pred_dict.items():
        atom_index.setdefault(k, []).extend(v)

    for atom_str in clingo_model_atoms:
        paren = atom_str.find("(")
        name = atom_str[:paren].strip() if paren >= 0 else atom_str.strip()

        if name not in _ACTION_TO_PRINCIPLE:
            continue

        # Collect supporting facts for this action
        fact_preds = _ACTION_TO_FACTS.get(name, [])
        supporting: List[str] = []
        for fp in fact_preds:
            supporting.extend(atom_index.get(fp, []))

        # Parse confidence from atom arguments
        confidence = 0.0
        if paren >= 0:
            args_str = atom_str[paren + 1 : -1]
            parts = [p.strip() for p in args_str.split(",")]
            if parts:
                try:
                    confidence = int(parts[-1]) / 100.0
                except ValueError:
                    pass

        traces.append(
            ProofTrace(
                atom=atom_str,
                rule_name=_ACTION_TO_RULE.get(name, "unknown"),
                supporting_facts=supporting,
                principle=_ACTION_TO_PRINCIPLE.get(name, "Unknown"),
                confidence=confidence,
            )
        )

    return traces


# ═══════════════════════════════════════════════════════════════════════════
# 2. Human-readable explanation
# ═══════════════════════════════════════════════════════════════════════════

def format_explanation(trace: ProofTrace, action: Optional[Any] = None) -> str:
    """Format a single restoration explanation as human-readable text.

    Parameters
    ----------
    trace : ProofTrace
    action : RestorationAction, optional

    Returns
    -------
    str

    Examples
    --------
    >>> t = ProofTrace(atom="complete_contour(0)", rule_name="closure_rule",
    ...               principle="Closure", confidence=0.9)
    >>> s = format_explanation(t)
    >>> "Closure" in s
    True
    """
    lines = [
        f"Action: {trace.atom}",
        f"  Rule: {trace.rule_name}",
        f"  Gestalt Principle: {trace.principle}",
        f"  Confidence: {trace.confidence:.2f}",
    ]
    if trace.supporting_facts:
        lines.append(f"  Supporting facts: {', '.join(trace.supporting_facts[:5])}")
    if action is not None:
        action_type = getattr(action, "action_type", str(action))
        lines.append(f"  Action type: {action_type}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SHAP classifier
# ═══════════════════════════════════════════════════════════════════════════

def _efd_feature_names(order: int = 10) -> List[str]:
    """Generate feature names for EFD coefficients."""
    names: List[str] = []
    for i in range(1, order + 1):
        for suffix in ["an", "bn", "cn", "dn"]:
            names.append(f"harmonic_{i}_{suffix}")
    # After normalization, first 3 values are dropped
    return names[3:]


def train_shap_classifier(
    shape_vocab: Any,
    config: Optional[XAIConfig] = None,
) -> Tuple[Any, Any]:
    """Train an XGBoost classifier on EFD features → shape categories.

    Parameters
    ----------
    shape_vocab : ShapeVocab
        Must have ``.features`` and ``.labels``.
    config : XAIConfig, optional

    Returns
    -------
    classifier, explainer
        (XGBClassifier, shap.TreeExplainer) or (None, None) if dependencies missing.

    Examples
    --------
    >>> train_shap_classifier(None)
    (None, None)
    """
    cfg = config or XAIConfig()

    if XGBClassifier is None or shap is None:
        logger.warning("XGBoost or SHAP not installed")
        return None, None

    if shape_vocab is None or shape_vocab.features is None or len(shape_vocab.labels) < 2:
        logger.warning("Insufficient vocabulary for SHAP training")
        return None, None

    X = np.asarray(shape_vocab.features, dtype=np.float32)
    # Encode labels to integers
    unique_labels = sorted(set(shape_vocab.labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in shape_vocab.labels], dtype=np.int32)

    if len(unique_labels) < 2:
        logger.warning("Need ≥2 classes for classification")
        return None, None

    try:
        clf = XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        clf.fit(X, y)

        explainer = shap.TreeExplainer(clf)
        logger.info("SHAP classifier trained on %d samples, %d classes", len(y), len(unique_labels))
        return clf, explainer
    except Exception as exc:
        logger.error("SHAP classifier training failed: %s", exc)
        return None, None


def explain_efd_match(
    coeffs: np.ndarray,
    classifier: Any,
    explainer: Any,
    config: Optional[XAIConfig] = None,
) -> SHAPExplanation:
    """Explain which harmonic coefficients drove the EFD classification.

    Parameters
    ----------
    coeffs : np.ndarray
        Flat EFD feature vector.
    classifier : XGBClassifier
    explainer : shap.TreeExplainer
    config : XAIConfig, optional

    Returns
    -------
    SHAPExplanation

    Examples
    --------
    >>> explain_efd_match(np.zeros(37), None, None)
    SHAPExplanation(feature_names=[], shap_values=None, predicted_class='', predicted_proba=0.0)
    """
    cfg = config or XAIConfig()

    if classifier is None or explainer is None:
        return SHAPExplanation()

    try:
        x = np.asarray(coeffs, dtype=np.float32).ravel().reshape(1, -1)
        # Ensure correct dimension
        expected_dim = classifier.n_features_in_
        if x.shape[1] < expected_dim:
            x = np.pad(x, ((0, 0), (0, expected_dim - x.shape[1])))
        elif x.shape[1] > expected_dim:
            x = x[:, :expected_dim]

        pred = classifier.predict(x)[0]
        proba = float(classifier.predict_proba(x).max())

        sv = explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[int(pred)]

        feature_names = _efd_feature_names(cfg.efd_order)
        if len(feature_names) > x.shape[1]:
            feature_names = feature_names[: x.shape[1]]

        return SHAPExplanation(
            feature_names=feature_names,
            shap_values=sv.ravel() if sv is not None else None,
            predicted_class=str(pred),
            predicted_proba=proba,
        )
    except Exception as exc:
        logger.error("SHAP explanation failed: %s", exc)
        return SHAPExplanation()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Surrogate decision tree
# ═══════════════════════════════════════════════════════════════════════════

_SURROGATE_FEATURES = [
    "gap_dist",
    "tangent_angle",
    "efd_dist",
    "loop_persistence",
    "sym_residual",
    "confidence",
]


def train_surrogate_tree(
    feature_records: np.ndarray,
    action_labels: np.ndarray,
    config: Optional[XAIConfig] = None,
) -> Any:
    """Train a shallow surrogate decision tree.

    Parameters
    ----------
    feature_records : np.ndarray, shape (N, 6)
        Columns: [gap_dist, tangent_angle, efd_dist, loop_persistence,
        sym_residual, confidence].
    action_labels : np.ndarray, shape (N,)
        Integer action-class labels.
    config : XAIConfig, optional

    Returns
    -------
    DecisionTreeClassifier or None

    Examples
    --------
    >>> train_surrogate_tree(np.zeros((5, 6)), np.zeros(5, dtype=int))
    DecisionTreeClassifier(max_depth=4)
    """
    cfg = config or XAIConfig()
    if DecisionTreeClassifier is None:
        return None

    try:
        tree = DecisionTreeClassifier(max_depth=cfg.surrogate_max_depth)
        tree.fit(feature_records, action_labels)
        logger.info("Surrogate tree trained: depth=%d, nodes=%d",
                     tree.get_depth(), tree.tree_.node_count)
        return tree
    except Exception as exc:
        logger.error("Surrogate tree training failed: %s", exc)
        return None


def extract_if_then_rules(
    tree: Any,
    feature_names: Optional[List[str]] = None,
) -> List[str]:
    """Export decision-tree paths as readable IF-THEN rules.

    Parameters
    ----------
    tree : DecisionTreeClassifier
    feature_names : List[str], optional

    Returns
    -------
    List[str]

    Examples
    --------
    >>> extract_if_then_rules(None)
    []
    """
    if tree is None:
        return []

    if feature_names is None:
        feature_names = _SURROGATE_FEATURES

    rules: List[str] = []

    try:
        tree_ = tree.tree_
        n_nodes = tree_.node_count
        children_left = tree_.children_left
        children_right = tree_.children_right
        features = tree_.feature
        thresholds = tree_.threshold
        values = tree_.value

        def _recurse(node: int, path: List[str]) -> None:
            if children_left[node] == children_right[node]:
                # Leaf
                class_idx = int(np.argmax(values[node]))
                samples = int(values[node].sum())
                rule = " AND ".join(path) if path else "TRUE"
                rules.append(f"IF {rule} THEN action={class_idx} (samples={samples})")
                return

            feat = features[node]
            thresh = thresholds[node]
            fname = feature_names[feat] if feat < len(feature_names) else f"f{feat}"

            _recurse(children_left[node], path + [f"{fname} <= {thresh:.3f}"])
            _recurse(children_right[node], path + [f"{fname} > {thresh:.3f}"])

        _recurse(0, [])
    except Exception as exc:
        logger.error("Rule extraction failed: %s", exc)

    return rules


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    hypotheses: List[Any],
    result: Any,
    vocab: Optional[Any] = None,
    config: Optional[XAIConfig] = None,
) -> ExplanationReport:
    """Generate a full per-element report combining all XAI components.

    Parameters
    ----------
    hypotheses : List[RankedHypothesis]
    result : RestorationResult
    vocab : ShapeVocab, optional
    config : XAIConfig, optional

    Returns
    -------
    ExplanationReport

    Examples
    --------
    >>> generate_report([], None)
    ExplanationReport(proof_traces=[], explanations=[], shap_explanations=[], surrogate_rules=[], summary='')
    """
    cfg = config or XAIConfig()
    report = ExplanationReport()

    if not hypotheses:
        report.summary = "No hypotheses to explain."
        return report

    best = hypotheses[0]
    atoms = best.model.atoms if hasattr(best, "model") else []
    actions = best.actions if hasattr(best, "actions") else []

    # 1. Proof traces
    report.proof_traces = extract_proof_trace(atoms)

    # 2. Human-readable explanations
    for trace in report.proof_traces:
        report.explanations.append(format_explanation(trace))

    # 3. SHAP explanations (if vocab available)
    if vocab is not None:
        try:
            clf, explainer = train_shap_classifier(vocab, cfg)
            if clf is not None and explainer is not None:
                # Explain each action related to EFD matching
                for action in actions:
                    if hasattr(action, "arguments"):
                        feat = action.arguments.get("coeffs")
                        if feat is not None:
                            se = explain_efd_match(feat, clf, explainer, cfg)
                            report.shap_explanations.append(se)
        except Exception as exc:
            logger.error("SHAP report generation failed: %s", exc)

    # 4. Surrogate rules (if we have enough action records)
    if len(actions) >= 2:
        try:
            # Build feature matrix from actions
            feat_rows = []
            labels = []
            action_type_map: Dict[str, int] = {}
            for act in actions:
                args = act.arguments if hasattr(act, "arguments") else {}
                row = [
                    float(args.get("gap_dist", args.get("gap", 0))),
                    float(args.get("tangent_angle", args.get("angle", 0))),
                    float(args.get("efd_dist", args.get("efd_distance", 0))),
                    float(args.get("loop_persistence", 0)),
                    float(args.get("sym_residual", args.get("residual", 0))),
                    float(act.confidence if hasattr(act, "confidence") else 0),
                ]
                feat_rows.append(row)
                atype = act.action_type if hasattr(act, "action_type") else "unknown"
                if atype not in action_type_map:
                    action_type_map[atype] = len(action_type_map)
                labels.append(action_type_map[atype])

            if len(set(labels)) >= 2:
                X = np.array(feat_rows, dtype=np.float64)
                y = np.array(labels, dtype=np.int32)
                tree = train_surrogate_tree(X, y, cfg)
                report.surrogate_rules = extract_if_then_rules(tree)
        except Exception as exc:
            logger.error("Surrogate rule generation failed: %s", exc)

    # Summary
    n_actions = len(actions)
    n_traces = len(report.proof_traces)
    report.summary = (
        f"Restoration report: {n_actions} actions analysed, "
        f"{n_traces} proof traces generated, "
        f"{len(report.shap_explanations)} SHAP explanations, "
        f"{len(report.surrogate_rules)} surrogate rules."
    )

    logger.info(report.summary)
    return report
