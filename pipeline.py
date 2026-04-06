"""
Restoration Pipeline — Entry Point
===================================
Top-level orchestrator connecting all stages end-to-end:

    Image → [Preprocessing] → BezierPaths + EFD
          → R-1 Feature Bridge
          → R-2 ASP Inference
          → R-3 Geometric Synthesis
          → R-4 XAI Explainer
          → Output (RestorationResult, ExplanationReport)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project path setup ───────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bezier_curves.bezier import BezierPath, fit_from_image, fit_from_image_skeleton
from eliptic_fourier_descriptors.efd import (
    compute_efd_features,
    extract_efd_from_image,
)
from restoration.feature_bridge import (
    FeatureBundle,
    FeatureBridgeConfig,
    extract_all_features,
    serialize_features_to_asp,
)
from restoration.asp.asp_inference import (
    ASPConfig,
    RankedHypothesis,
    run_asp,
)
from restoration.restoration import (
    RestorationConfig,
    RestorationResult,
    ShapeVocab,
    build_shape_vocabulary,
    execute_restoration,
    visualise,
)
from restoration.xai_explainer import (
    ExplanationReport,
    XAIConfig,
    generate_report,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline configuration (combines all sub-configs)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Master configuration aggregating all stage configs."""

    # Preprocessing
    use_skeleton: bool = True
    min_contour_area: float = 100.0
    max_bezier_error: float = 5.0
    efd_order: int = 10

    # Feature bridge
    feature_bridge: FeatureBridgeConfig = None  # type: ignore[assignment]

    # ASP
    asp: ASPConfig = None  # type: ignore[assignment]

    # Restoration
    restoration: RestorationConfig = None  # type: ignore[assignment]

    # XAI
    xai: XAIConfig = None  # type: ignore[assignment]

    # Paths
    shape_vocab_dir: str = ""
    output_dir: str = "restoration_output"

    def __post_init__(self) -> None:
        if self.feature_bridge is None:
            self.feature_bridge = FeatureBridgeConfig()
        if self.asp is None:
            self.asp = ASPConfig()
        if self.restoration is None:
            self.restoration = RestorationConfig()
        if self.xai is None:
            self.xai = XAIConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def restore(
    image_path: str,
    config: Optional[PipelineConfig] = None,
) -> Tuple[RestorationResult, ExplanationReport]:
    """Run the full restoration pipeline on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input sketch image.
    config : PipelineConfig, optional
        Master configuration.  Uses defaults when *None*.

    Returns
    -------
    result : RestorationResult
        New Bézier segments, paths, and blended EFD coefficients.
    report : ExplanationReport
        Full XAI report with proof traces, SHAP, and surrogate rules.

    Examples
    --------
    >>> # result, report = restore("sketch.png")
    """
    cfg = config or PipelineConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("═" * 60)
    logger.info("  Restoration pipeline — %s", image_path)
    logger.info("═" * 60)

    # ── Stage 0: Preprocessing (existing modules) ────────────────────────
    logger.info("[0/4] Preprocessing: extracting Bézier paths & EFD ...")

    adjacency: Dict[int, set] = {}
    if cfg.use_skeleton:
        paths, adjacency = fit_from_image_skeleton(image_path, max_error=cfg.max_bezier_error)
    else:
        paths = fit_from_image(
            image_path,
            min_contour_area=cfg.min_contour_area,
            max_error=cfg.max_bezier_error,
        )
    logger.info("  -> %d Bézier paths extracted", len(paths))

    # EFD extraction
    efd_result = extract_efd_from_image(
        image_path, order=cfg.efd_order, min_contour_area=cfg.min_contour_area, use_skeleton=cfg.use_skeleton
    )
    efd_data: Dict[int, np.ndarray] = {}
    for idx, res in enumerate(efd_result.get("efd_results", [])):
        feat = res.get("features")
        if feat is not None:
            efd_data[idx] = feat
    logger.info("  -> %d EFD feature vectors", len(efd_data))

    # ── Stage 1: Feature Bridge (R-1) ────────────────────────────────────
    logger.info("[1/4] Feature extraction ...")
    features = extract_all_features(
        paths, efd_data, config=cfg.feature_bridge, adjacency=adjacency
    )
    logger.info("  -> %d gaps, symmetry=%s",
                len(features.gaps), features.symmetry_axis is not None)

    # ── Stage 1b: Serialise features to ASP facts ────────────────────────
    facts_str = serialize_features_to_asp(features, paths=paths)
    logger.info("  -> ASP facts: %d bytes", len(facts_str))

    # ── Stage 2: ASP Inference (R-2) ─────────────────────────────────────
    logger.info("[2/4] ASP inference ...")
    try:
        hypotheses = run_asp(facts_str, config=cfg.asp)
        logger.info("  -> %d ranked hypotheses", len(hypotheses))
    except Exception as exc:
        logger.error("ASP inference failed: %s", exc)
        hypotheses = []

    # ── Stage 3: Geometric Synthesis (R-3) ───────────────────────────────
    logger.info("[3/4] Geometric synthesis ...")
    vocab: Optional[ShapeVocab] = None
    if cfg.shape_vocab_dir and os.path.isdir(cfg.shape_vocab_dir):
        vocab = build_shape_vocabulary(cfg.shape_vocab_dir, cfg.restoration)

    result = execute_restoration(
        hypotheses, paths, efd_data, vocab, cfg.restoration
    )
    logger.info(
        "  -> %d new segments, %d new paths",
        len(result.new_segments), len(result.new_paths),
    )

    # ── Stage 4: XAI Explanation (R-4) ───────────────────────────────────
    logger.info("[4/4] Generating explanation report ...")
    report = generate_report(hypotheses, result, vocab, cfg.xai)
    logger.info("  -> %s", report.summary)

    logger.info("═" * 60)
    logger.info("  Pipeline complete")
    logger.info("═" * 60)

    return result, report


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helper
# ═══════════════════════════════════════════════════════════════════════════

def visualise_result(
    result: RestorationResult,
    report: ExplanationReport,
    image_path: str,
    output_dir: str = "restoration_output",
) -> None:
    """Save side-by-side original vs restored sketch with explanation overlay.

    Parameters
    ----------
    result : RestorationResult
    report : ExplanationReport
    image_path : str
    output_dir : str

    Examples
    --------
    >>> # visualise_result(result, report, "sketch.png")
    """
    os.makedirs(output_dir, exist_ok=True)

    vis_path = os.path.join(output_dir, "restoration_visualisation.png")
    visualise(result, image_path=image_path, output_path=vis_path)
    original_copy_path = os.path.join(output_dir, "restoration_original.png")

    # Write text report
    report_path = os.path.join(output_dir, "restoration_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RESTORATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Summary: {report.summary}\n\n")

        if report.explanations:
            f.write("EXPLANATIONS\n")
            f.write("-" * 40 + "\n")
            for exp in report.explanations:
                f.write(exp + "\n\n")

        if report.surrogate_rules:
            f.write("SURROGATE RULES\n")
            f.write("-" * 40 + "\n")
            for rule in report.surrogate_rules:
                f.write(rule + "\n")
            f.write("\n")

        if result.additions:
            f.write("ADDITIONS (Overlay Index Mapping)\n")
            f.write("-" * 40 + "\n")
            for add in result.additions:
                f.write(
                    f"#{add.segment_id}: method={add.method}, conf={add.confidence:.3f}, "
                    f"paths=({add.path_a},{add.path_b}), "
                    f"endpoints=(e{add.endpoint_a_id},e{add.endpoint_b_id}), "
                    f"forced={add.is_forced}\n"
                )
            f.write("\n")

        f.write("OUTPUT IMAGES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original: {original_copy_path}\n")
        f.write(f"Overlay : {vis_path}\n")
        f.write("\n")

        f.write(f"\nActions applied: {len(result.actions_applied)}\n")
        for act in result.actions_applied:
            f.write(f"  • {act}\n")

    logger.info("Report saved to %s", report_path)
    logger.info("Visualisation saved to %s", vis_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point for the restoration pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sketch-Based Heritage Restoration Pipeline"
    )
    # Default image allows running `python pipeline.py` without arguments
    default_img = os.path.join(_PROJECT_ROOT, "test_images", "restoration_test_damaged_big.png")
    parser.add_argument("--image", default=default_img, help="Input sketch image path")
    parser.add_argument("--no-skeleton", action="store_false", dest="skeleton", help="Disable skeleton fitting")
    parser.add_argument("--vocab", default="", help="Shape vocabulary directory")
    parser.add_argument("--output", default="restoration_output", help="Output directory")
    parser.add_argument("--efd-order", type=int, default=40, help="EFD harmonic order")
    parser.set_defaults(skeleton=True)
    args = parser.parse_args()

    cfg = PipelineConfig(
        use_skeleton=args.skeleton,
        efd_order=args.efd_order,
        shape_vocab_dir=args.vocab,
        output_dir=args.output,
    )

    result, report = restore(args.image, cfg)
    visualise_result(result, report, args.image, args.output)


if __name__ == "__main__":
    main()
