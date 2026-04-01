"""
Restoration Pipeline Orchestrator
===================================
End-to-end: damaged image → preprocessing → feature extraction →
ASP inference → geometric synthesis → overlay + explanation.

Usage
-----
    from restoration.pipeline import restore_image
    result = restore_image("damaged.png", "outputs/")
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bezier_curves.bezier import (
    BezierPath,
    BezierSegment,
    fit_from_image_skeleton,
    fit_from_contours,
    _extract_raster_contours,
)
from restoration.preprocessing import preprocess_for_restoration, PreprocessResult
from restoration.feature_bridge import (
    FeatureBridgeConfig,
    FeatureBundle,
    extract_all_features,
    serialize_features_to_asp,
    extract_endpoint_gaps,
    _detect_closure_candidates,
)
from restoration.asp.asp_inference import ASPInferenceEngine, RankedHypothesis
from restoration.restoration import (
    execute_restoration,
    RestorationResult,
    efd_close_contour,
)
from restoration.xai_explainer import (
    generate_explanation,
    format_explanation_text,
    save_explanation,
    annotate_image,
    ExplanationEntry,
)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Top-level configuration for the restoration pipeline."""
    # Preprocessing
    noise_h: int = 10
    min_component_fraction: float = 0.001
    thick_stroke_threshold: float = 4.0
    spur_length: int = 8

    # Bézier fitting
    max_bezier_error: float = 5.0
    tangent_lookahead: int = 5
    merge_radius: float = 3.0

    # Feature extraction
    feature_config: FeatureBridgeConfig = field(
        default_factory=FeatureBridgeConfig
    )

    # EFD
    efd_order: int = 20
    efd_recon_points: int = 500

    # ASP
    asp_max_models: int = 5

    # Rendering
    restoration_color_bgr: Tuple[int, int, int] = (0, 200, 0)  # green
    restoration_thickness: int = 2
    save_explanation_file: bool = True


# ═══════════════════════════════════════════════════════════════════════
# Pipeline result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Full pipeline output."""
    original_image: np.ndarray
    overlay_image: np.ndarray
    restoration: RestorationResult
    explanations: List[ExplanationEntry]
    explanation_text: str
    paths: List[BezierPath]
    bundle: FeatureBundle
    timing: Dict[str, float] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def restore_image(
    image_path: str,
    output_dir: str = "restoration_outputs",
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Run the full restoration pipeline on a damaged sketch image.

    Parameters
    ----------
    image_path : str
        Path to the damaged image.
    output_dir : str
        Directory for output files.
    config : PipelineConfig, optional
        Pipeline configuration.  Uses sensible defaults.

    Returns
    -------
    PipelineResult
    """
    if config is None:
        config = PipelineConfig()

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    timings: Dict[str, float] = {}

    print(f"\n{'=' * 60}")
    print(f"  Restoration Pipeline: {os.path.basename(image_path)}")
    print(f"{'=' * 60}")

    # ── Step 1: Preprocess ─────────────────────────────────────────
    t0 = time.time()
    print("  [1/7] Preprocessing...")
    prep = preprocess_for_restoration(
        image_path,
        noise_h=config.noise_h,
        min_component_fraction=config.min_component_fraction,
        thick_stroke_threshold=config.thick_stroke_threshold,
        spur_length=config.spur_length,
    )
    timings["preprocess"] = time.time() - t0
    print(f"        Image: {prep.image_w}x{prep.image_h}, "
          f"stroke width ~ {prep.median_stroke_width:.1f}px, "
          f"thick={prep.is_thick_stroke}")

    # Adaptive gap distance based on image size
    diag = math.sqrt(prep.image_w**2 + prep.image_h**2)
    config.feature_config.max_gap_distance = max(
        config.feature_config.max_gap_distance,
        diag * 0.12,  # up to 12% of diagonal
    )

    # ── Step 2: Extract Bézier paths ───────────────────────────────
    t0 = time.time()
    print("  [2/7] Extracting Bézier paths...")

    if prep.is_thick_stroke:
        # Contour-based fitting for thick strokes
        contours, _ = _extract_raster_contours(image_path, min_contour_area=50)
        paths = fit_from_contours(
            contours,
            max_error=config.max_bezier_error,
            tangent_lookahead=config.tangent_lookahead,
        )
        adjacency: Dict[int, set] = {}
    else:
        # Skeleton-based fitting (default)
        paths, adjacency = fit_from_image_skeleton(
            image_path,
            max_error=config.max_bezier_error,
            tangent_lookahead=config.tangent_lookahead,
            merge_radius=config.merge_radius,
        )

    timings["bezier_fit"] = time.time() - t0
    open_count = sum(1 for p in paths if not p.is_closed)
    print(f"        Paths: {len(paths)} ({open_count} open, "
          f"{len(paths) - open_count} closed)")

    # ── Step 3: Compute EFD for contour points ─────────────────────
    t0 = time.time()
    print("  [3/7] Computing EFD features...")

    contour_points_map: Dict[int, np.ndarray] = {}
    efd_data: Dict[int, np.ndarray] = {}

    try:
        import pyefd
        from eliptic_fourier_descriptors.efd import compute_efd_features
    except ImportError:
        pyefd = None
        compute_efd_features = None

    for pid, path in enumerate(paths):
        sampled = path.sample(pts_per_segment=50)
        if len(sampled) >= 5:
            contour_points_map[pid] = sampled
            if compute_efd_features is not None:
                feat = compute_efd_features(sampled, order=config.efd_order)
                if feat is not None:
                    efd_data[pid] = feat

    timings["efd"] = time.time() - t0
    print(f"        EFD vectors computed for {len(efd_data)} paths")

    # ── Step 4: Feature extraction ─────────────────────────────────
    t0 = time.time()
    print("  [4/7] Extracting features...")

    bundle = extract_all_features(
        paths,
        adjacency=adjacency,
        efd_data=efd_data,
        config=config.feature_config,
    )
    timings["features"] = time.time() - t0
    print(f"        Gaps: {len(bundle.gaps)}, "
          f"Closure candidates: {len(bundle.closure_candidates)}, "
          f"Symmetry: {'yes' if bundle.symmetry_axis else 'no'}")

    # ── Step 5: ASP inference ──────────────────────────────────────
    t0 = time.time()
    print("  [5/7] Running ASP inference...")

    asp_facts = serialize_features_to_asp(
        bundle, paths=paths, config=config.feature_config
    )

    engine = ASPInferenceEngine(max_models=config.asp_max_models)
    models = engine.solve(asp_facts)
    hypotheses = engine.rank_hypotheses(models)

    timings["asp"] = time.time() - t0

    if hypotheses:
        best = hypotheses[0]
        print(f"        Models: {len(models)}, "
              f"Best score: {best.score:.2f}, "
              f"Actions: {len(best.actions)}")
    else:
        print("        No restoration hypotheses generated")

    # ── Step 6: Execute restoration ────────────────────────────────
    t0 = time.time()
    print("  [6/7] Executing restoration...")

    result = execute_restoration(hypotheses, paths, contour_points_map)

    # ── Second pass: EFD closure for any remaining open contours ──
    # Check which paths are still open (not involved in any action)
    used_path_ids = set()
    for action in result.actions_applied:
        atype = action.get("type", "")
        if atype == "extend_curve":
            used_path_ids.add(action.get("path_a"))
            used_path_ids.add(action.get("path_b"))
        elif atype == "complete_contour":
            used_path_ids.add(action.get("path_id"))

    for pid, path in enumerate(paths):
        if pid in used_path_ids or path.is_closed:
            continue
        # Check if this is a closure candidate
        sampled = path.sample(pts_per_segment=50)
        if len(sampled) < 5:
            continue
        start_pt = sampled[0]
        end_pt = sampled[-1]
        gap_dist = float(np.linalg.norm(start_pt - end_pt))
        path_len = float(np.sum(np.linalg.norm(np.diff(sampled, axis=0), axis=1)))
        if path_len < 1e-6:
            continue
        gap_frac = gap_dist / path_len
        if gap_frac < config.feature_config.max_closure_gap_fraction and gap_dist >= config.feature_config.min_closure_gap:
            efd_arc = efd_close_contour(
                sampled, order=config.efd_order,
                num_recon_points=config.efd_recon_points,
            )
            if efd_arc is not None and len(efd_arc) >= 2:
                result.efd_arcs.append(efd_arc)
                result.actions_applied.append({
                    "type": "complete_contour",
                    "path_id": pid,
                    "method": "efd_second_pass",
                    "confidence": 0.5,
                })

    timings["synthesis"] = time.time() - t0
    total_restorations = len(result.new_segments) + len(result.efd_arcs) + len(result.new_paths)
    print(f"        Bridges: {len(result.new_segments)}, "
          f"EFD arcs: {len(result.efd_arcs)}, "
          f"New paths: {len(result.new_paths)}")

    # ── Step 7: Generate explanation ───────────────────────────────
    t0 = time.time()
    print("  [7/7] Generating explanations...")

    explanations = generate_explanation(result, bundle)
    explanation_text = format_explanation_text(explanations)
    timings["explain"] = time.time() - t0

    # ── Render output images ───────────────────────────────────────
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    overlay = _render_overlay(original, result, config)
    annotated = annotate_image(overlay, explanations, result)

    # ── Save outputs ───────────────────────────────────────────────
    orig_path = os.path.join(output_dir, f"{stem}_original.png")
    overlay_path = os.path.join(output_dir, f"{stem}_restored.png")
    sidebyside_path = os.path.join(output_dir, f"{stem}_comparison.png")

    cv2.imwrite(orig_path, original)
    cv2.imwrite(overlay_path, annotated)

    # Side-by-side comparison
    h, w = original.shape[:2]
    comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w + 20:] = annotated

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 25), font, 0.7,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(comparison, "Restored", (w + 30, 25), font, 0.7,
                (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(sidebyside_path, comparison)

    # Explanation file
    if config.save_explanation_file:
        explain_path = os.path.join(output_dir, f"{stem}_explanation.txt")
        save_explanation(explanations, explain_path)

    print(f"\n  Outputs saved to: {output_dir}/")
    print(f"    * {stem}_original.png")
    print(f"    * {stem}_restored.png")
    print(f"    * {stem}_comparison.png")
    if config.save_explanation_file:
        print(f"    * {stem}_explanation.txt")

    total_time = sum(timings.values())
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"{'=' * 60}\n")

    return PipelineResult(
        original_image=original,
        overlay_image=annotated,
        restoration=result,
        explanations=explanations,
        explanation_text=explanation_text,
        paths=paths,
        bundle=bundle,
        timing=timings,
    )


# ═══════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════

def _render_overlay(
    original: np.ndarray,
    result: RestorationResult,
    config: PipelineConfig,
) -> np.ndarray:
    """Draw restored segments on top of the original image."""
    overlay = original.copy()
    color = config.restoration_color_bgr
    thickness = config.restoration_thickness

    # Draw Bézier bridge segments
    for seg in result.new_segments:
        pts = seg.sample(100)
        pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        if len(pts_int) >= 2:
            cv2.polylines(overlay, [pts_int], False, color, thickness,
                          lineType=cv2.LINE_AA)

    # Draw EFD arcs
    for arc in result.efd_arcs:
        pts_int = np.round(arc).astype(np.int32).reshape((-1, 1, 2))
        if len(pts_int) >= 2:
            cv2.polylines(overlay, [pts_int], False, color, thickness,
                          lineType=cv2.LINE_AA)

    # Draw new paths (e.g., mirrored)
    for path in result.new_paths:
        for seg in path.segments:
            pts = seg.sample(80)
            pts_int = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            if len(pts_int) >= 2:
                cv2.polylines(overlay, [pts_int], False, color, thickness,
                              lineType=cv2.LINE_AA)

    return overlay


# Needed for the adaptive gap calculation
import math
