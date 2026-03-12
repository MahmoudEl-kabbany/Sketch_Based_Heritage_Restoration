"""
Smoke test — unified heritage restoration pipeline
====================================================
Runs the eight-stage unified pipeline on one or more test images and
writes all intermediate + comparison outputs.

Usage (from repo root):
    python sketch_preprocessing/test_unified_preprocessor.py
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root OR from inside sketch_preprocessing/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from sketch_preprocessing.unified_preprocessor import UnifiedPreprocessor  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
TEST_IMAGES = [
    os.path.join(REPO_ROOT, "test_images", "amenherkhepshef.jpg"),
    os.path.join(REPO_ROOT, "test_images", "019.png"),
    os.path.join(REPO_ROOT, "test_images", "da37.png"),
    os.path.join(REPO_ROOT, "test_images", "sketches.jpg"),
]
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_images", "unified_preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  EDIT HERE — choose medium & pipeline overrides                        │
# └─────────────────────────────────────────────────────────────────────────┘
MEDIUM = "stone"  # "paper" | "stone"
PIPELINE_KWARGS: dict = {
    # Override any default for the chosen medium, e.g.:
    # "enhance_method": "blackhat",
    # "segment_method": "grabcut",
    # "canny_low": 60,
}

# Stage metadata — keys that always exist
SAVE_MAP_BASE = {
    "original":           "0_original.png",
    "grayscale":          "1_grayscale.png",
    "denoised":           "2_denoised.png",
    "background_removed": "3_background_removed.png",
    "enhanced":           "4_enhanced.png",
}
SAVE_MAP_BINARIZE = {
    "xdog":               "6a_xdog.png",
    "sauvola":            "6b_sauvola.png",
    "niblack":            "6c_niblack.png",
    "binarized":          "7_binarized_fused.png",
    "edges":              "8_edges_canny.png",
    "final":              "9_final.png",
}
# Segmentation is conditional
SEG_KEY = ("segmentation", "5_segmentation.png")


def _build_save_map(has_seg: bool) -> dict:
    """Combine stage maps, conditionally including segmentation."""
    m = dict(SAVE_MAP_BASE)
    if has_seg:
        m[SEG_KEY[0]] = SEG_KEY[1]
    m.update(SAVE_MAP_BINARIZE)
    return m


_ENHANCE_LABELS = {
    "linear": "Linear\n(g = a·f + b)",
    "clahe": "CLAHE",
    "histogram_eq": "Histogram EQ",
    "blackhat": "Black-Hat",
}
_SEGMENT_LABELS = {
    "kmeans":    "K-means",
    "grabcut":   "GrabCut",
    "watershed": "Watershed",
}


def _thumb(arr: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Downsample for display to keep the comparison figure small."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    scale = max_px / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def run_single(image_path: str, medium: str = "stone", **kwargs) -> None:
    """Process one image and save all outputs."""
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_out = os.path.join(OUTPUT_DIR, img_name)
    os.makedirs(img_out, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Processing: {image_path}")
    print(f"  Medium:     {medium}")
    print(f"{'─' * 60}")

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  ⚠  Cannot read {image_path}, skipping.")
        return

    print(f"  Image size : {img.shape[1]}×{img.shape[0]} px")

    preprocessor = UnifiedPreprocessor(medium=medium, **kwargs)
    stages: dict = preprocessor.process(img, return_intermediates=True)  # type: ignore[assignment]
    stages = {"original": img, **stages}
    print("  Pipeline complete.")

    has_seg = "segmentation" in stages
    save_map = _build_save_map(has_seg)

    # ── Print active parameters ────────────────────────────────────────
    print(f"\n  ▶ Parameters")
    print(f"    medium            = {medium}")
    print(f"    enhance_method    = {preprocessor.enhance_method}")
    print(f"    segment_method    = {preprocessor.segment_method}")
    print(f"    segment_as_voter  = {preprocessor.segment_as_voter}")
    print(f"    min_votes         = {preprocessor.min_votes}")
    print(f"    canny_low         = {preprocessor.canny_low}")
    print(f"    canny_high        = {preprocessor.canny_high}")

    # ── Save individual stage images ───────────────────────────────────
    print(f"\n  ▶ Saving stage images → {img_out}/")
    for key, fname in save_map.items():
        if key in stages:
            path = os.path.join(img_out, fname)
            cv2.imwrite(path, stages[key])
            print(f"    ➜  {fname}")

    # ── Statistics table ───────────────────────────────────────────────
    print(f"\n  ▶ Stage statistics")
    print(f"    {'Stage':<22} {'Min':>5}  {'Max':>5}  {'Mean':>8}")
    print(f"    {'-' * 48}")
    for key in save_map:
        if key not in stages:
            continue
        arr = stages[key]
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        print(f"    {key:<22} {int(arr.min()):>5}  {int(arr.max()):>5}  {arr.mean():>8.2f}")

    # ── Foreground coverage metric ─────────────────────────────────────
    final = stages["final"]
    fg_pixels = int(np.count_nonzero(final))
    total = final.shape[0] * final.shape[1]
    print(f"\n  ▶ Foreground coverage")
    print(f"    Foreground pixels : {fg_pixels:,} / {total:,}  "
          f"({100 * fg_pixels / total:.2f}%)")

    # ── Side-by-side comparison figure ─────────────────────────────────
    fig_path = os.path.join(img_out, "pipeline_comparison.png")
    print(f"\n  ▶ Saving comparison figure → {fig_path}")

    keys_in_order = list(save_map.keys())
    n = len(keys_in_order)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    # Build titles dynamically
    enh_label = _ENHANCE_LABELS.get(preprocessor.enhance_method, "?")
    seg_label = _SEGMENT_LABELS.get(preprocessor.segment_method or "", "—")
    title_map = {
        "original":           "Original",
        "grayscale":          "1 · Grayscale",
        "denoised":           "2 · Denoised\n(NLM + Bilateral)",
        "background_removed": "3 · Background\nRemoved",
        "enhanced":           f"4 · Enhanced\n({enh_label})",
        "segmentation":       f"5 · Segmentation\n({seg_label})",
        "xdog":               "6a · XDoG",
        "sauvola":            "6b · Sauvola",
        "niblack":            "6c · Niblack",
        "binarized":          "7 · Binarized\n(majority vote)",
        "edges":              "8 · Edges\n(Canny)",
        "final":              "9 · Final\n(fused + cleaned)",
    }

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Unified Preprocessing Pipeline  —  {img_name}  (medium={medium})",
        fontsize=12, y=1.01,
    )

    for idx, key in enumerate(keys_in_order):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        display = stages[key]
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ax.imshow(_thumb(display), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title_map.get(key, key), fontsize=8)
        ax.axis("off")

    # Hide unused subplot cells
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ➜  {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 62}")
    print("  Unified Heritage Restoration Pipeline — smoke test")
    print(f"{'═' * 62}")

    found = [p for p in TEST_IMAGES if os.path.exists(p)]
    if not found:
        print("\n  ⚠  No test images found, aborting.")
        sys.exit(1)

    print(f"\n  Found {len(found)} test image(s).")

    for path in found:
        run_single(path, medium=MEDIUM, **PIPELINE_KWARGS)

    print(f"\n{'═' * 62}")
    print(f"  All done — outputs in {OUTPUT_DIR}/")
    print(f"{'═' * 62}\n")
