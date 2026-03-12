"""
Smoke test — stone inscription restoration pipeline
=====================================================
Runs the two-stage pipeline (linear enhancement + Canny edge detection)
on one or more test images and writes all intermediate + comparison outputs.

Usage (from repo root):
    python sketch_preprocessing/test_preprocessing_stone.py

To choose a different image, edit the TEST_IMAGES list below.
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root OR from inside sketch_preprocessing/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from sketch_preprocessing.preprocessing_stone import StoneInscriptionRestorer  # noqa: E402

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  EDIT HERE — add or replace image paths to test different images       │
# └─────────────────────────────────────────────────────────────────────────┘
TEST_IMAGES = [
    os.path.join(REPO_ROOT, "test_images", "amenherkhepshef.jpg"),
    os.path.join(REPO_ROOT, "test_images", "019.png"),
    os.path.join(REPO_ROOT, "test_images", "da37.png"),
    os.path.join(REPO_ROOT, "test_images", "sketches.jpg"),
]

# Output root — each image gets its own sub-folder
OUTPUT_DIR = os.path.join(REPO_ROOT, "contour_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  EDIT HERE — tweak pipeline parameters (or leave as None for defaults) │
# └─────────────────────────────────────────────────────────────────────────┘
PIPELINE_KWARGS = {
    "enhance_method": "blackhat",  # "linear" | "clahe" | "histogram_eq" | "blackhat"
    "binarize_method": "sauvola",  # None | "otsu" | "adaptive_gaussian" | "sauvola" | "kmeans"
    # --- linear params (used when enhance_method="linear") ---
    # "gain": 1.5,              # contrast multiplier  (default 1.5)
    # "bias": 20,               # brightness offset    (default 20)
    # --- clahe params (used when enhance_method="clahe") ---
    # "clahe_clip": 2.0,        # CLAHE clip limit     (default 2.0)
    # "clahe_tile": 8,          # CLAHE tile grid size (default 8)
    # --- blackhat params (used when enhance_method="blackhat") ---
    # "blackhat_ksize": 51,     # structuring element  (default 51)
    # --- adaptive_gaussian params ---
    # "adaptive_block": 35,     # neighbourhood size   (default 35)
    # "adaptive_c": 10,         # constant subtracted  (default 10)
    # --- sauvola params ---
    # "sauvola_window": 35,     # window size          (default 35)
    # "sauvola_k": 0.2,         # sensitivity          (default 0.2)
    # --- kmeans params ---
    # "kmeans_k": 3,            # number of clusters   (default 3)
    # --- canny params (always used) ---
    "canny_low": 60,            # lower Canny threshold (default 30)
    "canny_high": 100,          # upper Canny threshold (default 100)
    "gaussian_ksize": 5,        # Gaussian blur kernel  (default 5)
    "blur_sigma": 1.4,          # Gaussian blur sigma   (default 1.4)
}

# Stage metadata for saving and display — built dynamically based on config
_ENHANCE_LABELS = {
    "linear": "Linear\n(g = a·f + b)",
    "clahe": "CLAHE",
    "histogram_eq": "Histogram EQ",
    "blackhat": "Black-Hat",
}
_BINARIZE_LABELS = {
    "otsu": "Otsu",
    "adaptive_gaussian": "Adaptive\nGaussian",
    "sauvola": "Sauvola",
    "kmeans": "K-means",
}

_enhance_tag = _ENHANCE_LABELS.get(
    PIPELINE_KWARGS.get("enhance_method", "clahe"), "?")
_bin_method = PIPELINE_KWARGS.get("binarize_method")

# Base stages (always present)
SAVE_MAP = {
    "original":  "0_original.png",
    "grayscale": "1_grayscale.png",
    "enhanced":  "2_enhanced.png",
}
TITLES = [
    "Original",
    "1 · Grayscale",
    f"2 · Enhanced\n({_enhance_tag})",
]

# Conditionally add binarization stage
if _bin_method is not None:
    _bin_tag = _BINARIZE_LABELS.get(_bin_method, "?")
    SAVE_MAP["binarized"] = "3_binarized.png"
    TITLES.append(f"3 · Binarized\n({_bin_tag})")
    SAVE_MAP["edges"] = "4_edges.png"
    TITLES.append("4 · Edges\n(Canny)")
else:
    SAVE_MAP["edges"] = "3_edges.png"
    TITLES.append("3 · Edges\n(Canny)")


def _thumb(arr: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Downsample for display to keep the comparison figure small."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    scale = max_px / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def run_single(image_path: str) -> None:
    """Process one image and save all outputs."""
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_out = os.path.join(OUTPUT_DIR, img_name)
    os.makedirs(img_out, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Processing: {image_path}")
    print(f"{'─' * 60}")

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  ⚠  Cannot read {image_path}, skipping.")
        return

    print(f"  Image size : {img.shape[1]}×{img.shape[0]} px")

    # ── Run pipeline ───────────────────────────────────────────────────
    restorer = StoneInscriptionRestorer(**PIPELINE_KWARGS)
    stages: dict = restorer.process(img, return_intermediates=True)
    stages = {"original": img, **stages}
    print("  Pipeline complete.")

    # ── Print active parameters ────────────────────────────────────────
    print(f"\n  ▶ Parameters")
    print(f"    enhance_method  = {restorer.enhance_method}")
    if restorer.enhance_method == "linear":
        print(f"    gain (a)        = {restorer.gain}")
        print(f"    bias (b)        = {restorer.bias}")
    elif restorer.enhance_method == "clahe":
        print(f"    clahe_clip      = {restorer.clahe_clip}")
        print(f"    clahe_tile      = {restorer.clahe_tile}")
    elif restorer.enhance_method == "blackhat":
        print(f"    blackhat_ksize  = {restorer.blackhat_ksize}")
    print(f"    binarize_method = {restorer.binarize_method}")
    if restorer.binarize_method == "adaptive_gaussian":
        print(f"    adaptive_block  = {restorer.adaptive_block}")
        print(f"    adaptive_c      = {restorer.adaptive_c}")
    elif restorer.binarize_method == "sauvola":
        print(f"    sauvola_window  = {restorer.sauvola_window}")
        print(f"    sauvola_k       = {restorer.sauvola_k}")
    elif restorer.binarize_method == "kmeans":
        print(f"    kmeans_k        = {restorer.kmeans_k}")
    print(f"    canny_low       = {restorer.canny_low}")
    print(f"    canny_high      = {restorer.canny_high}")
    print(f"    gaussian_ksize  = {restorer.gaussian_ksize}")
    print(f"    blur_sigma      = {restorer.blur_sigma}")

    # ── Save individual stage images ───────────────────────────────────
    print(f"\n  ▶ Saving stage images → {img_out}/")
    for key, fname in SAVE_MAP.items():
        path = os.path.join(img_out, fname)
        cv2.imwrite(path, stages[key])
        print(f"    ➜  {fname}")

    # ── Statistics table ───────────────────────────────────────────────
    print(f"\n  ▶ Stage statistics")
    print(f"    {'Stage':<14} {'Min':>5}  {'Max':>5}  {'Mean':>8}")
    print(f"    {'-' * 40}")
    for key in SAVE_MAP:
        arr = stages[key]
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        print(f"    {key:<14} {int(arr.min()):>5}  {int(arr.max()):>5}  {arr.mean():>8.2f}")

    # ── Edge pixel count ───────────────────────────────────────────────
    edges = stages["edges"]
    edge_pixels = int(np.count_nonzero(edges))
    total = edges.shape[0] * edges.shape[1]
    print(f"\n  ▶ Edge coverage")
    print(f"    Edge pixels : {edge_pixels:,} / {total:,}  "
          f"({100 * edge_pixels / total:.2f}%)")

    # ── Side-by-side comparison figure ─────────────────────────────────
    fig_path = os.path.join(img_out, "pipeline_comparison.png")
    print(f"\n  ▶ Saving comparison figure → {fig_path}")

    n = len(SAVE_MAP)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes] if n == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    fig.suptitle(
        f"Stone Inscription Restoration  —  {img_name}",
        fontsize=13, y=1.02,
    )

    for idx, (key, title) in enumerate(zip(SAVE_MAP.keys(), TITLES)):
        ax = axes[idx]
        display = stages[key]
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ax.imshow(_thumb(display), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    # Hide unused subplot slots
    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ➜  {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 62}")
    print("  Stone Inscription Restoration Pipeline — smoke test")
    print(f"{'═' * 62}")

    found = [p for p in TEST_IMAGES if os.path.exists(p)]
    if not found:
        print("\n  ⚠  No test images found. Edit TEST_IMAGES at the top of this file.")
        sys.exit(1)

    print(f"\n  Found {len(found)} test image(s).")

    for path in found:
        run_single(path)

    print(f"\n{'═' * 62}")
    print("  Done. Check outputs in contour_outputs/")
    print(f"{'═' * 62}\n")
