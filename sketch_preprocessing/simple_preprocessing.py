"""
Simple Binarization + Skeletonization Pipeline
================================================
A minimal two-step approach:
  1. Otsu thresholding  — dark strokes on light background → binary foreground mask
  2. Skeletonization    — collapses thick strokes to 1-pixel centerlines (Zhang–Suen)

Usage (from repo root):
    python sketch_preprocessing/simple_preprocessing.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from skimage.morphology import skeletonize

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_images", "preprocessed")


# ═══════════════════════════════════════════════════════════════════════════
# Core pipeline function
# ═══════════════════════════════════════════════════════════════════════════

def simple_preprocess(image_path: str) -> dict[str, np.ndarray]:
    """Load an image, binarize with Otsu, then skeletonize.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the source image (any format OpenCV
        can read).

    Returns
    -------
    dict with keys:
        ``"grayscale"``  — uint8 grayscale (0–255)
        ``"binarized"``  — uint8 binary, foreground = 255 (Otsu + invert)
        ``"skeleton"``   — uint8 binary, 1-px centerlines = 255
    """
    # ── Stage 0 · Load as grayscale ───────────────────────────────────────
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # ── Stage 1 · Mild Gaussian blur (noise reduction before thresholding) ─
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)

    # ── Stage 2 · Otsu binarization (dark strokes → foreground = 255) ─────
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # ── Stage 3 · Skeletonize (Zhang–Suen) ───────────────────────────────
    bool_mask = binary.astype(bool)
    skeleton_bool = skeletonize(bool_mask)
    skeleton = (skeleton_bool * 255).astype(np.uint8)

    return {
        "grayscale": gray,
        "binarized": binary,
        "skeleton": skeleton,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test / __main__ block
# ═══════════════════════════════════════════════════════════════════════════

TEST_IMAGES = [
    os.path.join(REPO_ROOT, "test_images", "019.PNG"),
    os.path.join(REPO_ROOT, "test_images", "amenherkhepshef.jpg"),
    os.path.join(REPO_ROOT, "test_images", "sketches.jpg"),
    os.path.join(REPO_ROOT, "test_images", "da80.png"),
]

STAGE_LABELS = {
    "grayscale": "0_grayscale",
    "binarized": "1_binarized",
    "skeleton":  "2_skeleton",
}

DISPLAY_TITLES = ["Original (grayscale)", "Binarized (Otsu)", "Skeleton (Zhang–Suen)"]


def _fg_percent(arr: np.ndarray) -> float:
    """Return percentage of foreground (255) pixels in a uint8 array."""
    return float(np.count_nonzero(arr)) / arr.size * 100.0


def _thumb(arr: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Downsample for display so the comparison figure stays compact."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    scale = max_px / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def run_single(image_path: str) -> None:
    """Process one image, save stage PNGs, and save a comparison figure."""
    import matplotlib
    matplotlib.use("Agg")           # headless — no display required
    import matplotlib.pyplot as plt

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Image : {image_path}")
    print(f"{'─' * 60}")

    if not os.path.exists(image_path):
        print(f"  ⚠  File not found, skipping.")
        return

    stages = simple_preprocess(image_path)
    h, w = stages["grayscale"].shape
    print(f"  Size  : {w}×{h} px")

    # ── Save individual stage PNGs ─────────────────────────────────────────
    print(f"\n  Saving stage images → {OUTPUT_DIR}/")
    for key, suffix in STAGE_LABELS.items():
        fname = f"simple_{img_name}_{suffix}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        cv2.imwrite(out_path, stages[key])
        arr = stages[key]
        fg = _fg_percent(arr)
        print(
            f"    {suffix:<20}  min={int(arr.min()):>3}  max={int(arr.max()):>3}"
            f"  mean={arr.mean():>7.2f}  fg={fg:>6.2f}%   → {fname}"
        )

    # ── Matplotlib comparison figure ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Simple Pipeline — {img_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, (key, _), title in zip(axes, STAGE_LABELS.items(), DISPLAY_TITLES):
        thumb = _thumb(stages[key])
        ax.imshow(thumb, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"simple_{img_name}_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison figure → {fig_path}")


if __name__ == "__main__":
    print(f"\n{'═' * 60}")
    print("  Simple Binarization + Skeletonization — test run")
    print(f"{'═' * 60}")

    for img_path in TEST_IMAGES:
        run_single(img_path)

    print(f"\n{'═' * 60}")
    print("  Done.  All outputs written to test_images/preprocessed/")
    print(f"{'═' * 60}\n")
