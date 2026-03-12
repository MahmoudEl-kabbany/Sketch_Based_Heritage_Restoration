"""
Smoke test — sketch preprocessing pipeline
============================================
Runs the seven-stage sketch pipeline on one or more test images and writes
all intermediate + comparison outputs.

Usage (from repo root):
    python sketch_preprocessing/test_sketch_preprocessor.py
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root OR from inside sketch_preprocessing/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from sketch_preprocessing.sketch_preprocessor import SketchPreprocessor  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
TEST_IMAGES = [
    os.path.join(REPO_ROOT, "test_images", "amenherkhepshef.jpg"),
    os.path.join(REPO_ROOT, "test_images", "019.png"),
]
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_images", "sketch_preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stage metadata for saving and display
SAVE_MAP = {
    "original":           "0_original.png",
    "grayscale":          "1_grayscale.png",
    "denoised":           "2_denoised.png",
    "background_removed": "3_background_removed.png",
    "contrast":           "4_contrast.png",
    "xdog":               "5a_xdog.png",
    "sauvola":            "5b_sauvola.png",
    "niblack":            "5c_niblack.png",
    "binarized":          "6_binarized_fused.png",
    "final":              "7_final.png",
}

TITLES = [
    "Original",
    "1 · Grayscale",
    "2 · Denoised\n(NLM + Bilateral)",
    "3 · Background\nRemoved",
    "4 · Contrast\n(CLAHE)",
    "5a · XDoG",
    "5b · Sauvola",
    "5c · Niblack",
    "6 · Fused\n(majority vote)",
    "7 · Final\n(clean-up)",
]


def _thumb(arr: np.ndarray, max_px: int = 512) -> np.ndarray:
    """Downsample for display to keep the comparison figure small."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    scale = max_px / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def run_single(image_path: str, medium: str = "paper") -> None:
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

    preprocessor = SketchPreprocessor(medium=medium)
    stages: dict = preprocessor.process(img, return_intermediates=True)  # type: ignore[assignment]
    stages = {"original": img, **stages}
    print("  Pipeline complete.")

    # ── Save individual stage images ───────────────────────────────────
    print(f"\n  ▶ Saving stage images → {img_out}/")
    for key, fname in SAVE_MAP.items():
        path = os.path.join(img_out, fname)
        cv2.imwrite(path, stages[key])
        print(f"    ➜  {fname}")

    # ── Statistics table ───────────────────────────────────────────────
    print(f"\n  ▶ Stage statistics")
    print(f"    {'Stage':<22} {'Min':>5}  {'Max':>5}  {'Mean':>8}")
    print(f"    {'-' * 48}")
    for key in SAVE_MAP:
        arr = stages[key]
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        print(f"    {key:<22} {int(arr.min()):>5}  {int(arr.max()):>5}  {arr.mean():>8.2f}")

    # ── Line preservation metric ───────────────────────────────────────
    final = stages["final"]
    fg_pixels = int(np.count_nonzero(final))
    total = final.shape[0] * final.shape[1]
    print(f"\n  ▶ Line preservation")
    print(f"    Foreground pixels : {fg_pixels:,} / {total:,}  "
          f"({100 * fg_pixels / total:.2f}%)")

    # ── Side-by-side comparison figure ─────────────────────────────────
    fig_path = os.path.join(img_out, "pipeline_comparison.png")
    print(f"\n  ▶ Saving comparison figure → {fig_path}")

    n_stages = len(SAVE_MAP)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(
        f"Sketch Preprocessing Pipeline  —  {img_name}  (medium={medium})",
        fontsize=12, y=1.01,
    )

    for idx, (key, title) in enumerate(zip(SAVE_MAP.keys(), TITLES)):
        ax = axes[idx // 5, idx % 5]
        display = stages[key]
        if display.ndim == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ax.imshow(_thumb(display), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    ➜  {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'═' * 62}")
    print("  Sketch Preprocessing Pipeline — smoke test")
    print(f"{'═' * 62}")

    found = [p for p in TEST_IMAGES if os.path.exists(p)]
    if not found:
        print("\n  ⚠  No test images found, aborting.")
        sys.exit(1)

    print(f"\n  Found {len(found)} test image(s).")

    for path in found:
        run_single(path, medium="paper")

    print(f"\n{'═' * 62}")
    print(f"  All done — outputs in {OUTPUT_DIR}/")
    print(f"{'═' * 62}\n")
