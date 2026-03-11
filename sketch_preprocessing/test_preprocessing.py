"""
Smoke test — historical document enhancement pipeline
=====================================================
Runs the five-step pipeline on test_images/da80.png and writes all
intermediate + comparison outputs to test_images/preprocessed/.

Usage (from repo root):
    python sketch_preprocessing/test_preprocessing.py
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root OR from inside sketch_preprocessing/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from sketch_preprocessing.document_enhancer import DocumentEnhancer  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
IMAGE      = os.path.join(REPO_ROOT, "test_images", "da37.png")
IMG_NAME   = os.path.splitext(os.path.basename(IMAGE))[0]
OUTPUT_DIR = os.path.join(REPO_ROOT, "test_images", "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────
print(f"\n{'═' * 62}")
print("  Historical Document Enhancement Pipeline — smoke test")
print(f"{'═' * 62}")

if not os.path.exists(IMAGE):
    print(f"\n  ⚠  {IMAGE} not found, aborting.")
    sys.exit(1)

# ── Load image ─────────────────────────────────────────────────────────────
print(f"\n▶ Loading  {IMAGE}")
img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
print(f"  Image size : {img.shape[1]}×{img.shape[0]} px")

# ── Run pipeline ───────────────────────────────────────────────────────────
print("\n▶ Running pipeline …")
enhancer = DocumentEnhancer()
stages: dict = enhancer.process(img, return_intermediates=True)  # type: ignore[assignment]
stages = {"original": img, **stages}
print("  Done.")

# ── Save individual stage images ───────────────────────────────────────────
print("\n▶ Saving stage images → test_images/preprocessed/")

SAVE_MAP = {
    "original":     "0_original.png",
    "denoised":     "1_denoised.png",
    "texture":      "2_texture.png",
    "illumination": "3_illumination.png",
    "binarized":    "4_binarized.png",
    "final":        "5_final.png",
}

for key, fname in SAVE_MAP.items():
    path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(path, stages[key])
    print(f"  ➜  {path}")

# ── Statistics table ───────────────────────────────────────────────────────
print(f"\n▶ Stage statistics")
print(f"  {'Stage':<16} {'Min':>5}  {'Max':>5}  {'Mean':>8}")
print(f"  {'-' * 42}")
for key in SAVE_MAP:
    arr = stages[key]
    print(f"  {key:<16} {int(arr.min()):>5}  {int(arr.max()):>5}  {arr.mean():>8.2f}")

# ── Side-by-side comparison figure ────────────────────────────────────────
FIG_PATH = os.path.join(OUTPUT_DIR, "pipeline_comparison.png")
print(f"\n▶ Saving comparison figure → {FIG_PATH}")

TITLES = [
    "Original",
    "Step 1 · Denoised\n(Wiener 5×5)",
    "Step 2 · Texture\n(CLAHE + LBP + Adaptive Gaussian)",
    "Step 3 · Illumination\n(Multi-Scale Retinex)",
    "Step 4 · Binarized\n(Sauvola  window=15, k=\u22120.2)",
    "Step 5 · Final\n(Morphological Opening 3×3)",
]

# Downsample display copies to keep the figure file small on large images
MAX_DISPLAY_PX = 512
def _thumb(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= MAX_DISPLAY_PX:
        return arr
    scale = MAX_DISPLAY_PX / max(h, w)
    return cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(1, 6, figsize=(24, 5))
fig.suptitle(
    f"Historical Document Enhancement Pipeline  —  {IMG_NAME}",
    fontsize=12,
    y=1.01,
)

for ax, key, title in zip(axes, SAVE_MAP.keys(), TITLES):
    ax.imshow(_thumb(stages[key]), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=7.5)
    ax.axis("off")

plt.tight_layout()
fig.savefig(FIG_PATH, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"  ➜  {FIG_PATH}")

print(f"\n{'═' * 62}")
print("  All done — 6 images written to test_images/preprocessed/")
print(f"{'═' * 62}\n")
