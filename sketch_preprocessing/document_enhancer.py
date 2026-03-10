"""
Document Enhancement Module
============================
Five-step pipeline for historical document image restoration:

  1. Denoising       — Wiener filter (5×5 kernel)                       [Eq. 1]
  2. Texture         — CLAHE (8-tile grid, clip=0.03)                   [Eqs. 2–5]
                       + LBP (P=8, R=1) + local stats combination       [Eqs. 6–8]
                       + Adaptive Gaussian filtering
  3. Illumination    — Multi-Scale Retinex (σ = 15, 80, 250)            [Eqs. 9–13]
  4. Binarization    — Sauvola adaptive threshold (15×15, k=−0.2)       [Eqs. 14–15]
  5. Post-processing — Morphological opening (3×3 kernel, 1 iter)       [Eqs. 16–18]

Reference:
  "Enhancing Deteriorated Images of Historical Documents"
  J. Electrical Systems 20-03 (2024): 4779–4796
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.signal import wiener
from skimage.exposure import equalize_adapthist
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_sauvola


# ═══════════════════════════════════════════════════════════════════════════
# DocumentEnhancer
# ═══════════════════════════════════════════════════════════════════════════

class DocumentEnhancer:
    """Five-step enhancement pipeline for degraded historical document images."""

    def __init__(
        self,
        # ── Step 1: Wiener ──────────────────────────────────────────────
        wiener_size: int = 5,
        # ── Step 2: CLAHE ───────────────────────────────────────────────
        clahe_tile: int = 8,            # number of tiles per dimension
        clahe_clip: float = 0.03,       # normalised clip limit
        # ── Step 2: LBP ─────────────────────────────────────────────────
        lbp_points: int = 8,
        lbp_radius: int = 1,
        lbp_alpha: float = 10.0,        # LBP contribution scale in Eq. 8
        lbp_window: int = 15,           # local-stats neighbourhood size
        lbp_sigma_heavy: float = 3.0,   # Gaussian σ for low-variance (flat) regions
        lbp_sigma_light: float = 1.0,   # Gaussian σ for high-variance (edge) regions
        # ── Step 3: MSR ─────────────────────────────────────────────────
        msr_sigmas: Tuple[float, ...] = (15.0, 80.0, 250.0),
        # ── Step 4: Sauvola ─────────────────────────────────────────────
        sauvola_window: int = 15,       # must be odd
        sauvola_k: float = 0.2,         # positive in skimage's convention (paper uses opposite sign)
        # ── Step 5: Morphological ───────────────────────────────────────
        morph_kernel_size: int = 3,
    ):
        self.wiener_size = wiener_size

        self.clahe_tile = clahe_tile
        self.clahe_clip = clahe_clip

        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.lbp_alpha = lbp_alpha
        self.lbp_window = lbp_window
        self.lbp_sigma_heavy = lbp_sigma_heavy
        self.lbp_sigma_light = lbp_sigma_light

        self.msr_sigmas = msr_sigmas

        self.sauvola_window = sauvola_window
        self.sauvola_k = sauvola_k

        self.morph_kernel_size = morph_kernel_size

    # ───────────────────────────────────────────────────────────────────────
    # Step 1 — Wiener filtering  (Eq. 1)
    # ───────────────────────────────────────────────────────────────────────

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Reduce noise with a Wiener filter.

        Eq. 1: weighted average between the original pixel value and the
        local-block mean intensity, scaled by local vs. global variance.

        Args:
            img: Grayscale uint8 image.
        Returns:
            Denoised uint8 image.
        """
        filtered = wiener(img.astype(np.float32), mysize=self.wiener_size)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Step 2 — Texture enhancement  (Eqs. 2–8)
    # ───────────────────────────────────────────────────────────────────────

    def enhance_texture(self, img: np.ndarray) -> np.ndarray:
        """Enhance texture using CLAHE, LBP, and adaptive Gaussian filtering.

        Eqs. 2–5  CLAHE: 8-tile grid, normalised clip limit 0.03.
        Eq.  6    LBP code (P=8, R=1, uniform mapping).
        Eq.  7    Binary sign function encoded into LBP values.
        Eq.  8    Combined enhancement:
                    E = μ + w·(I − μ) + α·LBP̂
                  where w = σ²/(σ²+C) is a Wiener-style adaptive weight,
                  μ and σ² are local mean and variance, and LBP̂ ∈ [0,1].
        Adaptive Gaussian: blend heavy blur (flat areas) and light blur
                  (textured areas) using the same weight w.

        Args:
            img: Grayscale uint8 image (typically Wiener-filtered).
        Returns:
            Texture-enhanced uint8 image.
        """
        # ── Eqs. 2–5: CLAHE ──────────────────────────────────────────────
        h, w_img = img.shape[:2]
        kernel_size = (
            max(1, h // self.clahe_tile),
            max(1, w_img // self.clahe_tile),
        )
        clahe_float = equalize_adapthist(
            img, kernel_size=kernel_size, clip_limit=self.clahe_clip
        )  # returns float64 in [0, 1]
        clahe_img = (clahe_float * 255.0)  # keep as float64 for later maths

        # ── Eqs. 6–7: LBP (P=8, R=1, uniform) ───────────────────────────
        lbp = local_binary_pattern(
            clahe_img.astype(np.uint8),
            P=self.lbp_points,
            R=self.lbp_radius,
            method="uniform",
        )
        lbp_max = lbp.max()
        lbp_norm = lbp / lbp_max if lbp_max > 1e-8 else np.zeros_like(lbp)  # [0, 1]

        # ── Eq. 8: local statistics for adaptive weighting ────────────────
        win = self.lbp_window
        img_f = clahe_img  # float64 in [0, 255]
        local_mean = cv2.blur(img_f, (win, win))
        local_mean_sq = cv2.blur(img_f ** 2, (win, win))
        local_var = np.maximum(local_mean_sq - local_mean ** 2, 0.0)

        # noise constant C: mean of the local variance (Wiener-style estimator)
        C = float(np.mean(local_var)) + 1e-8
        w = local_var / (local_var + C)  # adaptive weight ∈ [0, 1]

        # Eq. 8: E = μ + w·(I − μ) + α·LBP̂
        combined = local_mean + w * (img_f - local_mean) + self.lbp_alpha * lbp_norm
        combined = np.clip(combined, 0.0, 255.0)

        # ── Adaptive Gaussian ─────────────────────────────────────────────
        # Flat regions (w≈0)  → heavy smoothing (noise suppression)
        # Edge regions (w≈1)  → light smoothing (detail preservation)
        heavy = cv2.GaussianBlur(combined, (0, 0), self.lbp_sigma_heavy)
        light = cv2.GaussianBlur(combined, (0, 0), self.lbp_sigma_light)
        result = (1.0 - w) * heavy + w * light

        return np.clip(result, 0, 255).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Step 3 — Illumination correction  (Eqs. 9–13)
    # ───────────────────────────────────────────────────────────────────────

    def correct_illumination(self, img: np.ndarray) -> np.ndarray:
        """Remove uneven illumination with Multi-Scale Retinex.

        Eq.  9  / 12  log(I) − log(L)  [reflectance in log domain]
        Eq. 10        Gaussian smoothing to estimate the illumination L.
        Eq. 13        Average across scales and normalise to [0, 255].

        Args:
            img: Grayscale uint8 image (texture-enhanced).
        Returns:
            Illumination-corrected uint8 image.
        """
        img_f = img.astype(np.float64) + 1.0  # avoid log(0)
        log_img = np.log(img_f)

        msr = np.zeros_like(log_img)
        for sigma in self.msr_sigmas:
            blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)  # Eq. 10
            msr += log_img - np.log(blurred + 1.0)             # Eq. 9 / 12

        msr /= len(self.msr_sigmas)  # Eq. 13: average across scales

        # Normalise to [0, 255]
        msr_min, msr_max = msr.min(), msr.max()
        if msr_max - msr_min > 1e-8:
            msr = (msr - msr_min) / (msr_max - msr_min) * 255.0
        else:
            msr = np.zeros_like(msr)

        return np.clip(msr, 0, 255).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Step 4 — Binarization  (Eqs. 14–15)
    # ───────────────────────────────────────────────────────────────────────

    def binarize(self, img: np.ndarray) -> np.ndarray:
        """Segment foreground (text) from background using Sauvola's method.

        Eq. 14  T = μ · [1 + k · (σ/R − 1)]
                  window_size = 15×15, k = −0.2
                  Negative k raises the threshold above the local mean,
                  capturing faint/degraded text strokes.
        Eq. 15  Binary decision: foreground (text) where I ≤ T.

        Output convention (matches paper): foreground = 255, background = 0.

        Args:
            img: Grayscale uint8 image (illumination-corrected).
        Returns:
            Binary uint8 image, white foreground on black background.
        """
        thresh = threshold_sauvola(
            img, window_size=self.sauvola_window, k=self.sauvola_k
        )
        # Pixels at or below the local threshold are darker → text (foreground)
        binary = (img <= thresh).astype(np.uint8) * 255
        return binary

    # ───────────────────────────────────────────────────────────────────────
    # Step 5 — Post-processing  (Eqs. 16–18)
    # ───────────────────────────────────────────────────────────────────────

    def postprocess(self, binary: np.ndarray) -> np.ndarray:
        """Refine binary image with morphological opening.

        Eq. 16  Erosion:  shrinks foreground boundaries.
        Eq. 17  Dilation: expands surviving foreground, fills small gaps.
        Eq. 18  Opening (erosion ∘ dilation): removes small noise objects
                while preserving larger text strokes.

        3×3 rectangular kernel, 1 iteration (as specified in the paper).

        Args:
            binary: Binary uint8 image from the binarization step.
        Returns:
            Refined binary uint8 image.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # ───────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ───────────────────────────────────────────────────────────────────────

    def process(
        self,
        img: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run the complete five-step pipeline.

        Args:
            img: Grayscale uint8 input image.
            return_intermediates: When True, return a dict of all stage
                outputs keyed by stage name.
        Returns:
            Final binary image, or dict with keys:
            'denoised', 'texture', 'illumination', 'binarized', 'final'.
        """
        denoised    = self.denoise(img)
        texture     = self.enhance_texture(denoised)
        illumination = self.correct_illumination(texture)
        binarized   = self.binarize(illumination)
        final       = self.postprocess(binarized)

        if return_intermediates:
            return {
                "denoised":     denoised,
                "texture":      texture,
                "illumination": illumination,
                "binarized":    binarized,
                "final":        final,
            }
        return final


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def enhance_document(
    image_path: str,
    output_dir: Optional[str] = None,
    save_intermediates: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Load an image, run the full pipeline, and optionally save each stage.

    Args:
        image_path:        Path to a grayscale or colour image file.
        output_dir:        Directory to write output PNGs (created if needed).
        save_intermediates: When True, save all five stage images to
                           *output_dir* (requires *output_dir* to be set).
        **kwargs:          Forwarded to :class:`DocumentEnhancer.__init__`.
    Returns:
        Dict with keys: 'original', 'denoised', 'texture',
        'illumination', 'binarized', 'final'.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    enhancer = DocumentEnhancer(**kwargs)
    stages: Dict[str, np.ndarray] = enhancer.process(img, return_intermediates=True)  # type: ignore[assignment]
    stages["original"] = img

    if save_intermediates and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filenames = {
            "denoised":     "1_denoised.png",
            "texture":      "2_texture.png",
            "illumination": "3_illumination.png",
            "binarized":    "4_binarized.png",
            "final":        "5_final.png",
        }
        for key, fname in filenames.items():
            cv2.imwrite(os.path.join(output_dir, fname), stages[key])

    return stages
