"""
Sketch Preprocessing Module
============================
Seven-stage pipeline optimised for ancient sketch line preservation,
supporting both paper sketches and stone engravings.

  0. Input handling   — Load, convert to grayscale
  1. Denoising        — Non-Local Means + Bilateral Filter (edge-preserving)
  2. Background       — Morphological closing estimation + subtraction
  3. Contrast         — CLAHE (Contrast Limited Adaptive Histogram Equalisation)
  4. Binarization     — Multi-strategy fusion: XDoG + Sauvola + Niblack
  5. Post-processing  — Small-component removal + morphological closing + optional thinning
  6. Output           — Binary image (foreground = 255, background = 0)

Design rationale
----------------
The existing ``DocumentEnhancer`` is optimised for degraded *text* documents
and employs morphological opening that can destroy thin sketch lines.  This
module instead:

* Uses edge-preserving denoisers (NLM + bilateral) to protect faint strokes.
* Subtracts an estimated background surface to remove yellowing, staining,
  and uneven illumination in one step — more effective than Multi-Scale Retinex
  for non-text, line-art content.
* Fuses three complementary binarization strategies via majority voting so
  that faint lines captured by *any two* methods survive while isolated
  false-positives from a single method are suppressed.

Key references
--------------
* Winnemoeller, Kyprianidis & Olsen, "XDoG: An eXtended Difference-of-
  Gaussians Compendium including Advanced Image Stylization",
  Computers & Graphics 36(6), 2012.
* Sauvola & Pietikainen, "Adaptive document image binarization",
  Pattern Recognition 33(2), 2000.
* Niblack, "An Introduction to Digital Image Processing", Prentice-Hall, 1986.
* Buades, Coll & Morel, "A Non-Local Algorithm for Image Denoising",
  CVPR 2005.
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_niblack, threshold_sauvola
from skimage.morphology import remove_small_objects, thin


# ═══════════════════════════════════════════════════════════════════════════
# SketchPreprocessor
# ═══════════════════════════════════════════════════════════════════════════

class SketchPreprocessor:
    """Seven-stage sketch preprocessing pipeline with high line recall.

    Parameters are exposed in ``__init__`` so every stage can be tuned.
    The ``medium`` argument selects sensible defaults for paper vs. stone.
    """

    # ── Presets per medium ────────────────────────────────────────────────
    _PRESETS: Dict[str, Dict] = {
        "paper": dict(
            nlm_h=7,
            nlm_template_window=7,
            nlm_search_window=21,
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75,
            clahe_clip=3.0,
            clahe_tile=8,
            # Binarization params tuned against da37/da80 ground truth
            # (grid search + sequential 1-D sweeps, mean F1 0.4812 → 0.5980)
            xdog_sigma=0.75,
            xdog_k=1.6,
            xdog_p=100.0,
            xdog_epsilon=0.005,
            xdog_phi=10.0,
            sauvola_window=19,
            sauvola_k=0.35,
            niblack_window=9,
            niblack_k=-0.02,
            min_votes=2,
            min_component_area=150,
            closing_kernel=3,
            do_thinning=False,
        ),
        "stone": dict(
            nlm_h=10,
            nlm_template_window=7,
            nlm_search_window=21,
            bilateral_d=9,
            bilateral_sigma_color=60,
            bilateral_sigma_space=60,
            clahe_clip=4.0,
            clahe_tile=8,
            xdog_sigma=1.2,
            xdog_k=1.6,
            xdog_p=200.0,
            xdog_epsilon=0.01,
            xdog_phi=12.0,
            sauvola_window=31,
            sauvola_k=0.08,
            niblack_window=31,
            niblack_k=-0.15,
            min_votes=2,
            min_component_area=15,
            closing_kernel=3,
            do_thinning=False,
        ),
    }

    def __init__(
        self,
        medium: Literal["paper", "stone"] = "stone",
        *,
        # ── Stage 1: NLM denoising ─────────────────────────────────────
        nlm_h: Optional[int] = None,
        nlm_template_window: Optional[int] = None,
        nlm_search_window: Optional[int] = None,
        # ── Stage 1: bilateral filter ──────────────────────────────────
        bilateral_d: Optional[int] = None,
        bilateral_sigma_color: Optional[float] = None,
        bilateral_sigma_space: Optional[float] = None,
        # ── Stage 3: CLAHE ─────────────────────────────────────────────
        clahe_clip: Optional[float] = None,
        clahe_tile: Optional[int] = None,
        # ── Stage 4a: XDoG ─────────────────────────────────────────────
        xdog_sigma: Optional[float] = None,
        xdog_k: Optional[float] = None,
        xdog_p: Optional[float] = None,
        xdog_epsilon: Optional[float] = None,
        xdog_phi: Optional[float] = None,
        # ── Stage 4b: Sauvola ──────────────────────────────────────────
        sauvola_window: Optional[int] = None,
        sauvola_k: Optional[float] = None,
        # ── Stage 4c: Niblack ──────────────────────────────────────────
        niblack_window: Optional[int] = None,
        niblack_k: Optional[float] = None,
        # ── Stage 4: fusion ────────────────────────────────────────────
        min_votes: Optional[int] = None,
        # ── Stage 5: post-processing ───────────────────────────────────
        min_component_area: Optional[int] = None,
        closing_kernel: Optional[int] = None,
        do_thinning: Optional[bool] = None,
    ):
        preset = self._PRESETS[medium]

        # Merge: explicit kwarg wins over preset default
        def _pick(name: str, value):
            return value if value is not None else preset[name]

        # Stage 1
        self.nlm_h: int                    = _pick("nlm_h", nlm_h)
        self.nlm_template_window: int      = _pick("nlm_template_window", nlm_template_window)
        self.nlm_search_window: int        = _pick("nlm_search_window", nlm_search_window)
        self.bilateral_d: int              = _pick("bilateral_d", bilateral_d)
        self.bilateral_sigma_color: float  = _pick("bilateral_sigma_color", bilateral_sigma_color)
        self.bilateral_sigma_space: float  = _pick("bilateral_sigma_space", bilateral_sigma_space)

        # Stage 3
        self.clahe_clip: float = _pick("clahe_clip", clahe_clip)
        self.clahe_tile: int   = _pick("clahe_tile", clahe_tile)

        # Stage 4a — XDoG
        self.xdog_sigma: float   = _pick("xdog_sigma", xdog_sigma)
        self.xdog_k: float       = _pick("xdog_k", xdog_k)
        self.xdog_p: float       = _pick("xdog_p", xdog_p)
        self.xdog_epsilon: float = _pick("xdog_epsilon", xdog_epsilon)
        self.xdog_phi: float     = _pick("xdog_phi", xdog_phi)

        # Stage 4b/c — adaptive thresholds
        self.sauvola_window: int  = _pick("sauvola_window", sauvola_window)
        self.sauvola_k: float     = _pick("sauvola_k", sauvola_k)
        self.niblack_window: int  = _pick("niblack_window", niblack_window)
        self.niblack_k: float     = _pick("niblack_k", niblack_k)

        # Stage 4 — fusion
        self.min_votes: int = _pick("min_votes", min_votes)

        # Stage 5
        self.min_component_area: int = _pick("min_component_area", min_component_area)
        self.closing_kernel: int     = _pick("closing_kernel", closing_kernel)
        self.do_thinning: bool       = _pick("do_thinning", do_thinning)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 1 — Edge-preserving denoising
    # ───────────────────────────────────────────────────────────────────────

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """Two-pass edge-preserving denoising.

        Pass 1 — Non-Local Means:
            Averages similar patches across the image.  The filter strength
            *h* is kept conservative (default 7 for paper) so that faint
            pencil/ink lines are not blurred away.

        Pass 2 — Bilateral filter:
            Smooths flat regions while preserving sharp edge gradients,
            further reducing low-frequency noise without affecting strokes.

        Args:
            img: Grayscale uint8 image.
        Returns:
            Denoised uint8 image.
        """
        nlm = cv2.fastNlMeansDenoising(
            img,
            h=self.nlm_h,
            templateWindowSize=self.nlm_template_window,
            searchWindowSize=self.nlm_search_window,
        )
        bilateral = cv2.bilateralFilter(
            nlm,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )
        return bilateral

    # ───────────────────────────────────────────────────────────────────────
    # Stage 2 — Background estimation & subtraction
    # ───────────────────────────────────────────────────────────────────────

    def remove_background(self, img: np.ndarray) -> np.ndarray:
        """Estimate and subtract the slowly-varying background.

        A large morphological closing with a circular structuring element
        produces a smooth approximation of the paper/stone surface
        (yellowing, stains, uneven illumination).  Subtracting this from
        the denoised image leaves only the *local* intensity deviations,
        i.e. the sketch strokes.

        The kernel radius is set to ~1/20 of the shorter image dimension,
        clamped to [31, 101] and forced to an odd number.

        Args:
            img: Denoised grayscale uint8 image.
        Returns:
            Background-subtracted uint8 image, normalised to [0, 255].
        """
        h, w = img.shape[:2]
        ksize = int(min(h, w) / 20)
        ksize = max(31, min(ksize, 101))
        ksize = ksize if ksize % 2 == 1 else ksize + 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # Subtract and normalise ──────────────────────────────────────────
        diff = background.astype(np.float64) - img.astype(np.float64)
        diff = np.clip(diff, 0, None)

        d_max = diff.max()
        if d_max > 1e-8:
            diff = diff / d_max * 255.0

        # Invert so that sketch lines are *dark* on a *light* background,
        # matching the convention expected by Sauvola, Niblack, and XDoG.
        return (255 - diff).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 3 — Contrast enhancement (CLAHE)
    # ───────────────────────────────────────────────────────────────────────

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Boost local contrast with CLAHE to reveal faint strokes.

        Operates via scikit-image ``equalize_adapthist`` (CLAHE) so that
        faint pencil marks in brighter areas and strong strokes in darker
        areas are both lifted into a useful dynamic range.

        Args:
            img: Background-subtracted uint8 image.
        Returns:
            Contrast-enhanced uint8 image.
        """
        h, w = img.shape[:2]
        kernel_size = (max(1, h // self.clahe_tile), max(1, w // self.clahe_tile))
        enhanced = equalize_adapthist(
            img, kernel_size=kernel_size, clip_limit=self.clahe_clip / 100.0
        )
        return (enhanced * 255.0).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 4 — Multi-strategy binarization with fusion
    # ───────────────────────────────────────────────────────────────────────

    def _xdog(self, img: np.ndarray) -> np.ndarray:
        """Extended Difference-of-Gaussians (XDoG) line extraction.

        The algorithm applies two Gaussian blurs at scales σ and k·σ,
        computes an amplified difference via a sharpness parameter *p*,
        and passes through a soft thresholding function that enhances
        line-like features while suppressing flat regions.

        Sharpened DoG (Winnemoeller et al. 2012):

            S(x) = (1 + p) · G_σ(x) − p · G_{kσ}(x)

        Soft threshold:
            T(x) = 1                            if S(x) ≥ ε
                  = 1 + tanh(φ · (S(x) − ε))    otherwise

        With dark lines on a light background, line pixels produce
        strongly negative S values (below ε), yielding T ≈ 0.  These
        are classified as foreground.

        Args:
            img: Contrast-enhanced uint8 image (dark lines, light bg).
        Returns:
            Binary uint8 mask from XDoG (foreground = 255, bg = 0).
        """
        img_f = img.astype(np.float64) / 255.0

        g1 = cv2.GaussianBlur(img_f, (0, 0), self.xdog_sigma)
        g2 = cv2.GaussianBlur(img_f, (0, 0), self.xdog_sigma * self.xdog_k)

        # Amplified DoG: sharpness p boosts the difference so that even
        # faint lines produce strong negative excursions.
        dog = (1.0 + self.xdog_p) * g1 - self.xdog_p * g2

        # Soft thresholding ────────────────────────────────────────────────
        result = np.where(
            dog >= self.xdog_epsilon,
            1.0,
            1.0 + np.tanh(self.xdog_phi * (dog - self.xdog_epsilon)),
        )

        # Pixels with result < 0.5 correspond to sketch lines.
        binary = (result < 0.5).astype(np.uint8) * 255
        return binary

    def _sauvola(self, img: np.ndarray) -> np.ndarray:
        """Sauvola adaptive thresholding.

        T(x,y) = μ(x,y) · [1 + k · (σ(x,y)/R − 1)]

        A negative *k* raises the threshold above the local mean, which
        helps capture faint foreground strokes on a light background.

        We threshold with ≤ (pixel darker than threshold → foreground).

        Args:
            img: Contrast-enhanced uint8 image.
        Returns:
            Binary uint8 mask from Sauvola.
        """
        thresh = threshold_sauvola(
            img, window_size=self.sauvola_window, k=self.sauvola_k,
        )
        return (img <= thresh).astype(np.uint8) * 255

    def _niblack(self, img: np.ndarray) -> np.ndarray:
        """Niblack adaptive thresholding.

        T(x,y) = μ(x,y) + k · σ(x,y)

        With a more negative *k* the threshold drops further below the
        local mean, aggressively capturing faint strokes at the expense
        of slightly more noise — a complementary strategy to Sauvola.

        Args:
            img: Contrast-enhanced uint8 image.
        Returns:
            Binary uint8 mask from Niblack.
        """
        thresh = threshold_niblack(
            img, window_size=self.niblack_window, k=self.niblack_k,
        )
        return (img <= thresh).astype(np.uint8) * 255

    def binarize(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fuse three binarization strategies via majority voting.

        A pixel is classified as foreground **only if at least
        ``min_votes``** of the three strategies agree.  The default
        ``min_votes=2`` gives the best trade-off: lines detected by any
        two methods survive, while isolated false-positives from a single
        method are suppressed.

        Args:
            img: Contrast-enhanced uint8 image.
        Returns:
            A tuple ``(fused, individual)`` where *fused* is the binary
            fusion result and *individual* is a dict with keys
            ``'xdog'``, ``'sauvola'``, ``'niblack'``.
        """
        xdog_bin    = self._xdog(img)
        sauvola_bin = self._sauvola(img)
        niblack_bin = self._niblack(img)

        # Majority-vote fusion ────────────────────────────────────────────
        votes = (
            (xdog_bin > 0).astype(np.uint8)
            + (sauvola_bin > 0).astype(np.uint8)
            + (niblack_bin > 0).astype(np.uint8)
        )
        fused = (votes >= self.min_votes).astype(np.uint8) * 255

        individual = {
            "xdog":    xdog_bin,
            "sauvola": sauvola_bin,
            "niblack": niblack_bin,
        }
        return fused, individual

    # ───────────────────────────────────────────────────────────────────────
    # Stage 5 — Morphological post-processing
    # ───────────────────────────────────────────────────────────────────────

    def postprocess(self, binary: np.ndarray) -> np.ndarray:
        """Clean the fused binary with component filtering, closing, and
        optional thinning.

        1. **Small-component removal** — connected components smaller than
           ``min_component_area`` pixels are deleted.  This removes salt
           noise without touching real line fragments.
        2. **Morphological closing** (small kernel, default 2×2) — bridges
           single-pixel gaps in strokes caused by binarization artefacts.
        3. **Optional thinning** — if ``do_thinning=True``, the Zhang–Suen
           thinning algorithm (``skimage.morphology.thin``) reduces all
           strokes to 1 px width.  Useful when the downstream Bezier
           fitter expects thin contours; disabled by default to avoid
           fragmenting already-clean lines.

        Args:
            binary: Binary uint8 image (0 / 255).
        Returns:
            Cleaned binary uint8 image.
        """
        # 1. Small-component removal ──────────────────────────────────────
        bool_mask = binary > 0
        cleaned = remove_small_objects(
            bool_mask, max_size=self.min_component_area, connectivity=2,
        )

        # 2. Morphological closing ────────────────────────────────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.closing_kernel, self.closing_kernel),
        )
        out = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1,
        )

        # 3. Optional thinning ────────────────────────────────────────────
        if self.do_thinning:
            thinned = thin(out > 0)
            out = thinned.astype(np.uint8) * 255

        return out

    # ───────────────────────────────────────────────────────────────────────
    # Full pipeline
    # ───────────────────────────────────────────────────────────────────────

    def process(
        self,
        img: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run the complete seven-stage pipeline.

        Args:
            img: Input image (grayscale uint8 or colour BGR).  If colour,
                 it is converted to grayscale automatically.
            return_intermediates: When True, return a dict of all stage
                outputs keyed by stage name.
        Returns:
            * ``return_intermediates=False``: final binary uint8 image.
            * ``return_intermediates=True``: ordered dict with keys
              ``'grayscale'``, ``'denoised'``, ``'background_removed'``,
              ``'contrast'``, ``'xdog'``, ``'sauvola'``, ``'niblack'``,
              ``'binarized'``, ``'final'``.
        """
        # Stage 0 — to grayscale ──────────────────────────────────────────
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Stage 1 — denoise ───────────────────────────────────────────────
        denoised = self.denoise(gray)

        # Stage 2 — background removal ────────────────────────────────────
        bg_removed = self.remove_background(denoised)

        # Stage 3 — contrast enhance ─────────────────────────────────────
        contrast = self.enhance_contrast(bg_removed)

        # Stage 4 — binarize (fusion) ─────────────────────────────────────
        binarized, individual = self.binarize(contrast)

        # Stage 5 — post-process ──────────────────────────────────────────
        final = self.postprocess(binarized)

        if return_intermediates:
            return {
                "grayscale":          gray,
                "denoised":           denoised,
                "background_removed": bg_removed,
                "contrast":           contrast,
                "xdog":               individual["xdog"],
                "sauvola":            individual["sauvola"],
                "niblack":            individual["niblack"],
                "binarized":          binarized,
                "final":              final,
            }
        return final


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_sketch(
    image_path: str,
    output_dir: Optional[str] = None,
    save_intermediates: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Load an image, run the sketch pipeline, and optionally save stages.

    Args:
        image_path:        Path to a sketch image file (any format OpenCV
                           can read).
        output_dir:        Directory to write output PNGs (created if
                           needed).
        save_intermediates: When True, save all stage images to
                           *output_dir*.
        **kwargs:          Forwarded to :class:`SketchPreprocessor.__init__`.
    Returns:
        Dict with keys: ``'original'``, ``'grayscale'``, ``'denoised'``,
        ``'background_removed'``, ``'contrast'``, ``'xdog'``,
        ``'sauvola'``, ``'niblack'``, ``'binarized'``, ``'final'``.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    preprocessor = SketchPreprocessor(**kwargs)
    stages: Dict[str, np.ndarray] = preprocessor.process(
        img, return_intermediates=True,
    )  # type: ignore[assignment]
    stages["original"] = img

    if save_intermediates and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filenames = {
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
        for key, fname in filenames.items():
            cv2.imwrite(os.path.join(output_dir, fname), stages[key])

    return stages
