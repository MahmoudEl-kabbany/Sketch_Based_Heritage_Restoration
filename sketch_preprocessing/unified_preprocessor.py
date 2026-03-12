"""
Unified Heritage Restoration Preprocessor
==========================================
Eight-stage pipeline combining the best of ``SketchPreprocessor`` and
``StoneInscriptionRestorer`` for virtual restoration of ancient Egyptian
paintings, carvings, and paper sketches.

  0. **Input**          — Load, keep BGR copy for colour segmentation,
                          convert to grayscale.
  1. **Denoise**        — Non-Local Means + Bilateral Filter
                          (edge-preserving; from *SketchPreprocessor*).
  2. **Background**     — Morphological closing estimation + subtraction
                          (from *SketchPreprocessor*).
  3. **Enhancement**    — Selectable: CLAHE / blackhat / linear /
                          histogram EQ (from *StoneInscriptionRestorer*).
  4. **Segmentation**   — Optional: K-means colour clustering, GrabCut,
                          or Watershed.  Produces a foreground mask that
                          can participate in the binarization vote.
  5. **Binarization**   — XDoG + Sauvola + Niblack majority-vote fusion
                          (from *SketchPreprocessor*).  When a
                          segmentation mask is available it joins as a
                          4th voter.
  6. **Edge detection** — Canny on the enhanced image
                          (from *StoneInscriptionRestorer*).
  7. **Fusion + Post-processing** — OR(binarization, edges) →
                          small-component removal → morphological
                          closing → optional thinning.

The final output contains **both** filled binary regions and thin
contour edges, maximising line coverage at very low noise.

Design rationale
----------------
* *SketchPreprocessor* excels at preserving faint strokes via
  edge-preserving denoising, background subtraction, and multi-strategy
  binarization fusion — but lacks selectable enhancement and Canny edges.
* *StoneInscriptionRestorer* offers powerful enhancement methods
  (especially black-hat for carved inscriptions) and Canny edge
  detection — but uses only a single binarization strategy and no
  background subtraction.
* Combining both yields a pipeline that handles everything from faded
  pencil sketches to deeply carved stone reliefs.  The new segmentation
  stage (K-means, GrabCut, Watershed) leverages colour information that
  grayscale-only methods miss.
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_niblack, threshold_sauvola
from skimage.morphology import remove_small_objects, thin

_ENHANCE_METHODS = ("clahe", "linear", "histogram_eq", "blackhat")
_SEGMENT_METHODS = ("kmeans", "grabcut", "watershed")


# ═══════════════════════════════════════════════════════════════════════════
# UnifiedPreprocessor
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedPreprocessor:
    """Eight-stage restoration pipeline for heritage sketches and stones.

    Parameters are exposed in ``__init__`` so every stage can be tuned.
    The ``medium`` argument selects sensible defaults for paper vs. stone.
    """

    # ── Presets per medium ────────────────────────────────────────────────
    _PRESETS: Dict[str, Dict] = {
        "paper": dict(
            # Stage 1 — denoise
            nlm_h=7,
            nlm_template_window=7,
            nlm_search_window=21,
            bilateral_d=9,
            bilateral_sigma_color=75,
            bilateral_sigma_space=75,
            # Stage 3 — enhancement
            enhance_method="clahe",
            gain=1.5,
            bias=20.0,
            clahe_clip=3.0,
            clahe_tile=8,
            blackhat_ksize=51,
            # Stage 4 — segmentation
            segment_method=None,
            kmeans_k=3,
            grabcut_iters=5,
            watershed_min_distance=20,
            segment_as_voter=True,
            # Stage 5 — binarization
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
            # Stage 6 — Canny
            canny_low=30,
            canny_high=100,
            gaussian_ksize=5,
            blur_sigma=1.4,
            # Stage 7 — post-processing
            min_component_area=150,
            closing_kernel=3,
            do_thinning=False,
        ),
        "stone": dict(
            # Stage 1 — denoise
            nlm_h=10,
            nlm_template_window=7,
            nlm_search_window=21,
            bilateral_d=9,
            bilateral_sigma_color=60,
            bilateral_sigma_space=60,
            # Stage 3 — enhancement
            enhance_method="blackhat",
            gain=1.5,
            bias=20.0,
            clahe_clip=4.0,
            clahe_tile=8,
            blackhat_ksize=51,
            # Stage 4 — segmentation
            segment_method="kmeans",
            kmeans_k=3,
            grabcut_iters=5,
            watershed_min_distance=15,
            segment_as_voter=True,
            # Stage 5 — binarization
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
            # Stage 6 — Canny
            canny_low=60,
            canny_high=100,
            gaussian_ksize=5,
            blur_sigma=1.4,
            # Stage 7 — post-processing
            min_component_area=15,
            closing_kernel=3,
            do_thinning=False,
        ),
    }

    def __init__(
        self,
        medium: Literal["paper", "stone"] = "paper",
        *,
        # ── Stage 1: NLM denoising ─────────────────────────────────────
        nlm_h: Optional[int] = None,
        nlm_template_window: Optional[int] = None,
        nlm_search_window: Optional[int] = None,
        # ── Stage 1: bilateral filter ──────────────────────────────────
        bilateral_d: Optional[int] = None,
        bilateral_sigma_color: Optional[float] = None,
        bilateral_sigma_space: Optional[float] = None,
        # ── Stage 3: enhancement ───────────────────────────────────────
        enhance_method: Optional[str] = None,
        gain: Optional[float] = None,
        bias: Optional[float] = None,
        clahe_clip: Optional[float] = None,
        clahe_tile: Optional[int] = None,
        blackhat_ksize: Optional[int] = None,
        # ── Stage 4: segmentation ──────────────────────────────────────
        segment_method: Optional[str] = None,
        kmeans_k: Optional[int] = None,
        grabcut_iters: Optional[int] = None,
        watershed_min_distance: Optional[int] = None,
        segment_as_voter: Optional[bool] = None,
        # ── Stage 5a: XDoG ─────────────────────────────────────────────
        xdog_sigma: Optional[float] = None,
        xdog_k: Optional[float] = None,
        xdog_p: Optional[float] = None,
        xdog_epsilon: Optional[float] = None,
        xdog_phi: Optional[float] = None,
        # ── Stage 5b: Sauvola ──────────────────────────────────────────
        sauvola_window: Optional[int] = None,
        sauvola_k: Optional[float] = None,
        # ── Stage 5c: Niblack ──────────────────────────────────────────
        niblack_window: Optional[int] = None,
        niblack_k: Optional[float] = None,
        # ── Stage 5: fusion ────────────────────────────────────────────
        min_votes: Optional[int] = None,
        # ── Stage 6: Canny ─────────────────────────────────────────────
        canny_low: Optional[int] = None,
        canny_high: Optional[int] = None,
        gaussian_ksize: Optional[int] = None,
        blur_sigma: Optional[float] = None,
        # ── Stage 7: post-processing ───────────────────────────────────
        min_component_area: Optional[int] = None,
        closing_kernel: Optional[int] = None,
        do_thinning: Optional[bool] = None,
    ):
        preset = self._PRESETS[medium]

        def _pick(name: str, value):
            return value if value is not None else preset[name]

        # Stage 1 — denoise
        self.nlm_h: int                    = _pick("nlm_h", nlm_h)
        self.nlm_template_window: int      = _pick("nlm_template_window", nlm_template_window)
        self.nlm_search_window: int        = _pick("nlm_search_window", nlm_search_window)
        self.bilateral_d: int              = _pick("bilateral_d", bilateral_d)
        self.bilateral_sigma_color: float  = _pick("bilateral_sigma_color", bilateral_sigma_color)
        self.bilateral_sigma_space: float  = _pick("bilateral_sigma_space", bilateral_sigma_space)

        # Stage 3 — enhancement
        method = _pick("enhance_method", enhance_method)
        if method not in _ENHANCE_METHODS:
            raise ValueError(
                f"enhance_method must be one of {_ENHANCE_METHODS}, got {method!r}"
            )
        self.enhance_method: str = method
        self.gain: float            = float(_pick("gain", gain))
        self.bias: float            = float(_pick("bias", bias))
        self.clahe_clip: float      = float(_pick("clahe_clip", clahe_clip))
        self.clahe_tile: int        = int(_pick("clahe_tile", clahe_tile))
        self.blackhat_ksize: int    = int(_pick("blackhat_ksize", blackhat_ksize))

        # Stage 4 — segmentation
        seg = _pick("segment_method", segment_method)
        if seg is not None and seg not in _SEGMENT_METHODS:
            raise ValueError(
                f"segment_method must be None or one of {_SEGMENT_METHODS}, "
                f"got {seg!r}"
            )
        self.segment_method: Optional[str] = seg
        self.kmeans_k: int                 = int(_pick("kmeans_k", kmeans_k))
        self.grabcut_iters: int            = int(_pick("grabcut_iters", grabcut_iters))
        self.watershed_min_distance: int   = int(_pick("watershed_min_distance", watershed_min_distance))
        self.segment_as_voter: bool        = _pick("segment_as_voter", segment_as_voter)

        # Stage 5 — binarization (XDoG + Sauvola + Niblack)
        self.xdog_sigma: float   = float(_pick("xdog_sigma", xdog_sigma))
        self.xdog_k: float       = float(_pick("xdog_k", xdog_k))
        self.xdog_p: float       = float(_pick("xdog_p", xdog_p))
        self.xdog_epsilon: float = float(_pick("xdog_epsilon", xdog_epsilon))
        self.xdog_phi: float     = float(_pick("xdog_phi", xdog_phi))
        self.sauvola_window: int = int(_pick("sauvola_window", sauvola_window))
        self.sauvola_k: float    = float(_pick("sauvola_k", sauvola_k))
        self.niblack_window: int = int(_pick("niblack_window", niblack_window))
        self.niblack_k: float    = float(_pick("niblack_k", niblack_k))
        self.min_votes: int      = int(_pick("min_votes", min_votes))

        # Stage 6 — Canny
        self.canny_low: int      = int(_pick("canny_low", canny_low))
        self.canny_high: int     = int(_pick("canny_high", canny_high))
        self.gaussian_ksize: int = int(_pick("gaussian_ksize", gaussian_ksize))
        self.blur_sigma: float   = float(_pick("blur_sigma", blur_sigma))

        # Stage 7 — post-processing
        self.min_component_area: int = int(_pick("min_component_area", min_component_area))
        self.closing_kernel: int     = int(_pick("closing_kernel", closing_kernel))
        self.do_thinning: bool       = _pick("do_thinning", do_thinning)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 1 — Edge-preserving denoising
    # ───────────────────────────────────────────────────────────────────────

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """NLM + bilateral filter — protects faint strokes while reducing
        noise in flat regions."""
        nlm = cv2.fastNlMeansDenoising(
            gray,
            h=self.nlm_h,
            templateWindowSize=self.nlm_template_window,
            searchWindowSize=self.nlm_search_window,
        )
        return cv2.bilateralFilter(
            nlm,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Stage 2 — Background estimation & subtraction
    # ───────────────────────────────────────────────────────────────────────

    def remove_background(self, gray: np.ndarray) -> np.ndarray:
        """Morphological closing estimates the slowly-varying surface
        (stains, yellowing, uneven illumination).  Subtraction isolates
        local deviations, i.e. the sketch strokes."""
        h, w = gray.shape[:2]
        ksize = int(min(h, w) / 20)
        ksize = max(31, min(ksize, 101))
        ksize = ksize if ksize % 2 == 1 else ksize + 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        diff = background.astype(np.float64) - gray.astype(np.float64)
        diff = np.clip(diff, 0, None)
        d_max = diff.max()
        if d_max > 1e-8:
            diff = diff / d_max * 255.0

        # Invert: dark lines on light background (convention for binarizers)
        return (255 - diff).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 3 — Enhancement (selectable)
    # ───────────────────────────────────────────────────────────────────────

    def enhance(self, gray: np.ndarray) -> np.ndarray:
        """Dispatch to the configured enhancement strategy."""
        dispatch = {
            "clahe":        self._enhance_clahe,
            "linear":       self._enhance_linear,
            "histogram_eq": self._enhance_hist_eq,
            "blackhat":     self._enhance_blackhat,
        }
        return dispatch[self.enhance_method](gray)

    def _enhance_clahe(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE via scikit-image — adapts contrast locally per tile."""
        h, w = gray.shape[:2]
        kernel_size = (max(1, h // self.clahe_tile), max(1, w // self.clahe_tile))
        enhanced = equalize_adapthist(
            gray, kernel_size=kernel_size, clip_limit=self.clahe_clip / 100.0,
        )
        return (enhanced * 255.0).astype(np.uint8)

    def _enhance_linear(self, gray: np.ndarray) -> np.ndarray:
        """g(x,y) = gain · f(x,y) + bias."""
        return cv2.convertScaleAbs(gray, alpha=self.gain, beta=self.bias)

    def _enhance_hist_eq(self, gray: np.ndarray) -> np.ndarray:
        """Global histogram equalisation — parameter-free."""
        return cv2.equalizeHist(gray)

    def _enhance_blackhat(self, gray: np.ndarray) -> np.ndarray:
        """Morphological black-hat: closing(I) − I.
        Isolates dark carved strokes on bright stone."""
        ksize = self.blackhat_ksize
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        bh_max = bh.max()
        if bh_max > 0:
            bh = (bh.astype(np.float64) / bh_max * 255).astype(np.uint8)
        return bh

    # ───────────────────────────────────────────────────────────────────────
    # Stage 4 — Segmentation (optional)
    # ───────────────────────────────────────────────────────────────────────

    def segment(
        self,
        enhanced: np.ndarray,
        color_img: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Produce a binary foreground mask using the configured method.

        Returns ``None`` when ``segment_method`` is ``None``.
        """
        if self.segment_method is None:
            return None
        dispatch = {
            "kmeans":    self._segment_kmeans,
            "grabcut":   self._segment_grabcut,
            "watershed": self._segment_watershed,
        }
        return dispatch[self.segment_method](enhanced, color_img)

    def _segment_kmeans(
        self, enhanced: np.ndarray, color_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """K-means colour clustering — the darkest cluster is foreground."""
        if color_img is not None and color_img.ndim == 3:
            data = color_img.reshape(-1, 3).astype(np.float32)
        else:
            data = enhanced.reshape(-1, 1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, self.kmeans_k, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()

        brightness = centers.mean(axis=1) if centers.shape[1] == 3 else centers.flatten()
        fg_label = int(np.argmin(brightness))
        mask = np.where(labels == fg_label, 255, 0).astype(np.uint8)
        return mask.reshape(enhanced.shape[:2])

    def _segment_grabcut(
        self, enhanced: np.ndarray, color_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """GrabCut initialised from Otsu thresholding on the enhanced image."""
        # GrabCut requires a 3-channel image
        if color_img is not None and color_img.ndim == 3:
            img3 = color_img
        else:
            img3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Seed mask from Otsu on the enhanced grayscale
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        gc_mask = np.full(enhanced.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[otsu == 255] = cv2.GC_PR_FGD

        # Definite background: pixels far from any foreground
        dilated = cv2.dilate(otsu, np.ones((15, 15), np.uint8), iterations=2)
        gc_mask[dilated == 0] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img3, gc_mask, None, bgd_model, fgd_model,
            self.grabcut_iters, cv2.GC_INIT_WITH_MASK,
        )
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0,
        ).astype(np.uint8)
        return result

    def _segment_watershed(
        self, enhanced: np.ndarray, color_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Watershed segmentation with distance-transform markers."""
        from scipy import ndimage

        # Otsu to get initial binary
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background via dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Sure foreground via distance transform
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist, self.watershed_min_distance / 100.0 * dist.max(), 255, 0,
        )
        sure_fg = sure_fg.astype(np.uint8)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # background becomes 1, not 0
        markers[unknown == 255] = 0  # unknown region marked 0

        # Watershed needs 3-channel image
        if color_img is not None and color_img.ndim == 3:
            img3 = color_img
        else:
            img3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(img3, markers)

        # Foreground = all markers > 1 (marker 1 is background)
        # Boundary pixels (markers == -1) are included as edges
        mask = np.where((markers > 1) | (markers == -1), 255, 0).astype(np.uint8)
        return mask

    # ───────────────────────────────────────────────────────────────────────
    # Stage 5 — Multi-strategy binarization with fusion
    # ───────────────────────────────────────────────────────────────────────

    def _xdog(self, img: np.ndarray) -> np.ndarray:
        """Extended Difference-of-Gaussians line extraction."""
        img_f = img.astype(np.float64) / 255.0
        g1 = cv2.GaussianBlur(img_f, (0, 0), self.xdog_sigma)
        g2 = cv2.GaussianBlur(img_f, (0, 0), self.xdog_sigma * self.xdog_k)
        dog = (1.0 + self.xdog_p) * g1 - self.xdog_p * g2
        result = np.where(
            dog >= self.xdog_epsilon,
            1.0,
            1.0 + np.tanh(self.xdog_phi * (dog - self.xdog_epsilon)),
        )
        return (result < 0.5).astype(np.uint8) * 255

    def _sauvola(self, img: np.ndarray) -> np.ndarray:
        """Sauvola adaptive thresholding."""
        thresh = threshold_sauvola(
            img, window_size=self.sauvola_window, k=self.sauvola_k,
        )
        return (img <= thresh).astype(np.uint8) * 255

    def _niblack(self, img: np.ndarray) -> np.ndarray:
        """Niblack adaptive thresholding."""
        thresh = threshold_niblack(
            img, window_size=self.niblack_window, k=self.niblack_k,
        )
        return (img <= thresh).astype(np.uint8) * 255

    def binarize(
        self,
        img: np.ndarray,
        seg_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Fuse binarization strategies via majority voting.

        When *seg_mask* is provided and ``segment_as_voter`` is True it
        participates as a 4th voter and ``min_votes`` is auto-adjusted
        to 3 (unless the user explicitly set ``min_votes``).
        """
        xdog_bin    = self._xdog(img)
        sauvola_bin = self._sauvola(img)
        niblack_bin = self._niblack(img)

        votes = (
            (xdog_bin > 0).astype(np.uint8)
            + (sauvola_bin > 0).astype(np.uint8)
            + (niblack_bin > 0).astype(np.uint8)
        )

        effective_min_votes = self.min_votes

        individual: Dict[str, np.ndarray] = {
            "xdog":    xdog_bin,
            "sauvola": sauvola_bin,
            "niblack": niblack_bin,
        }

        if seg_mask is not None and self.segment_as_voter:
            votes = votes + (seg_mask > 0).astype(np.uint8)
            individual["segmentation"] = seg_mask
            # Auto-adjust: with 4 voters, bump min_votes from 2 to 3
            if self.min_votes == 2:
                effective_min_votes = 3

        fused = (votes >= effective_min_votes).astype(np.uint8) * 255
        return fused, individual

    # ───────────────────────────────────────────────────────────────────────
    # Stage 6 — Edge detection (Canny)
    # ───────────────────────────────────────────────────────────────────────

    def detect_edges(self, enhanced: np.ndarray) -> np.ndarray:
        """Gaussian blur → Canny edge detection on the enhanced image."""
        ksize = self.gaussian_ksize
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), self.blur_sigma)
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 7 — Fusion + Post-processing
    # ───────────────────────────────────────────────────────────────────────

    def postprocess(self, binarized: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Combine binarization and edge maps, then clean.

        1. Bitwise-OR of binarization result and Canny edges.
        2. Small-component removal.
        3. Morphological closing to bridge single-pixel gaps.
        4. Optional Zhang–Suen thinning.
        """
        combined = cv2.bitwise_or(binarized, edges)

        # Small-component removal
        bool_mask = combined > 0
        cleaned = remove_small_objects(
            bool_mask, max_size=self.min_component_area, connectivity=2,
        )

        # Morphological closing
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.closing_kernel, self.closing_kernel),
        )
        out = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=1,
        )

        # Optional thinning
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
        """Run the complete eight-stage pipeline.

        Args:
            img: Input image (BGR colour or grayscale uint8).
            return_intermediates: When True, return a dict of all stage
                outputs keyed by stage name.
        Returns:
            * ``return_intermediates=False``: final binary uint8 image.
            * ``return_intermediates=True``: dict with keys
              ``'grayscale'``, ``'denoised'``, ``'background_removed'``,
              ``'enhanced'``, [``'segmentation'``], ``'xdog'``,
              ``'sauvola'``, ``'niblack'``, ``'binarized'``, ``'edges'``,
              ``'final'``.
        """
        # Stage 0 — input handling
        # Keep a 3-channel BGR copy for colour-based segmentation
        if img.ndim == 3:
            if img.shape[2] == 4:
                # RGBA → BGR (drop alpha)
                color_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                color_img = img
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        else:
            color_img = None
            gray = img.copy()

        # Stage 1 — denoise
        denoised = self.denoise(gray)

        # Stage 2 — background removal
        bg_removed = self.remove_background(denoised)

        # Stage 3 — enhancement
        enhanced = self.enhance(bg_removed)

        # Stage 4 — segmentation (optional)
        seg_mask = self.segment(enhanced, color_img)

        # Stage 5 — binarization (fusion)
        binarized, individual = self.binarize(enhanced, seg_mask)

        # Stage 6 — edge detection
        edges = self.detect_edges(enhanced)

        # Stage 7 — fusion + post-processing
        final = self.postprocess(binarized, edges)

        if return_intermediates:
            stages: Dict[str, np.ndarray] = {
                "grayscale":          gray,
                "denoised":           denoised,
                "background_removed": bg_removed,
                "enhanced":           enhanced,
            }
            if seg_mask is not None:
                stages["segmentation"] = seg_mask
            stages.update({
                "xdog":      individual["xdog"],
                "sauvola":   individual["sauvola"],
                "niblack":   individual["niblack"],
                "binarized": binarized,
                "edges":     edges,
                "final":     final,
            })
            return stages
        return final


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def unified_preprocess(
    image_path: str,
    output_dir: Optional[str] = None,
    save_intermediates: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Load an image, run the unified pipeline, and optionally save stages.

    Args:
        image_path:        Path to an image file (any format OpenCV reads).
        output_dir:        Directory to write output PNGs (created if needed).
        save_intermediates: When True, save all stage images to *output_dir*.
        **kwargs:          Forwarded to :class:`UnifiedPreprocessor.__init__`.
    Returns:
        Dict of all pipeline stages (always includes intermediates).
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    preprocessor = UnifiedPreprocessor(**kwargs)
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
            "enhanced":           "4_enhanced.png",
            "segmentation":       "5_segmentation.png",
            "xdog":               "6a_xdog.png",
            "sauvola":            "6b_sauvola.png",
            "niblack":            "6c_niblack.png",
            "binarized":          "7_binarized_fused.png",
            "edges":              "8_edges_canny.png",
            "final":              "9_final.png",
        }
        for key, fname in filenames.items():
            if key in stages:
                cv2.imwrite(os.path.join(output_dir, fname), stages[key])

    return stages
