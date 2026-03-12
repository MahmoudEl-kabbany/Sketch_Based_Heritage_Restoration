"""
Stone Inscription Restoration Module
=====================================
Two-stage pipeline for virtual restoration of historical stone inscriptions,
optimised for ancient Egyptian paintings and carvings.

  Stage 1 — **Image Enhancement** (selectable method)

      ``"linear"``
          Point-based linear transformation:
          :math:`g(x,y) = a \\cdot f(x,y) + b`.
          Good for dark / low-contrast stone.

      ``"clahe"``
          Contrast Limited Adaptive Histogram Equalisation.
          Operates on local tiles — handles uneven illumination and
          bright images without saturating highlights.

      ``"histogram_eq"``
          Global histogram equalisation.
          Redistributes pixel intensities across the full [0, 255] range.
          Fast single-parameter-free option for washed-out images.

      ``"blackhat"``
          Morphological black-hat transform:
          :math:`\\text{blackhat} = \\text{closing}(I) - I`.
          Directly isolates dark carved strokes on a bright stone surface.
          Excellent for relief inscriptions.

  Stage 1.5 — **Binarization / Segmentation** (optional, selectable)

      ``None``  (default)
          Skip binarization — Canny runs directly on the enhanced image.

      ``"otsu"``
          Otsu's automatic global thresholding.  Minimises intra-class
          variance to find the optimal split point.  Parameter-free.

      ``"adaptive_gaussian"``
          Local adaptive thresholding with a Gaussian-weighted
          neighbourhood.  Handles lighting gradients across the stone.

      ``"sauvola"``
          Sauvola adaptive thresholding — designed for degraded documents.
          Uses local mean and standard deviation:
          :math:`T(x,y) = \mu(x,y)\bigl[1 + k\bigl(\sigma(x,y)/R - 1\bigr)\bigr]`.

      ``"kmeans"``
          K-means colour clustering on the **original colour** image.
          Can separate painted hieroglyphs from stone even when grayscale
          contrast is poor (e.g. faded red ochre on sandstone).

  Stage 2 — **Edge Detection** (Canny Operator)
      Extract smooth, complete text outlines using the Canny algorithm:

      1. Gaussian low-pass filtering to reduce noise.
      2. Gradient calculation (Sobel kernels internally):

         .. math::
             d_x = \\begin{bmatrix}-1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1\\end{bmatrix},\\quad
             d_y = \\begin{bmatrix}-1 & -2 & -1 \\\\  0 &  0 &  0 \\\\  1 &  2 &  1\\end{bmatrix}

      3. Non-maximum suppression for thin edges.
      4. Hysteresis thresholding with *minVal* / *maxVal*.

Design rationale
----------------
Ancient Egyptian stone inscriptions present unique challenges:

* **Low native contrast** — carved or painted regions on limestone differ
  only subtly from the surrounding surface after millennia of weathering.
* **Uneven illumination** — museum / field photographs often have lighting
  gradients across the stone surface.
* **Variable carving depth** — hieroglyphic strokes range from deep incisions
  to very shallow relief, requiring sensitive edge detection.

Multiple enhancement strategies are provided because no single method
works optimally for all lighting conditions:

* *Linear* amplifies contrast but can saturate bright images.
* *CLAHE* adapts locally — the best general-purpose choice.
* *Histogram EQ* is parameter-free and effective for globally washed-out images.
* *Black-hat* is purpose-built for dark carvings on bright stone.

An optional binarization / segmentation step (*Stage 1.5*) can produce
filled-region masks that complement the Canny edge output:

* *Otsu* — automatic, parameter-free global threshold.
* *Adaptive Gaussian* — local thresholding; best for uneven lighting.
* *Sauvola* — purpose-built for degraded text on noisy backgrounds.
* *K-means* — colour-based clustering; captures painted inscriptions
  invisible in grayscale.

The Canny operator (*Stage 2*) then produces clean, connected contour maps
that faithfully represent the original writing style.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import cv2
import numpy as np

_ENHANCE_METHODS = ("linear", "clahe", "histogram_eq", "blackhat")
_BINARIZE_METHODS = ("otsu", "adaptive_gaussian", "sauvola", "kmeans")


class StoneInscriptionRestorer:
    """Two-stage restoration pipeline for stone inscriptions.

    Parameters
    ----------
    enhance_method : str
        Enhancement strategy for Stage 1.  One of:
        ``"linear"`` | ``"clahe"`` | ``"histogram_eq"`` | ``"blackhat"``.
        Default ``"clahe"``.
    gain : float
        *(linear only)* Contrast multiplier (*a*).  Default ``1.5``.
    bias : float
        *(linear only)* Brightness offset (*b*).  Default ``20``.
    clahe_clip : float
        *(clahe only)* Clip limit for CLAHE.  Default ``2.0``.
    clahe_tile : int
        *(clahe only)* Tile grid size (NxN).  Default ``8``.
    blackhat_ksize : int
        *(blackhat only)* Structuring-element diameter.  Default ``51``.
    binarize_method : str or None
        Binarization strategy for Stage 1.5.  ``None`` to skip (default).
        One of: ``"otsu"`` | ``"adaptive_gaussian"`` | ``"sauvola"`` |
        ``"kmeans"``.
    adaptive_block : int
        *(adaptive_gaussian only)* Neighbourhood size (must be odd).
        Default ``35``.
    adaptive_c : float
        *(adaptive_gaussian only)* Constant subtracted from the mean.
        Default ``10``.
    sauvola_window : int
        *(sauvola only)* Window size (must be odd).  Default ``35``.
    sauvola_k : float
        *(sauvola only)* Sensitivity parameter.  Default ``0.2``.
    kmeans_k : int
        *(kmeans only)* Number of colour clusters.  Default ``3``.
    canny_low : int
        Lower hysteresis threshold for Canny.  Default ``30``.
    canny_high : int
        Upper hysteresis threshold for Canny.  Default ``100``.
    gaussian_ksize : int
        Kernel size for the Gaussian blur before Canny (must be odd).
        Default ``5``.
    blur_sigma : float
        Standard deviation of the Gaussian kernel.  Default ``1.4``.
    """

    # Default presets tuned for ancient Egyptian stone inscriptions
    _DEFAULTS: Dict[str, Union[float, int, str]] = {
        "enhance_method": "clahe",
        # linear params
        "gain": 1.5,
        "bias": 20.0,
        # clahe params
        "clahe_clip": 2.0,
        "clahe_tile": 8,
        # blackhat params
        "blackhat_ksize": 51,
        # binarization params
        "binarize_method": None,
        "adaptive_block": 35,
        "adaptive_c": 10.0,
        "sauvola_window": 35,
        "sauvola_k": 0.2,
        "kmeans_k": 3,
        # canny params
        "canny_low": 30,
        "canny_high": 100,
        "gaussian_ksize": 5,
        "blur_sigma": 1.4,
    }

    def __init__(
        self,
        enhance_method: Optional[str] = None,
        gain: Optional[float] = None,
        bias: Optional[float] = None,
        clahe_clip: Optional[float] = None,
        clahe_tile: Optional[int] = None,
        blackhat_ksize: Optional[int] = None,
        binarize_method: Optional[str] = None,
        adaptive_block: Optional[int] = None,
        adaptive_c: Optional[float] = None,
        sauvola_window: Optional[int] = None,
        sauvola_k: Optional[float] = None,
        kmeans_k: Optional[int] = None,
        canny_low: Optional[int] = None,
        canny_high: Optional[int] = None,
        gaussian_ksize: Optional[int] = None,
        blur_sigma: Optional[float] = None,
    ) -> None:
        d = self._DEFAULTS

        method = enhance_method if enhance_method is not None else d["enhance_method"]
        if method not in _ENHANCE_METHODS:
            raise ValueError(
                f"enhance_method must be one of {_ENHANCE_METHODS}, got {method!r}"
            )
        self.enhance_method: str = method

        # linear
        self.gain: float = gain if gain is not None else d["gain"]
        self.bias: float = bias if bias is not None else d["bias"]
        # clahe
        self.clahe_clip: float = float(clahe_clip if clahe_clip is not None else d["clahe_clip"])
        self.clahe_tile: int = int(clahe_tile if clahe_tile is not None else d["clahe_tile"])
        # blackhat
        self.blackhat_ksize: int = int(
            blackhat_ksize if blackhat_ksize is not None else d["blackhat_ksize"]
        )

        # binarization
        bin_method = binarize_method if binarize_method is not None else d["binarize_method"]
        if bin_method is not None and bin_method not in _BINARIZE_METHODS:
            raise ValueError(
                f"binarize_method must be None or one of {_BINARIZE_METHODS}, "
                f"got {bin_method!r}"
            )
        self.binarize_method: Optional[str] = bin_method
        self.adaptive_block: int = int(
            adaptive_block if adaptive_block is not None else d["adaptive_block"]
        )
        self.adaptive_c: float = float(
            adaptive_c if adaptive_c is not None else d["adaptive_c"]
        )
        self.sauvola_window: int = int(
            sauvola_window if sauvola_window is not None else d["sauvola_window"]
        )
        self.sauvola_k: float = float(
            sauvola_k if sauvola_k is not None else d["sauvola_k"]
        )
        self.kmeans_k: int = int(kmeans_k if kmeans_k is not None else d["kmeans_k"])

        # canny
        self.canny_low: int = int(canny_low if canny_low is not None else d["canny_low"])
        self.canny_high: int = int(canny_high if canny_high is not None else d["canny_high"])
        self.gaussian_ksize: int = int(
            gaussian_ksize if gaussian_ksize is not None else d["gaussian_ksize"]
        )
        self.blur_sigma: float = float(
            blur_sigma if blur_sigma is not None else d["blur_sigma"]
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale if necessary."""
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # ── Stage 1: Enhancement (selectable) ─────────────────────────────────

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Enhance the grayscale image using the configured method.

        Dispatches to one of:
        * ``_enhance_linear``   — g = a·f + b
        * ``_enhance_clahe``    — CLAHE (local adaptive histogram eq.)
        * ``_enhance_hist_eq``  — global histogram equalisation
        * ``_enhance_blackhat`` — morphological black-hat transform

        Parameters
        ----------
        gray : np.ndarray
            Single-channel grayscale image (uint8).

        Returns
        -------
        np.ndarray
            Enhanced grayscale image (uint8).
        """
        dispatch = {
            "linear": self._enhance_linear,
            "clahe": self._enhance_clahe,
            "histogram_eq": self._enhance_hist_eq,
            "blackhat": self._enhance_blackhat,
        }
        return dispatch[self.enhance_method](gray)

    # -- individual enhancement strategies ----------------------------------

    def _enhance_linear(self, gray: np.ndarray) -> np.ndarray:
        """g(x,y) = a · f(x,y) + b  via ``cv2.convertScaleAbs``."""
        return cv2.convertScaleAbs(gray, alpha=self.gain, beta=self.bias)

    def _enhance_clahe(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE — adapts contrast locally per tile.

        Handles uneven illumination and avoids saturating bright regions.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_tile, self.clahe_tile),
        )
        return clahe.apply(gray)

    def _enhance_hist_eq(self, gray: np.ndarray) -> np.ndarray:
        """Global histogram equalisation — parameter-free."""
        return cv2.equalizeHist(gray)

    def _enhance_blackhat(self, gray: np.ndarray) -> np.ndarray:
        """Black-hat: closing(I) - I.  Isolates dark strokes on bright stone."""
        ksize = self.blackhat_ksize
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        # Stretch to full [0, 255] range for best Canny input
        bh_max = bh.max()
        if bh_max > 0:
            bh = (bh.astype(np.float64) / bh_max * 255).astype(np.uint8)
        return bh

    # ── Stage 1.5: Binarization / Segmentation (optional) ─────────────────

    def binarize(self, enhanced: np.ndarray, color_img: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Binarize the enhanced image using the configured method.

        Parameters
        ----------
        enhanced : np.ndarray
            Enhanced grayscale image (uint8).
        color_img : np.ndarray or None
            Original colour image — required only for ``"kmeans"``.

        Returns
        -------
        np.ndarray or None
            Binary mask (0 = background, 255 = foreground), or ``None``
            if ``binarize_method`` is ``None``.
        """
        if self.binarize_method is None:
            return None

        dispatch = {
            "otsu": self._binarize_otsu,
            "adaptive_gaussian": self._binarize_adaptive,
            "sauvola": self._binarize_sauvola,
            "kmeans": self._binarize_kmeans,
        }
        fn = dispatch[self.binarize_method]
        if self.binarize_method == "kmeans":
            return fn(enhanced, color_img)
        return fn(enhanced)

    # -- individual binarization strategies ---------------------------------

    @staticmethod
    def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
        """Otsu's method — automatic global threshold."""
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _binarize_adaptive(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian — local threshold per neighbourhood."""
        block = self.adaptive_block
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, self.adaptive_c,
        )

    def _binarize_sauvola(self, gray: np.ndarray) -> np.ndarray:
        """Sauvola — designed for degraded documents.

        T(x,y) = mu(x,y) * [1 + k * (sigma(x,y)/R - 1)]
        """
        win = self.sauvola_window
        if win % 2 == 0:
            win += 1

        # Compute local mean and std via box filter
        gray_f = gray.astype(np.float64)
        mean = cv2.boxFilter(gray_f, -1, (win, win))
        sq_mean = cv2.boxFilter(gray_f ** 2, -1, (win, win))
        std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

        R = 128.0  # dynamic range of std for uint8
        thresh = mean * (1.0 + self.sauvola_k * (std / R - 1.0))
        binary = np.where(gray_f < thresh, 255, 0).astype(np.uint8)
        return binary

    def _binarize_kmeans(self, gray: np.ndarray, color_img: Optional[np.ndarray] = None) -> np.ndarray:
        """K-means colour clustering — separates painted inscriptions by colour."""
        if color_img is not None and color_img.ndim == 3:
            # Use colour information
            data = color_img.reshape(-1, 3).astype(np.float32)
        else:
            data = gray.reshape(-1, 1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            data, self.kmeans_k, None, criteria, 5, cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()

        # The darkest cluster centre is assumed to be the inscription
        if centers.shape[1] == 3:
            brightness = centers.mean(axis=1)
        else:
            brightness = centers.flatten()
        fg_label = int(np.argmin(brightness))

        mask = np.where(labels == fg_label, 255, 0).astype(np.uint8)
        return mask.reshape(gray.shape)

    # ── Stage 2: Edge Detection (Canny) ───────────────────────────────────

    def detect_edges(self, enhanced: np.ndarray) -> np.ndarray:
        """Extract contours via Gaussian blur → Canny edge detection.

        Parameters
        ----------
        enhanced : np.ndarray
            Pre-processed grayscale image (uint8).

        Returns
        -------
        np.ndarray
            Binary edge map (0 or 255, uint8).
        """
        ksize = self.gaussian_ksize
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), self.blur_sigma)
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    # ── Orchestrator ──────────────────────────────────────────────────────

    def process(
        self,
        img: np.ndarray,
        return_intermediates: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run the full two-stage pipeline.

        Parameters
        ----------
        img : np.ndarray
            Input image (BGR colour or grayscale).
        return_intermediates : bool
            If ``True``, return a dict of every intermediate stage.
            If ``False`` (default), return only the final edge map.

        Returns
        -------
        np.ndarray or dict
            Final binary edge map, or a dict with keys:
            ``"grayscale"``, ``"enhanced"``, ``"edges"``.
        """
        gray = self._to_grayscale(img)
        enhanced = self.enhance_contrast(gray)
        binarized = self.binarize(enhanced, color_img=img)
        edges = self.detect_edges(enhanced)

        if return_intermediates:
            stages: Dict[str, np.ndarray] = {
                "grayscale": gray,
                "enhanced": enhanced,
            }
            if binarized is not None:
                stages["binarized"] = binarized
            stages["edges"] = edges
            return stages
        return edges
