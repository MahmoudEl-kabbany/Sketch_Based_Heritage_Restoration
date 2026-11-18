"""Phase 1, step 0 — Preprocessing: from a grayscale scan to a stroke mask and its skeleton.

Everything later in the pipeline (the skeleton graph, the stroke radii, the
line width and the distance map) is derived from the one mask produced here.

PreprocessConfig() is Otsu threshold, speck removal, scikit-image's default
skeleton and the radius-based spur pruning of bezier._prune_skeleton_spurs,
plus background flattening on unevenly lit pages (background="auto": only
when the paper's brightness varies by more than UNEVEN_PAPER; on even paper
the result is exactly that of plain Otsu). The scorer is trained on images
binarized without the flattening (train_scoring.training_preprocess). The
other options are experimental variants, measured in docs/experiments/WP-3.md
and docs/experiments/hard-binarization.md.

A variant can be chosen per call (extract_paths(..., preprocess=cfg),
restore(..., preprocess=cfg)) or, for whole benchmark runs whose worker
processes call extract_paths themselves, through the environment variable
SKETCH_PREPROCESS holding a JSON object of field overrides, e.g.
    SKETCH_PREPROCESS='{"threshold": "sauvola"}' python check_changes.py
(check_changes.py --preprocess '...' sets it).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import NamedTuple, Optional

import cv2
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola
from skimage.morphology import skeletonize

ENV_VAR = "SKETCH_PREPROCESS"

# Blobs smaller than this many squared line widths are specks, not strokes.
SPECK_AREA_WIDTHS2 = 0.5

THRESHOLDS = ("otsu", "sauvola", "niblack", "adaptive_gaussian")
DENOISERS = ("none", "median", "nlmeans")
SKELETONS = ("zhang", "lee")
BACKGROUNDS = ("none", "divide", "auto")
# "auto" flattens the paper only when its estimated brightness varies by more than
# this share (1st to 99th percentile). The 64 benchmark images and the synthetic
# "easy" pages measure at most 0.028, stained or unevenly lit pages ("medium",
# "hard") at least 0.080.
UNEVEN_PAPER = 0.05


@dataclass(frozen=True)
class PreprocessConfig:
    threshold: str = "otsu"            # one of THRESHOLDS
    window: int = 25                   # odd window (px) of the local thresholds
    k: float = 0.2                     # Sauvola / Niblack k
    adaptive_c: float = 10.0           # offset of the adaptive Gaussian threshold
    denoise: str = "none"              # one of DENOISERS, applied to the gray image before thresholding
    denoise_size: int = 3              # median kernel (odd, px) / nlmeans filter strength h
    close_px: int = 0                  # morphological closing of the mask (elliptical kernel, px); 0 = off
    skeleton: str = "zhang"            # one of SKELETONS ("zhang" is scikit-image's default)
    junction_aware_pruning: bool = False   # prune a spur only where its junction keeps two long branches
    # Uneven paper (stains, light gradient): divide by an estimate of the paper's brightness first.
    background: str = "auto"           # one of BACKGROUNDS (training images use "none", see train_scoring.py)
    background_size: float = 0.06      # closing kernel for the paper estimate, as a share of the image diagonal
    window_relative: float = 0.0       # if > 0: local-threshold window as a share of the diagonal (overrides window)

    def __post_init__(self) -> None:
        for name, allowed in (("threshold", THRESHOLDS), ("denoise", DENOISERS), ("skeleton", SKELETONS),
                              ("background", BACKGROUNDS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"PreprocessConfig.{name} must be one of {allowed}, not {getattr(self, name)!r}")

    @classmethod
    def from_dict(cls, values: dict) -> "PreprocessConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown preprocessing option(s): {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_env(cls) -> "PreprocessConfig":
        """The configuration named by SKETCH_PREPROCESS, or the default."""
        text = os.environ.get(ENV_VAR, "").strip()
        return cls.from_dict(json.loads(text)) if text else cls()

    def to_dict(self) -> dict:
        return asdict(self)


def resolve(cfg: Optional[PreprocessConfig]) -> PreprocessConfig:
    return cfg if cfg is not None else PreprocessConfig.from_env()


class StrokeGeometry(NamedTuple):
    binary: np.ndarray       # stroke mask, 255 on strokes, specks removed
    radius_map: np.ndarray   # distance of every stroke pixel to the background
    line_width: float        # twice the median radius along the skeleton
    skeleton: np.ndarray     # bool skeleton of *binary*


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def paper_estimate(img: np.ndarray, size_fraction: float) -> np.ndarray:
    """The paper's brightness without the strokes (float32, same size as *img*).

    A grey-level closing with a disk wider than any stroke (and than filled
    details such as a drawn pupil) removes the dark strokes and leaves the
    paper, which is then smoothed.
    """
    h, w = img.shape
    size = _odd(int(round(size_fraction * float(np.hypot(h, w)))))
    # Close on a reduced image (the paper varies slowly), then bring it back to full size.
    factor = max(1, size // 15)
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    k = _odd(size // factor)
    paper = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    paper = cv2.GaussianBlur(paper, (0, 0), k / 2.0)
    return cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def paper_unevenness(paper: np.ndarray) -> float:
    """How much the paper's brightness varies: (99th - 1st percentile) / 99th percentile."""
    lo, hi = np.percentile(paper, [1, 99])
    return float((hi - lo) / max(hi, 1.0))


def flatten_background(img: np.ndarray, size_fraction: float, paper: Optional[np.ndarray] = None) -> np.ndarray:
    """Divide out the paper's brightness: stains and light gradients become flat white."""
    paper = paper_estimate(img, size_fraction) if paper is None else paper
    flat = img.astype(np.float32) / np.maximum(paper, 1.0) * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def binarize(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Stroke mask (uint8, 255 on dark strokes) of a grayscale image."""
    if cfg.background == "divide":
        img = flatten_background(img, cfg.background_size)
    elif cfg.background == "auto":
        paper = paper_estimate(img, cfg.background_size)
        if paper_unevenness(paper) > UNEVEN_PAPER:
            img = flatten_background(img, cfg.background_size, paper)
    window = cfg.window
    if cfg.window_relative > 0:
        window = int(round(cfg.window_relative * float(np.hypot(*img.shape))))
    if cfg.denoise == "median":
        img = cv2.medianBlur(img, _odd(cfg.denoise_size))
    elif cfg.denoise == "nlmeans":
        img = cv2.fastNlMeansDenoising(img, None, h=float(cfg.denoise_size))

    if cfg.threshold == "otsu":
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif cfg.threshold == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, _odd(window), cfg.adaptive_c,
        )
    else:
        local = threshold_sauvola if cfg.threshold == "sauvola" else threshold_niblack
        binary = np.where(img < local(img, window_size=_odd(window), k=cfg.k), 255, 0).astype(np.uint8)

    if cfg.close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(cfg.close_px), int(cfg.close_px)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def skeleton_of(mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    if cfg.skeleton == "lee":
        return skeletonize(mask > 0, method="lee") > 0
    return skeletonize(mask > 0)


def stroke_geometry_of(img: np.ndarray, cfg: Optional[PreprocessConfig] = None) -> StrokeGeometry:
    """Binarize a grayscale sketch and measure its strokes (see StrokeGeometry)."""
    cfg = resolve(cfg)
    binary = binarize(img, cfg)

    def measure(mask: np.ndarray):
        radius_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        skeleton = skeleton_of(mask, cfg)
        on_skeleton = radius_map[skeleton]
        return radius_map, (2.0 * float(np.median(on_skeleton)) if len(on_skeleton) else 2.0), skeleton

    radius_map, line_width, skeleton = measure(binary)

    # Remove specks: blobs much smaller than a piece of stroke one line width long.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    specks = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] < SPECK_AREA_WIDTHS2 * line_width ** 2]
    if specks:
        binary[np.isin(labels, specks)] = 0
        radius_map, line_width, skeleton = measure(binary)

    return StrokeGeometry(binary, radius_map, line_width, skeleton)


def stroke_geometry_from_file(image_path: str, cfg: Optional[PreprocessConfig] = None) -> StrokeGeometry:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return stroke_geometry_of(img, cfg)
