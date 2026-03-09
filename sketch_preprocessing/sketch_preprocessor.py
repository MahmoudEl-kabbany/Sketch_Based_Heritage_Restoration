"""
Heritage Sketch Preprocessor
=============================
Cleans and isolates sketch strokes from aged, damaged, or scanned heritage
documents.  The pipeline removes background artifacts (stains, foxing, uneven
lighting, paper texture) while preserving fine line work.

Pipeline
--------
1. Grayscale conversion + intensity normalisation
2. Denoising  (Non-Local Means)
3. Background removal  (divide-by-blur method)
3b. Segmentation  (isolate sketch region from background artifacts)
4. Binarisation  (Sauvola threshold, with OpenCV adaptive fallback)
4b. Canny edge recovery  (recover faint lines missed by thresholding)
5. Morphological cleanup  (open → close)
6. Connected-component filtering  (remove tiny / huge blobs)

Usage
-----
    python sketch_preprocessor.py <image_path> [--output_dir <dir>]

All intermediate results are saved so you can inspect every stage.
"""

import argparse
import cv2
import numpy as np
import os

# Try to import Sauvola from scikit-image; fall back to OpenCV adaptive
try:
    from skimage.filters import threshold_sauvola
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠  scikit-image not found — falling back to OpenCV adaptive threshold.\n"
          "   Install it for better results:  pip install scikit-image")


# ═══════════════════════════════════════════════════════════════════════════
#  Default parameters  (tweak these for your specific documents)
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    # Denoising
    denoise_h=10,                # filter strength  (higher → more smoothing)
    denoise_template=7,          # template window size
    denoise_search=21,           # search window size

    # Background removal
    bg_blur_ksize=10,            # Gaussian blur kernel for background estimate

    # Segmentation
    seg_blur_ksize=51,           # heavy blur for region detection
    seg_morph_ksize=25,          # morph kernel to clean region mask
    seg_margin=10,               # dilate mask margin (pixels) to avoid clipping edges

    # Binarisation
    sauvola_window=25,           # local window size  (must be odd)
    sauvola_k=0.2,               # sensitivity  (lower → more ink kept)
    adaptive_block=15,           # OpenCV adaptive block size (fallback)
    adaptive_C=10,               # OpenCV adaptive constant   (fallback)

    # Canny edge recovery
    canny_low=30,                # Canny lower hysteresis threshold
    canny_high=100,              # Canny upper hysteresis threshold
    canny_dilate=2,              # dilation kernel to merge double-edges into strokes

    # Morphology
    morph_open_ksize=1,          # opening kernel — removes tiny noise specks
    morph_close_ksize=1,         # closing kernel — set to 1 to disable

    # Connected-component filtering
    cc_min_area=5,               # blobs smaller than this are removed
    cc_max_area_ratio=0.25,      # blobs larger than this fraction of image are removed
)


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════

def to_grayscale(img):
    """Convert to grayscale if needed and normalise intensity to full range."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def denoise(gray, h=10, template_ws=7, search_ws=21):
    """Non-Local Means denoising — reduces noise while keeping edges."""
    return cv2.fastNlMeansDenoising(gray, h=h,
                                     templateWindowSize=template_ws,
                                     searchWindowSize=search_ws)


def remove_background(gray, blur_ksize=51):
    """
    Divide-by-blur background removal.

    Divides the image by a heavily blurred version of itself, flattening
    illumination variations, stains, and paper-colour gradients.
    """
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    # scale=255 keeps the result in 0-255 uint8 range
    divided = cv2.divide(gray, bg, scale=255)
    return divided


def segment_sketch_region(gray, blur_ksize=51, morph_ksize=25, margin=10):
    """
    Segment the sketch region from the background using Otsu thresholding
    on a heavily blurred image.  Returns a mask where 255 = sketch region.

    This suppresses large-scale artifacts (scanner borders, paper edges,
    large stains) by identifying the area that actually contains ink.

    Parameters
    ----------
    gray        : grayscale input (after background removal)
    blur_ksize  : kernel for heavy blur before Otsu (larger = coarser regions)
    morph_ksize : kernel for morphological close+open to clean the mask
    margin      : pixels to dilate the mask to avoid clipping edges
    """
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Otsu finds optimal threshold to separate sketch from background
    _, region_mask = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up the mask: close gaps then remove small islands
    mk = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)

    # Dilate slightly so we don't clip sketch edges
    if margin > 0:
        margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (margin * 2 + 1, margin * 2 + 1))
        region_mask = cv2.dilate(region_mask, margin_kernel)

    # Count region coverage
    total = region_mask.shape[0] * region_mask.shape[1]
    covered = int(np.sum(region_mask > 0))
    print(f"    Segmentation: sketch region covers {covered/total:.1%} of image")

    return region_mask


def binarise(gray, method="sauvola", **kwargs):
    """
    Binarise the image.

    Parameters
    ----------
    method : str
        "sauvola"  – Sauvola local threshold  (needs scikit-image)
        "adaptive" – OpenCV adaptive Gaussian threshold
    """
    if method == "sauvola" and HAS_SKIMAGE:
        window = kwargs.get("sauvola_window", 25)
        k = kwargs.get("sauvola_k", 0.2)
        window = window if window % 2 == 1 else window + 1
        thresh_map = threshold_sauvola(gray, window_size=window, k=k)
        binary = ((gray < thresh_map).astype(np.uint8)) * 255
    else:
        block = kwargs.get("adaptive_block", 15)
        C = kwargs.get("adaptive_C", 10)
        block = block if block % 2 == 1 else block + 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, C
        )
    return binary


def canny_recovery(denoised, binary, low=30, high=100, dilate_ksize=2):
    """
    Recover faint lines missed by thresholding using Canny edge detection.

    Canny detects edges (intensity transitions), producing two parallel edges
    per stroke.  A small dilation merges them into a single filled stroke.
    The result is OR-ed with the existing binary to add back weak lines
    without replacing good detections.

    Parameters
    ----------
    denoised    : grayscale image (pre-binarisation, after denoising)
    binary      : existing binary mask from thresholding
    low, high   : Canny hysteresis thresholds
    dilate_ksize: kernel size for dilating edges (merges double-edges)
    """
    edges = cv2.Canny(denoised, low, high)

    if dilate_ksize > 0:
        kernel = np.ones((dilate_ksize, dilate_ksize), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel)

    # Combine: keep everything from the threshold + add recovered edges
    combined = cv2.bitwise_or(binary, edges)

    recovered_px = int(np.sum((edges > 0) & (binary == 0)))
    print(f"    Canny recovery: {recovered_px} new pixels added  "
          f"(thresholds={low}/{high}, dilate={dilate_ksize})")
    return combined


def morph_cleanup(binary, open_ksize=3, close_ksize=3):
    """
    Morphological opening (remove tiny specks) then closing (reconnect
    broken strokes).
    """
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    if close_ksize > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    return binary


def filter_components(binary, min_area=20, max_area_ratio=0.25):
    """
    Remove connected components that are too small (noise) or too large
    (background artefacts / borders).
    """
    total_pixels = binary.shape[0] * binary.shape[1]
    max_area = int(total_pixels * max_area_ratio)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = binary.copy()

    removed_small = 0
    removed_large = 0
    for i in range(1, num_labels):          # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            cleaned[labels == i] = 0
            removed_small += 1
        elif area > max_area:
            cleaned[labels == i] = 0
            removed_large += 1

    print(f"    CC filter: {num_labels - 1} components → "
          f"removed {removed_small} small, {removed_large} large")
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_sketch(image_path, output_dir=None, save_intermediates=True,
                      params=None):
    """
    Run the full heritage-sketch cleaning pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_dir : str or None
        Directory for outputs.  Defaults to ``<image_dir>/preprocessed/``.
    save_intermediates : bool
        If True, save each stage for inspection.
    params : dict or None
        Override any of the DEFAULTS parameters.

    Returns
    -------
    result : dict
        Dictionary with keys for each stage image (numpy arrays).
    """
    p = {**DEFAULTS, **(params or {})}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    def _save(name, img_data):
        if save_intermediates:
            path = os.path.join(output_dir, f"{basename}_{name}.png")
            cv2.imwrite(path, img_data)
            print(f"  ➜  {name:.<30s} saved to {path}")

    print(f"\n{'=' * 60}")
    print(f"  Preprocessing: {os.path.basename(image_path)}")
    print(f"{'=' * 60}")

    # 1. Grayscale + normalise
    gray = to_grayscale(img)
    _save("1_grayscale", gray)

    # 2. Denoise
    denoised = denoise(gray, h=p["denoise_h"],
                       template_ws=p["denoise_template"],
                       search_ws=p["denoise_search"])
    _save("2_denoised", denoised)

    # 3. Background removal
    bg_removed = remove_background(denoised, blur_ksize=p["bg_blur_ksize"])
    _save("3_bg_removed", bg_removed)

    # 3b. Segmentation — isolate sketch region, suppress background
    region_mask = segment_sketch_region(
        bg_removed,
        blur_ksize=p["seg_blur_ksize"],
        morph_ksize=p["seg_morph_ksize"],
        margin=p["seg_margin"])
    _save("3b_region_mask", region_mask)
    # Apply mask: set non-sketch areas to white (255) so they're ignored
    bg_removed_masked = bg_removed.copy()
    bg_removed_masked[region_mask == 0] = 255
    _save("3b_masked", bg_removed_masked)

    # 4. Binarise
    method = "sauvola" if HAS_SKIMAGE else "adaptive"
    binary = binarise(bg_removed_masked, method=method, **p)
    _save("4_binary", binary)

    # 4b. Canny edge recovery — add back faint lines missed by threshold
    binary = canny_recovery(denoised, binary,
                            low=p["canny_low"],
                            high=p["canny_high"],
                            dilate_ksize=p["canny_dilate"])
    _save("4b_canny_recovered", binary)

    # 5. Morphological cleanup
    cleaned = morph_cleanup(binary,
                            open_ksize=p["morph_open_ksize"],
                            close_ksize=p["morph_close_ksize"])
    _save("5_morph_cleaned", cleaned)

    # 6. Connected-component filtering
    final = filter_components(cleaned,
                              min_area=p["cc_min_area"],
                              max_area_ratio=p["cc_max_area_ratio"])
    _save("6_final", final)

    # Also save an inverted version (white bg / black strokes) for display
    display = cv2.bitwise_not(final)
    _save("7_display", display)

    print(f"\n✅  Done.  Final sketch: {output_dir}/{basename}_6_final.png")

    return {
        "original": img,
        "grayscale": gray,
        "denoised": denoised,
        "bg_removed": bg_removed,
        "binary": binary,
        "morph_cleaned": cleaned,
        "final": final,
        "display": display,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and isolate sketch strokes from heritage documents."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory  (default: <image_dir>/preprocessed/)")
    parser.add_argument("--denoise_h", type=int, default=DEFAULTS["denoise_h"],
                        help=f"Denoising strength (default: {DEFAULTS['denoise_h']})")
    parser.add_argument("--bg_blur", type=int, default=DEFAULTS["bg_blur_ksize"],
                        help=f"Background blur kernel (default: {DEFAULTS['bg_blur_ksize']})")
    parser.add_argument("--sauvola_window", type=int,
                        default=DEFAULTS["sauvola_window"],
                        help=f"Sauvola window size (default: {DEFAULTS['sauvola_window']})")
    parser.add_argument("--sauvola_k", type=float,
                        default=DEFAULTS["sauvola_k"],
                        help=f"Sauvola sensitivity (default: {DEFAULTS['sauvola_k']})")
    parser.add_argument("--min_area", type=int, default=DEFAULTS["cc_min_area"],
                        help=f"Min blob area to keep (default: {DEFAULTS['cc_min_area']})")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Only save the final result")

    args = parser.parse_args()

    user_params = {
        "denoise_h": args.denoise_h,
        "bg_blur_ksize": args.bg_blur,
        "sauvola_window": args.sauvola_window,
        "sauvola_k": args.sauvola_k,
        "cc_min_area": args.min_area,
    }

    preprocess_sketch(
        args.image,
        output_dir=args.output_dir,
        save_intermediates=not args.no_intermediates,
        params=user_params,
    )
