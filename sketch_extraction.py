"""
Unified Sketch Extraction Pipeline
====================================
Combines the best techniques from both heritage sketch preprocessors into a
single optimised pipeline.  Goal: produce a clean binary image where sketch
lines are foreground (white on black).

Pipeline
--------
 1. Load + grayscale + intensity normalisation
 2. Bilateral filter  (edge-preserving denoising / JPEG artifact removal)
 3. Morph-dilation background normalisation
 4. Rolling-ball background subtraction  (illumination correction)
 5. CLAHE  (local contrast enhancement)
 6. Frangi vesselness  (recover faded / thin strokes)
 7. Segmentation  (isolate sketch region, suppress borders & large stains)
 8. Sauvola binarisation + Canny edge recovery
 9. Connected-component filtering  (remove noise blobs)
10. Skeleton verification  (validate stroke continuity)

Dependencies
------------
    pip install opencv-python numpy scikit-image scipy

Usage
-----
    python sketch_extraction.py <image_path> [--output_dir <dir>]

    from sketch_extraction import extract_sketch
    result = extract_sketch("path/to/image.png")
"""

import argparse
import cv2
import numpy as np
import os

from skimage.filters import threshold_sauvola, frangi
from skimage.morphology import skeletonize
from skimage.restoration import rolling_ball
from scipy import ndimage


# ═══════════════════════════════════════════════════════════════════════════
#  Default parameters
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    # Step 2 — Bilateral filter
    bilateral_d=9,
    bilateral_sigma_color=75,
    bilateral_sigma_space=75,

    # Step 3 — Background normalisation (morph-dilation)
    bg_dilate_ksize=21,

    # Step 4 — Rolling-ball
    rolling_ball_radius=50,

    # Step 5 — CLAHE
    clahe_clip=2.0,
    clahe_grid=(8, 8),

    # Step 6 — Frangi vesselness
    frangi_scale_min=1,
    frangi_scale_max=5,
    frangi_scale_step=1,
    frangi_beta1=0.5,
    frangi_beta2=15,
    frangi_blend_weight=0.4,

    # Step 7 — Segmentation
    seg_blur_ksize=51,
    seg_morph_ksize=25,
    seg_margin=10,

    # Step 8 — Sauvola + Canny recovery
    sauvola_window=25,
    sauvola_k=0.2,
    canny_low=30,
    canny_high=100,
    canny_dilate=2,

    # Step 9 — CC filtering
    cc_min_area=5,
    cc_max_area_ratio=0.25,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Step functions
# ═══════════════════════════════════════════════════════════════════════════

def step01_grayscale(img):
    """Convert to grayscale and normalise intensity to full 0–255 range."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    print(f"  Step 1  ✓  Grayscale {gray.shape[1]}×{gray.shape[0]}")
    return gray


def step02_bilateral(gray, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter — smooths noise / JPEG artifacts while keeping edges."""
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    print(f"  Step 2  ✓  Bilateral filter  (d={d}, σc={sigma_color}, σs={sigma_space})")
    return filtered


def step03_bg_normalise(gray, dilate_ksize=21):
    """
    Morph-dilation background estimation + division normalisation.
    Handles paper texture and uneven tone better than simple Gaussian divide.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_ksize, dilate_ksize))
    bg = cv2.dilate(gray, kernel)
    bg = np.maximum(bg, 1).astype(np.float32)
    normalised = cv2.divide(gray.astype(np.float32), bg, scale=255)
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    print(f"  Step 3  ✓  Background normalisation  (dilate kernel={dilate_ksize})")
    return normalised


def step04_rolling_ball(gray, radius=50):
    """Rolling-ball subtraction — corrects large-scale illumination and stains."""
    gray_f = gray.astype(np.float64)
    background = rolling_ball(gray_f, radius=radius)
    subtracted = np.clip(gray_f - background, 0, 255).astype(np.uint8)
    print(f"  Step 4  ✓  Rolling-ball subtraction  (radius={radius})")
    return subtracted


def step05_clahe(gray, clip_limit=2.0, grid_size=(8, 8)):
    """CLAHE — boosts local contrast without amplifying noise."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    print(f"  Step 5  ✓  CLAHE  (clip={clip_limit}, grid={grid_size})")
    return enhanced


def step06_frangi(gray, scale_min=1, scale_max=5, scale_step=1,
                  beta1=0.5, beta2=15, blend_weight=0.4):
    """
    Frangi vesselness — enhances elongated structures (thin/faded strokes).
    Blends the vesselness response back into the grayscale to darken strokes.
    """
    gray_f = gray.astype(np.float64) / 255.0
    sigmas = range(scale_min, scale_max + 1, scale_step)
    vessel = frangi(gray_f, sigmas=sigmas, beta=beta1, gamma=beta2,
                    black_ridges=True)
    vessel_norm = (vessel / (vessel.max() + 1e-8) * 255).astype(np.uint8)

    blended = np.clip(
        gray.astype(np.float32) - blend_weight * vessel_norm.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    print(f"  Step 6  ✓  Frangi vesselness  (σ={list(sigmas)}, blend={blend_weight})")
    return blended, vessel_norm


def step07_segment(gray, blur_ksize=51, morph_ksize=25, margin=10):
    """
    Otsu-based segmentation mask — isolates the sketch region from scanner
    borders, large stains, and other background artifacts.
    """
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    _, mask = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mk = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if margin > 0:
        margin_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        mask = cv2.dilate(mask, margin_k)

    coverage = np.sum(mask > 0) / mask.size
    print(f"  Step 7  ✓  Segmentation  (sketch region = {coverage:.1%} of image)")

    # Apply: set non-sketch areas to white so Sauvola ignores them
    masked = gray.copy()
    masked[mask == 0] = 255
    return masked, mask


def step08_binarise_and_recover(gray, denoised,
                                 sauvola_window=25, sauvola_k=0.2,
                                 canny_low=30, canny_high=100,
                                 canny_dilate=2):
    """
    Sauvola adaptive binarisation followed by Canny edge recovery.
    Sauvola captures the main strokes; Canny adds back faint lines that
    thresholding missed.
    """
    # Sauvola
    window = sauvola_window if sauvola_window % 2 == 1 else sauvola_window + 1
    thresh_map = threshold_sauvola(gray, window_size=window, k=sauvola_k)
    binary = ((gray < thresh_map).astype(np.uint8)) * 255
    print(f"  Step 8a ✓  Sauvola threshold  (window={window}, k={sauvola_k})")

    # Canny recovery
    edges = cv2.Canny(denoised, canny_low, canny_high)
    if canny_dilate > 0:
        kernel = np.ones((canny_dilate, canny_dilate), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel)
    recovered = int(np.sum((edges > 0) & (binary == 0)))
    binary = cv2.bitwise_or(binary, edges)
    print(f"  Step 8b ✓  Canny recovery  (+{recovered} px, "
          f"thresholds={canny_low}/{canny_high})")
    return binary


def step09_cc_filter(binary, min_area=5, max_area_ratio=0.25):
    """Remove connected components that are too small (noise) or too large."""
    total = binary.shape[0] * binary.shape[1]
    max_area = int(total * max_area_ratio)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = binary.copy()

    n_small = n_large = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            cleaned[labels == i] = 0
            n_small += 1
        elif area > max_area:
            cleaned[labels == i] = 0
            n_large += 1

    print(f"  Step 9  ✓  CC filter  ({num_labels - 1} components, "
          f"removed {n_small} small + {n_large} large)")
    return cleaned


def step10_skeleton(binary):
    """
    Skeletonise to single-pixel-width and count branch / end points
    to verify stroke continuity.
    """
    skel = skeletonize(binary > 0)
    skel_u8 = (skel.astype(np.uint8)) * 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    branches = np.logical_and(skel, neighbours >= 3)
    endpoints = np.logical_and(skel, neighbours == 1)

    n_b = int(np.sum(branches))
    n_e = int(np.sum(endpoints))
    n_px = int(np.sum(skel))

    # Visualisation
    vis = cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2BGR)
    vis[branches] = (0, 0, 255)   # red
    vis[endpoints] = (255, 0, 0)  # blue

    print(f"  Step 10 ✓  Skeleton  ({n_px} px, {n_b} branches, {n_e} endpoints)")
    return skel_u8, vis, n_b, n_e


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def extract_sketch(image_path, output_dir=None, save_intermediates=True,
                   params=None):
    """
    Run the unified 10-step sketch extraction pipeline.

    Returns dict with: binary_mask, display, enhanced_gray,
    skeleton, skeleton_vis, branch_count, endpoint_count.
    """
    p = {**DEFAULTS, **(params or {})}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    basename = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    def _save(name, data):
        if save_intermediates:
            path = os.path.join(output_dir, f"{basename}_{name}.png")
            cv2.imwrite(path, data)

    print(f"\n{'═' * 60}")
    print(f"  Sketch Extraction: {os.path.basename(image_path)}")
    print(f"{'═' * 60}")

    # 1. Grayscale
    gray = step01_grayscale(img)
    _save("01_grayscale", gray)

    # 2. Bilateral filter
    bilateral = step02_bilateral(gray, p["bilateral_d"],
                                 p["bilateral_sigma_color"],
                                 p["bilateral_sigma_space"])
    _save("02_bilateral", bilateral)

    # 3. Background normalisation
    normalised = step03_bg_normalise(bilateral, p["bg_dilate_ksize"])
    _save("03_normalised", normalised)

    # 4. Rolling-ball subtraction
    rb_sub = step04_rolling_ball(normalised, p["rolling_ball_radius"])
    _save("04_rolling_ball", rb_sub)

    # 5. CLAHE
    enhanced = step05_clahe(rb_sub, p["clahe_clip"], p["clahe_grid"])
    _save("05_clahe", enhanced)

    # 6. Frangi vesselness
    frangi_blended, vessel_map = step06_frangi(
        enhanced, p["frangi_scale_min"], p["frangi_scale_max"],
        p["frangi_scale_step"], p["frangi_beta1"], p["frangi_beta2"],
        p["frangi_blend_weight"])
    _save("06_frangi", frangi_blended)
    _save("06_vesselness", vessel_map)

    # 7. Segmentation
    masked, region_mask = step07_segment(
        frangi_blended, p["seg_blur_ksize"], p["seg_morph_ksize"],
        p["seg_margin"])
    _save("07_region_mask", region_mask)
    _save("07_masked", masked)

    # 8. Sauvola + Canny recovery
    binary = step08_binarise_and_recover(
        masked, bilateral,
        p["sauvola_window"], p["sauvola_k"],
        p["canny_low"], p["canny_high"], p["canny_dilate"])
    _save("08_binary", binary)

    # 9. CC filtering
    final = step09_cc_filter(binary, p["cc_min_area"], p["cc_max_area_ratio"])
    _save("09_final", final)

    # 10. Skeleton verification
    skel, skel_vis, n_branches, n_endpoints = step10_skeleton(final)
    _save("10_skeleton", skel)
    _save("10_skeleton_vis", skel_vis)

    # ── Save final outputs ──
    display = cv2.bitwise_not(final)
    _save("final_binary_mask", final)
    _save("final_display", display)
    _save("final_enhanced_gray", frangi_blended)

    print(f"\n{'─' * 60}")
    print(f"  Outputs → {output_dir}/")
    print(f"    • {basename}_final_binary_mask.png  (white lines, black bg)")
    print(f"    • {basename}_final_display.png      (black lines, white bg)")
    print(f"    • {basename}_final_enhanced_gray.png")
    print(f"  Skeleton: {n_branches} branch points, {n_endpoints} endpoints")
    print(f"{'═' * 60}\n")

    return {
        "original": img,
        "enhanced_gray": frangi_blended,
        "binary_mask": final,
        "display": display,
        "skeleton": skel,
        "skeleton_vis": skel_vis,
        "branch_count": n_branches,
        "endpoint_count": n_endpoints,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clean sketch lines from heritage document images."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--output_dir", "-o", default=None)
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Only save final outputs")

    # Key tuneable parameters
    parser.add_argument("--bilateral_d", type=int, default=DEFAULTS["bilateral_d"])
    parser.add_argument("--bg_dilate", type=int, default=DEFAULTS["bg_dilate_ksize"])
    parser.add_argument("--rb_radius", type=int, default=DEFAULTS["rolling_ball_radius"])
    parser.add_argument("--clahe_clip", type=float, default=DEFAULTS["clahe_clip"])
    parser.add_argument("--frangi_blend", type=float, default=DEFAULTS["frangi_blend_weight"])
    parser.add_argument("--sauvola_window", type=int, default=DEFAULTS["sauvola_window"])
    parser.add_argument("--sauvola_k", type=float, default=DEFAULTS["sauvola_k"])
    parser.add_argument("--canny_low", type=int, default=DEFAULTS["canny_low"])
    parser.add_argument("--canny_high", type=int, default=DEFAULTS["canny_high"])
    parser.add_argument("--min_area", type=int, default=DEFAULTS["cc_min_area"])

    args = parser.parse_args()

    extract_sketch(
        args.image,
        output_dir=args.output_dir,
        save_intermediates=not args.no_intermediates,
        params={
            "bilateral_d": args.bilateral_d,
            "bg_dilate_ksize": args.bg_dilate,
            "rolling_ball_radius": args.rb_radius,
            "clahe_clip": args.clahe_clip,
            "frangi_blend_weight": args.frangi_blend,
            "sauvola_window": args.sauvola_window,
            "sauvola_k": args.sauvola_k,
            "canny_low": args.canny_low,
            "canny_high": args.canny_high,
            "cc_min_area": args.min_area,
        },
    )
