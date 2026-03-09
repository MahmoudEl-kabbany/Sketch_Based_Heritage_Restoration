"""
Advanced Heritage Sketch Preprocessing Pipeline
================================================
A 10-step pipeline designed to clean scanned heritage sketches by removing
background artifacts from age, damage, and scanning while preserving fine
stroke detail and identifying damaged regions.

Pipeline
--------
 1. Load image → grayscale
 2. Bilateral filter  (JPEG artifact removal)
 3. Morph-dilation background estimation + division normalisation
 4. Rolling-ball background subtraction  (uneven illumination / stains)
 5. CLAHE  (local contrast enhancement)
 6. Frangi vesselness filter  (recover faded / thin strokes)
 7. Sauvola adaptive thresholding  → binary mask
 8. Hole filling + damage-mask generation  (scipy ndimage)
 9. Skeletonisation + branch-point verification
10. Output: enhanced grayscale, clean binary mask, damage map

Dependencies
------------
    pip install opencv-python numpy scikit-image scipy

Usage
-----
    python advanced_preprocessor.py <image_path> [--output_dir <dir>]

Or import programmatically:
    from advanced_preprocessor import run_pipeline
    results = run_pipeline("path/to/image.png")
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
    bilateral_d=9,               # pixel neighbourhood diameter
    bilateral_sigma_color=75,    # colour-space filter sigma
    bilateral_sigma_space=75,    # coordinate-space filter sigma

    # Step 3 — Background estimation
    bg_dilate_ksize=21,          # dilation kernel for background estimate

    # Step 4 — Rolling-ball
    rolling_ball_radius=50,      # ball radius (larger = catches bigger stains)

    # Step 5 — CLAHE
    clahe_clip=2.0,              # contrast limit
    clahe_grid=(8, 8),           # tile grid size

    # Step 6 — Frangi
    frangi_scale_min=1,          # minimum sigma for Frangi
    frangi_scale_max=5,          # maximum sigma for Frangi
    frangi_scale_step=1,         # sigma step
    frangi_beta1=0.5,            # Frangi structureness parameter
    frangi_beta2=15,             # Frangi blobness parameter
    frangi_blend_weight=0.4,     # how much Frangi to blend back in (0–1)

    # Step 7 — Sauvola
    sauvola_window=25,           # local window size (must be odd)
    sauvola_k=0.2,               # sensitivity (lower → keeps more ink)

    # Step 8 — Damage mask
    damage_fill_structure=3,     # structuring element connectivity for fill
    damage_min_region=500,       # minimum hole area to flag as damage
    damage_max_region_ratio=0.4, # max hole area as fraction of image

    # Step 9 — Skeleton
    # (no tuneable params — just verification)
)


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline steps
# ═══════════════════════════════════════════════════════════════════════════

# ── Step 1 ────────────────────────────────────────────────────────────────
def step1_load_grayscale(image_path):
    """Load image and convert to grayscale."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"  Step 1  ✓  Loaded {gray.shape[1]}×{gray.shape[0]} grayscale")
    return img_bgr, gray


# ── Step 2 ────────────────────────────────────────────────────────────────
def step2_bilateral_filter(gray, d=9, sigma_color=75, sigma_space=75):
    """Remove JPEG compression artifacts while preserving stroke edges."""
    filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    print(f"  Step 2  ✓  Bilateral filter  (d={d}, σc={sigma_color}, σs={sigma_space})")
    return filtered


# ── Step 3 ────────────────────────────────────────────────────────────────
def step3_background_normalise(gray, dilate_ksize=21):
    """
    Estimate the background by morphological dilation, then divide the
    grayscale image by the background to normalise uneven illumination.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilate_ksize, dilate_ksize))
    bg_estimate = cv2.dilate(gray, kernel)
    # Avoid division by zero
    bg_estimate = np.maximum(bg_estimate, 1).astype(np.float32)
    normalised = cv2.divide(gray.astype(np.float32), bg_estimate, scale=255)
    normalised = np.clip(normalised, 0, 255).astype(np.uint8)
    print(f"  Step 3  ✓  Background normalisation  (dilate kernel={dilate_ksize})")
    return normalised, bg_estimate.astype(np.uint8)


# ── Step 4 ────────────────────────────────────────────────────────────────
def step4_rolling_ball(gray, radius=50):
    """
    Rolling-ball background subtraction (scikit-image) to correct for
    large-scale illumination gradients and stains.
    """
    # rolling_ball expects float image
    gray_float = gray.astype(np.float64)
    background = rolling_ball(gray_float, radius=radius)
    subtracted = np.clip(gray_float - background, 0, 255).astype(np.uint8)
    print(f"  Step 4  ✓  Rolling-ball subtraction  (radius={radius})")
    return subtracted, background.astype(np.uint8)


# ── Step 5 ────────────────────────────────────────────────────────────────
def step5_clahe(gray, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE to enhance local contrast without amplifying noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(gray)
    print(f"  Step 5  ✓  CLAHE  (clip={clip_limit}, grid={grid_size})")
    return enhanced


# ── Step 6 ────────────────────────────────────────────────────────────────
def step6_frangi(gray, scale_min=1, scale_max=5, scale_step=1,
                 beta1=0.5, beta2=15, blend_weight=0.4):
    """
    Apply Frangi vesselness filter to enhance elongated structures
    (thin/faded strokes).  The result is blended back into the enhanced
    grayscale to recover lost detail.
    """
    gray_float = gray.astype(np.float64) / 255.0

    sigmas = range(scale_min, scale_max + 1, scale_step)
    # Frangi returns float64 in [0, 1]; black_ridges=True detects dark lines
    vessel = frangi(gray_float, sigmas=sigmas,
                    beta=beta1, gamma=beta2,
                    black_ridges=True)

    # Normalise to 0-255
    vessel_norm = (vessel / (vessel.max() + 1e-8) * 255).astype(np.uint8)

    # Blend: enhanced = gray − weight * vesselness  (darken where strokes are)
    blended = np.clip(
        gray.astype(np.float32) - blend_weight * vessel_norm.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    print(f"  Step 6  ✓  Frangi vesselness  (σ={list(sigmas)}, blend={blend_weight})")
    return blended, vessel_norm


# ── Step 7 ────────────────────────────────────────────────────────────────
def step7_sauvola(gray, window_size=25, k=0.2):
    """
    Sauvola adaptive thresholding — works well for documents with varying
    background intensity.
    """
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    thresh_map = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = ((gray < thresh_map).astype(np.uint8)) * 255
    print(f"  Step 7  ✓  Sauvola threshold  (window={window_size}, k={k})")
    return binary


# ── Step 8 ────────────────────────────────────────────────────────────────
def step8_damage_mask(binary, fill_structure=3, min_region=500,
                      max_region_ratio=0.4):
    """
    Fill holes in the binary image, then identify large interior gaps that
    indicate torn or missing regions (damage), as opposed to intentional
    negative space (small holes).

    Returns
    -------
    filled   : binary image with holes filled
    damage   : damage mask (255 = damaged region, 0 = ok)
    """
    total_pixels = binary.shape[0] * binary.shape[1]
    max_region = int(total_pixels * max_region_ratio)

    # Fill holes using scipy ndimage
    struct = ndimage.generate_binary_structure(2, fill_structure)
    filled = ndimage.binary_fill_holes(binary > 0, structure=struct)
    filled = (filled.astype(np.uint8)) * 255

    # Holes = regions that were filled (were background, now foreground)
    holes = cv2.subtract(filled, binary)

    # Label each hole region and filter by size
    labelled, num_features = ndimage.label(holes > 0)
    damage = np.zeros_like(binary)

    damage_count = 0
    for i in range(1, num_features + 1):
        region_area = np.sum(labelled == i)
        if min_region <= region_area <= max_region:
            damage[labelled == i] = 255
            damage_count += 1

    print(f"  Step 8  ✓  Damage mask  ({num_features} holes found, "
          f"{damage_count} flagged as damage)")
    return filled, damage


# ── Step 9 ────────────────────────────────────────────────────────────────
def step9_skeletonise(binary):
    """
    Skeletonise the binary image to single-pixel-width strokes, then
    count branch points to verify stroke continuity.

    A branch point is a skeleton pixel with ≥ 3 neighbours.
    """
    # skeletonize expects bool
    skel = skeletonize(binary > 0)
    skel_uint8 = (skel.astype(np.uint8)) * 255

    # Count branch points (pixels with 3+ neighbours in skeleton)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # don't count the pixel itself
    neighbour_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    branch_points = np.logical_and(skel, neighbour_count >= 3)
    num_branches = int(np.sum(branch_points))

    # Also count endpoints (pixels with exactly 1 neighbour)
    endpoints = np.logical_and(skel, neighbour_count == 1)
    num_endpoints = int(np.sum(endpoints))

    total_skel_pixels = int(np.sum(skel))
    print(f"  Step 9  ✓  Skeleton  ({total_skel_pixels} px, "
          f"{num_branches} branch pts, {num_endpoints} endpoints)")

    # Create a visualisation: skeleton in white, branches in red, endpoints in blue
    skel_vis = cv2.cvtColor(skel_uint8, cv2.COLOR_GRAY2BGR)
    skel_vis[branch_points] = (0, 0, 255)    # red = branch points
    skel_vis[endpoints] = (255, 0, 0)         # blue = endpoints

    return skel_uint8, skel_vis, num_branches, num_endpoints


# ═══════════════════════════════════════════════════════════════════════════
#  Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(image_path, output_dir=None, save_intermediates=True,
                 params=None):
    """
    Run the full 10-step heritage sketch preprocessing pipeline.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_dir : str or None
        Directory for outputs.  Defaults to ``<image_dir>/preprocessed/``.
    save_intermediates : bool
        If True, save an image for every stage.
    params : dict or None
        Override any DEFAULTS parameter.

    Returns
    -------
    dict with keys:
        enhanced_gray, binary_mask, damage_map,
        skeleton, skeleton_vis, branch_count, endpoint_count
    """
    p = {**DEFAULTS, **(params or {})}

    basename = os.path.splitext(os.path.basename(image_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    def _save(name, img_data):
        if save_intermediates:
            path = os.path.join(output_dir, f"{basename}_{name}.png")
            cv2.imwrite(path, img_data)

    print(f"\n{'═' * 60}")
    print(f"  Advanced Preprocessing: {os.path.basename(image_path)}")
    print(f"{'═' * 60}")

    # Step 1 — Load + grayscale
    img_bgr, gray = step1_load_grayscale(image_path)
    _save("01_grayscale", gray)

    # Step 2 — Bilateral filter
    bilateral = step2_bilateral_filter(
        gray, d=p["bilateral_d"],
        sigma_color=p["bilateral_sigma_color"],
        sigma_space=p["bilateral_sigma_space"])
    _save("02_bilateral", bilateral)

    # Step 3 — Background normalisation
    normalised, bg_est = step3_background_normalise(
        bilateral, dilate_ksize=p["bg_dilate_ksize"])
    _save("03_normalised", normalised)
    _save("03_bg_estimate", bg_est)

    # Step 4 — Rolling-ball subtraction
    rb_sub, rb_bg = step4_rolling_ball(
        normalised, radius=p["rolling_ball_radius"])
    _save("04_rolling_ball", rb_sub)
    _save("04_rb_background", rb_bg)

    # Step 5 — CLAHE
    enhanced = step5_clahe(
        rb_sub, clip_limit=p["clahe_clip"], grid_size=p["clahe_grid"])
    _save("05_clahe", enhanced)

    # Step 6 — Frangi vesselness
    frangi_blended, vessel_map = step6_frangi(
        enhanced,
        scale_min=p["frangi_scale_min"],
        scale_max=p["frangi_scale_max"],
        scale_step=p["frangi_scale_step"],
        beta1=p["frangi_beta1"],
        beta2=p["frangi_beta2"],
        blend_weight=p["frangi_blend_weight"])
    _save("06_frangi_blended", frangi_blended)
    _save("06_vesselness", vessel_map)

    # Step 7 — Sauvola binarisation
    binary = step7_sauvola(
        frangi_blended,
        window_size=p["sauvola_window"],
        k=p["sauvola_k"])
    _save("07_binary", binary)

    # Step 8 — Damage mask
    filled, damage = step8_damage_mask(
        binary,
        fill_structure=p["damage_fill_structure"],
        min_region=p["damage_min_region"],
        max_region_ratio=p["damage_max_region_ratio"])
    _save("08_filled", filled)
    _save("08_damage_mask", damage)

    # Step 9 — Skeletonisation + branch-point verification
    skel, skel_vis, n_branches, n_endpoints = step9_skeletonise(binary)
    _save("09_skeleton", skel)
    _save("09_skeleton_vis", skel_vis)

    # Step 10 — Final outputs (the three artifacts for the next stage)
    # a) Enhanced grayscale
    _save("10_enhanced_grayscale", frangi_blended)
    # b) Clean binary mask  (white strokes on black bg)
    _save("10_binary_mask", binary)
    # c) Damage map
    _save("10_damage_map", damage)
    # d) Display version of binary (black strokes on white bg)
    display = cv2.bitwise_not(binary)
    _save("10_display", display)

    print(f"\n{'─' * 60}")
    print(f"  Final outputs saved to: {output_dir}/")
    print(f"    • {basename}_10_enhanced_grayscale.png")
    print(f"    • {basename}_10_binary_mask.png")
    print(f"    • {basename}_10_damage_map.png")
    print(f"    • {basename}_10_display.png")
    print(f"{'─' * 60}")
    print(f"  Skeleton stats: {n_branches} branch points, {n_endpoints} endpoints")
    print(f"{'═' * 60}\n")

    return {
        "original": img_bgr,
        "enhanced_gray": frangi_blended,
        "binary_mask": binary,
        "damage_map": damage,
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
        description="Advanced heritage sketch preprocessing pipeline."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Output directory (default: <image_dir>/preprocessed/)")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Only save the three final artifacts")

    # Expose key parameters
    parser.add_argument("--bilateral_d", type=int, default=DEFAULTS["bilateral_d"])
    parser.add_argument("--bg_dilate", type=int, default=DEFAULTS["bg_dilate_ksize"])
    parser.add_argument("--rb_radius", type=int, default=DEFAULTS["rolling_ball_radius"])
    parser.add_argument("--clahe_clip", type=float, default=DEFAULTS["clahe_clip"])
    parser.add_argument("--frangi_blend", type=float, default=DEFAULTS["frangi_blend_weight"])
    parser.add_argument("--sauvola_window", type=int, default=DEFAULTS["sauvola_window"])
    parser.add_argument("--sauvola_k", type=float, default=DEFAULTS["sauvola_k"])
    parser.add_argument("--damage_min", type=int, default=DEFAULTS["damage_min_region"])

    args = parser.parse_args()

    user_params = {
        "bilateral_d": args.bilateral_d,
        "bg_dilate_ksize": args.bg_dilate,
        "rolling_ball_radius": args.rb_radius,
        "clahe_clip": args.clahe_clip,
        "frangi_blend_weight": args.frangi_blend,
        "sauvola_window": args.sauvola_window,
        "sauvola_k": args.sauvola_k,
        "damage_min_region": args.damage_min,
    }

    run_pipeline(
        args.image,
        output_dir=args.output_dir,
        save_intermediates=not args.no_intermediates,
        params=user_params,
    )
