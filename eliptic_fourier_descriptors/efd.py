"""
Elliptic Fourier Descriptor (EFD) Extraction
=============================================
Raster-only EFD extraction and contour reconstruction.

Raster pipeline:  image → denoise → threshold → findContours → EFD

Demonstrates:
    1. Loading & binarizing a raster image
  2. Extracting contours & properties (area, perimeter, bounding box)
  3. Computing EFD coefficients (raw and normalized)
  4. Reconstructing contours from EFD at multiple harmonic orders
  5. Matplotlib 3-panel visualization
"""

import cv2
import numpy as np
import pyefd
from pyefd import normalize_efd
import matplotlib.pyplot as plt
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "contour_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# EFD core helpers
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_contour_efd(contour, order=10, num_points=300):
    """
    Compute EFD for a contour (Nx2 array) and reconstruct it.

    Args:
        contour:    (N, 2) or (N, 1, 2) array of (x, y) points
        order:      number of Fourier harmonics
        num_points: points in the reconstructed contour

    Returns:
        reconstructed: (num_points, 2) ndarray, or None
        coeffs:        (order, 4) EFD coefficient matrix, or None
    """
    coords = np.squeeze(contour)
    if coords.ndim != 2 or len(coords) < 5:
        return None, None

    coeffs = pyefd.elliptic_fourier_descriptors(coords, order=order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(coords)
    reconstructed = pyefd.reconstruct_contour(coeffs, locus=(a0, c0),
                                              num_points=num_points)
    return reconstructed, coeffs


def compute_efd_features(contour, order=10):
    """
    Compute a flat, normalized EFD feature vector (rotation/size-invariant).

    The first 3 values (a1=1, b1=0, c1=0 after normalization) are dropped.

    Returns:
        1-D ndarray of length (order * 4 - 3), or None.
    """
    coords = np.squeeze(contour)
    if coords.ndim != 2 or len(coords) < 5:
        return None
    coeffs = pyefd.elliptic_fourier_descriptors(coords, order=order, normalize=True)
    return coeffs.flatten()[3:]


def _contour_area(contour):
    """Compute area using the shoelace formula (works for any Nx2 array)."""
    c = np.squeeze(contour)
    if c.ndim != 2 or len(c) < 3:
        return 0.0
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _contour_perimeter(contour):
    """Compute perimeter as the sum of point-to-point distances."""
    c = np.squeeze(contour)
    if c.ndim != 2 or len(c) < 2:
        return 0.0
    diffs = np.diff(c, axis=0, append=c[:1])
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def print_efd_summary(significant, total_count, efd_results):
    """
    Print a structured summary of the EFD extraction results.

    Args:
        significant:  list of (index, contour_Nx2) tuples
        total_count:  total number of contours/paths before filtering
        efd_results:  list of result dicts
    """
    print(f"\n{'=' * 60}")
    print(f"  EFD Extraction Summary  –  {len(efd_results)} contour(s) "
          f"(of {total_count} total)")
    print(f"{'=' * 60}")
    for idx, res in enumerate(efd_results):
        i = significant[idx][0]
        cnt = significant[idx][1]
        print(f"\n  Contour {i}:")
        print(f"    Points         : {len(np.squeeze(cnt))}")
        print(f"    Area           : {res['area']:.1f}")
        print(f"    Perimeter      : {res['perimeter']:.1f}")
        print(f"    Coeffs shape   : {res['coeffs'].shape}")
        print(f"    Locus (A0, C0) : ({res['locus'][0]:.2f}, {res['locus'][1]:.2f})")
        if res['features'] is not None:
            print(f"    Feature length : {len(res['features'])}")
        if res['norm_coeffs'] is not None:
            print(f"    Norm coeffs    : {res['norm_coeffs'].shape}")
    print(f"\n{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Raster image contour extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_raster_contours(image_path, min_contour_area=100):
    """
    Load a raster image, denoise, binarize, and extract contours.

    Returns:
        contours:   list of OpenCV contours (Nx1x2 arrays)
        hierarchy:  OpenCV hierarchy array
        img:        original grayscale image
        binary:     the binary image used for detection
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None, None

    # Denoise
    denoised = cv2.fastNlMeansDenoising(
        img, h=10, templateWindowSize=7, searchWindowSize=21
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_denoised.png"), denoised)
    print(f"  ➜  Denoised image saved to {OUTPUT_DIR}/real_denoised.png")

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_binary.png"), cv2.bitwise_not(binary))

    # Find full contour hierarchy (outer + inner contours)
    contours, hierarchy = cv2.findContours(
        binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours, hierarchy, img, binary


# ═══════════════════════════════════════════════════════════════════════════
# Shared EFD processing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _process_contours(significant, total_count, efd_orders, colors):
    """
    Run the shared EFD pipeline on a list of (index, contour_Nx2) tuples.

    Returns:
        efd_results: list of per-contour result dicts
    """
    best_order = efd_orders[-1]
    efd_results = []

    for idx, (i, cnt) in enumerate(significant):
        coords = np.squeeze(cnt)
        if coords.ndim != 2 or len(coords) < 5:
            continue

        raw_coeffs = pyefd.elliptic_fourier_descriptors(
            coords, order=best_order, normalize=False
        )
        norm_coeffs = normalize_efd(raw_coeffs)
        a0, c0 = pyefd.calculate_dc_coefficients(coords)
        features = compute_efd_features(cnt, order=best_order)

        efd_results.append({
            "coeffs": raw_coeffs,
            "norm_coeffs": norm_coeffs,
            "locus": (a0, c0),
            "features": features,
            "area": _contour_area(cnt),
            "perimeter": _contour_perimeter(cnt),
        })

    # Print structured summary
    print_efd_summary(significant, total_count, efd_results)

    # EFD reconstruction at multiple harmonic orders
    for order in efd_orders:
        print(f"\n{'-' * 60}")
        print(f"  EFD Reconstruction  —  order = {order}")
        print(f"{'-' * 60}")

        for idx, (i, cnt) in enumerate(significant):
            recon, coeffs = reconstruct_contour_efd(cnt, order=order)
            if recon is None:
                continue
            print(f"    Contour {i:>2d} → {len(recon)} pts reconstructed "
                  f"({coeffs.shape[0]} harmonics × {coeffs.shape[1]} coeffs)")

    return efd_results


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point — raster only
# ═══════════════════════════════════════════════════════════════════════════

def process_image(image_path, efd_orders=(5, 10, 20, 40), min_contour_area=100):
    """
    Process a raster image through the full EFD pipeline.

    Args:
        image_path:       path to the input raster image (PNG, JPG, BMP, etc.)
        efd_orders:       tuple of harmonic orders to compare
        min_contour_area: minimum contour area
    """
    if image_path.lower().endswith(".svg"):
        raise ValueError("SVG parsing has been removed from EFD. Use a raster image instead.")

    _process_raster(image_path, efd_orders, min_contour_area)


def extract_efd_from_image(image_path, order=10, min_contour_area=100):
    """
    Backward-compatible helper to extract EFDs from a raster image.

    Returns:
        dict with keys: significant, total_count, efd_results
    """
    if image_path.lower().endswith(".svg"):
        raise ValueError("SVG parsing has been removed from EFD. Use a raster image instead.")

    contours, hierarchy, _img, _binary = _extract_raster_contours(
        image_path, min_contour_area
    )
    if contours is None:
        return {"significant": [], "total_count": 0, "efd_results": []}
    significant = [(i, cnt) for i, cnt in enumerate(contours)
                   if cv2.contourArea(cnt) >= min_contour_area]
    total_count = len(contours)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(significant), 1)))
    colors = [cmap[i % len(cmap)] for i in range(len(significant))]
    efd_results = _process_contours(significant, total_count, (order,), colors)
    return {
        "significant": significant,
        "total_count": total_count,
        "efd_results": efd_results,
    }


def visualize_efd(image_path, order=10, save_path=None, min_contour_area=100):
    """
    Backward-compatible visualization wrapper.

    Generates plots/files via the current pipeline and optionally copies the
    Matplotlib overview image to a custom save path.
    """
    process_image(
        image_path,
        efd_orders=(order,),
        min_contour_area=min_contour_area,
    )

    if save_path:
        generated = os.path.join(OUTPUT_DIR, "efd_matplotlib_overview.png")
        if os.path.exists(generated):
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            shutil.copyfile(generated, save_path)


def _process_raster(image_path, efd_orders, min_contour_area):
    """Raster processing pipeline: denoise → threshold → contours → EFD."""
    contours, hierarchy, img, binary = _extract_raster_contours(
        image_path, min_contour_area
    )
    if contours is None:
        print(f"  ⚠  Could not load image: {image_path}")
        return

    # Keep all contours from the RETR_TREE hierarchy, filtered only by area.
    significant = [(i, cnt) for i, cnt in enumerate(contours)
                   if cv2.contourArea(cnt) >= min_contour_area]

    print(f"\n{'=' * 60}")
    print(f"  Raster Image  —  {len(contours)} total contours, "
          f"{len(significant)} with area ≥ {min_contour_area}")
    print(f"{'=' * 60}")

    # Draw RETR_TREE contours on a standalone canvas.
    canvas_orig = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)

    colors_bgr = [tuple(int(c) for c in np.random.randint(60, 255, 3))
                  for _ in significant]

    for idx, (i, cnt) in enumerate(significant):
        cv2.drawContours(canvas_orig, [cnt], -1, colors_bgr[idx], 2)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        x, y, w, h = cv2.boundingRect(cnt)
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]
        print(f"  Contour {i:>2d} | area={area:>8.1f}  perim={perimeter:>7.1f}"
              f"  bbox=({x},{y},{w},{h})"
              f"  hierarchy(N={hier[0]}, P={hier[1]}, C={hier[2]}, Par={hier[3]})")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "contours_real.png"), canvas_orig)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "contours_tree.png"), canvas_orig)

    # Matplotlib-compatible colours
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(significant), 1)))
    colors = [cmap[i % len(cmap)] for i in range(len(significant))]

    # Run shared EFD pipeline
    efd_results = _process_contours(
        significant, len(contours), efd_orders, colors
    )

    # OpenCV-based contour-only EFD reconstruction images
    best_order = efd_orders[-1]
    best_canvas_efd = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    for order in efd_orders:
        canvas_efd = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        for idx, (i, cnt) in enumerate(significant):
            recon, coeffs = reconstruct_contour_efd(cnt, order=order)
            if recon is None:
                continue
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(canvas_efd, [recon_pts], -1, colors_bgr[idx], 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"efd_order_{order}.png"), canvas_efd)
        if order == best_order:
            best_canvas_efd = canvas_efd.copy()

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "efd_reconstructed_contours_only.png"),
        best_canvas_efd
    )

    # Side-by-side comparison on contour-only canvases
    cmp_left = canvas_orig.copy()
    cmp_right = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    for idx, (i, cnt) in enumerate(significant):
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is not None:
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(cmp_right, [recon_pts], -1, colors_bgr[idx], 2)
    cv2.putText(cmp_left, "Detected Tree Contours", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(cmp_right, f"EFD Contours order={best_order}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "efd_comparison.png"),
                np.hstack([cmp_left, cmp_right]))

    # Matplotlib visualization
    _visualize_raster(image_path, significant, colors, best_order)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (raster)
# ═══════════════════════════════════════════════════════════════════════════


def _visualize_raster(image_path, significant, colors, best_order):
    """
    3-panel matplotlib visualization for raster images:
    Panel 1 — Original image
    Panel 2 — RETR_TREE contours filtered by area
    Panel 3 — EFD reconstruction on a standalone contour canvas
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Original
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    h, w = img_bgr.shape[:2]

    # Panel 2 — Detected contours (contour-only)
    contour_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, (i, cnt) in enumerate(significant):
        color_bgr = tuple(int(c * 255) for c in colors[idx][:3])
        cv2.drawContours(contour_vis, [cnt.reshape(-1, 1, 2).astype(np.int32)],
                         -1, color_bgr, 2)
    axes[1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected Tree Contours ({len(significant)})")
    axes[1].axis("off")

    # Panel 3 — EFD reconstruction (contour-only)
    recon_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, (i, cnt) in enumerate(significant):
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is None:
            continue
        color_bgr = tuple(int(c * 255) for c in colors[idx][:3])
        recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(recon_vis, [recon_pts], -1, color_bgr, 2)
    axes[2].imshow(cv2.cvtColor(recon_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"EFD Contours Only (order={best_order})")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "efd_matplotlib_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  ➜  Matplotlib overview saved to {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)

    # Process a raster image
    process_image(
        r"C:\Users\Mahmoud\Documents\GitHub\Sketch_Based_Heritage_Restoration\test_images\sketches.jpg"
    )

    print("\n✅  All tests complete. Check the 'contour_outputs' folder for results.")
