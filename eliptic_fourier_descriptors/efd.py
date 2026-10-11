"""
Elliptic Fourier Descriptor (EFD) Extraction
=============================================
Standalone demo of raster-only EFD extraction and contour reconstruction.
The restoration pipeline does not use this module.

Raster pipeline:  image → denoise → threshold → findContours → EFD

Demonstrates:
  1. Loading & binarizing a raster image
  2. Extracting contours & properties (area, perimeter, bounding box)
  3. Computing EFD coefficients (raw and normalized)
  4. Reconstructing contours from EFD
  5. Matplotlib 3-panel visualization
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyefd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "contour_outputs")

_CONTOUR_RETRIEVAL_MODES = {
    "tree": cv2.RETR_TREE,
    "external": cv2.RETR_EXTERNAL,
}


# ═══════════════════════════════════════════════════════════════════════════
# EFD core helpers
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_contour_efd(contour, order=40, num_points=300):
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


def compute_efd_features(contour, order=40):
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


def _compute_efd_results(significant, order):
    """Per-contour EFD data for a list of (index, contour) tuples."""
    efd_results = []
    for _, cnt in significant:
        coords = np.squeeze(cnt)
        if coords.ndim != 2 or len(coords) < 5:
            continue

        # Translation invariance: subtract centroid
        raw_coeffs = pyefd.elliptic_fourier_descriptors(
            coords - coords.mean(axis=0), order=order, normalize=False
        )
        efd_results.append({
            "coeffs": raw_coeffs,
            "norm_coeffs": pyefd.normalize_efd(raw_coeffs),
            "locus": pyefd.calculate_dc_coefficients(coords),
            "features": compute_efd_features(cnt, order=order),
            "area": _contour_area(cnt),
            "perimeter": _contour_perimeter(cnt),
        })
    return efd_results


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

def _morphological_skeleton(binary):
    """Return a 1-pixel skeleton from a binary uint8 image (0/255)."""
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, residue)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def _extract_raster_contours(image_path, use_skeleton=False, contour_retrieval="tree"):
    """
    Load a raster image, denoise, binarize, and extract contours.

    Returns (contours, hierarchy, binary), or (None, None, None) if the image
    cannot be read. Intermediate images are saved to OUTPUT_DIR.
    """
    retrieval_key = contour_retrieval.lower()
    if retrieval_key not in _CONTOUR_RETRIEVAL_MODES:
        raise ValueError(
            f"contour_retrieval must be one of {tuple(_CONTOUR_RETRIEVAL_MODES.keys())}, "
            f"got {contour_retrieval!r}"
        )

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None

    # Denoise
    denoised = cv2.fastNlMeansDenoising(
        img, h=10, templateWindowSize=7, searchWindowSize=21
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_denoised.png"), denoised)
    print(f"  ->  Denoised image saved to {OUTPUT_DIR}/real_denoised.png")

    # Adaptive threshold, then morphological closing
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_binary.png"), cv2.bitwise_not(binary))

    if use_skeleton:
        binary = _morphological_skeleton(binary)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "real_skeleton.png"), cv2.bitwise_not(binary))
        print(f"  ->  Skeleton image saved to {OUTPUT_DIR}/real_skeleton.png")

    contours, hierarchy = cv2.findContours(
        binary.copy(), _CONTOUR_RETRIEVAL_MODES[retrieval_key], cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy, binary


def _draw_contours(shape, contours, colors):
    """Draw each contour (None entries are skipped) in its colour on a black BGR canvas."""
    canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for cnt, color in zip(contours, colors):
        if cnt is not None:
            cv2.drawContours(canvas, [cnt.astype(np.int32).reshape((-1, 1, 2))], -1, color, 2)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point — raster only
# ═══════════════════════════════════════════════════════════════════════════

def process_image(
    image_path,
    order=40,
    min_contour_area=100,
    use_skeleton=True,
    contour_retrieval="tree",
):
    """
    Process a raster image through the full EFD pipeline:
    denoise → threshold → contours → EFD → reconstruction → visualization.

    Args:
        image_path:        path to the input raster image (PNG, JPG, BMP, etc.)
        order:             number of Fourier harmonics
        min_contour_area:  minimum contour area
        use_skeleton:      skeletonize the binary image before contour extraction
        contour_retrieval: contour retrieval mode: "tree" or "external"

    Returns the list of per-contour EFD result dicts (None if the image cannot be read).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    contours, hierarchy, binary = _extract_raster_contours(
        image_path, use_skeleton=use_skeleton, contour_retrieval=contour_retrieval,
    )
    if contours is None:
        print(f"  !  Could not load image: {image_path}")
        return None

    retrieval_key = contour_retrieval.lower()

    # Keep all contours from the selected hierarchy, filtered only by area.
    significant = [(i, cnt) for i, cnt in enumerate(contours)
                   if cv2.contourArea(cnt) >= min_contour_area]
    raw_contours = [cnt for _, cnt in significant]

    print(f"\n{'=' * 60}")
    print(f"  Raster Image ({retrieval_key})  —  {len(contours)} total contours, "
          f"{len(significant)} with area ≥ {min_contour_area}")
    print(f"{'=' * 60}")

    for i, cnt in significant:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        x, y, w, h = cv2.boundingRect(cnt)
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]
        print(f"  Contour {i:>2d} | area={area:>8.1f}  perim={perimeter:>7.1f}"
              f"  bbox=({x},{y},{w},{h})"
              f"  hierarchy(N={hier[0]}, P={hier[1]}, C={hier[2]}, Par={hier[3]})")

    # Detected contours on a standalone canvas, one random colour each.
    colors_bgr = [tuple(int(c) for c in np.random.randint(60, 255, 3))
                  for _ in significant]
    canvas_orig = _draw_contours(binary.shape, raw_contours, colors_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"contours_{retrieval_key}.png"), canvas_orig)

    efd_results = _compute_efd_results(significant, order)
    print_efd_summary(significant, len(contours), efd_results)

    # EFD reconstruction (None for contours too short to describe)
    print(f"\n{'-' * 60}")
    print(f"  EFD Reconstruction  —  order = {order}")
    print(f"{'-' * 60}")
    reconstructions = []
    for i, cnt in significant:
        recon, coeffs = reconstruct_contour_efd(cnt, order=order)
        reconstructions.append(recon)
        if recon is not None:
            print(f"    Contour {i:>2d} -> {len(recon)} pts reconstructed "
                  f"({coeffs.shape[0]} harmonics × {coeffs.shape[1]} coeffs)")

    canvas_efd = _draw_contours(binary.shape, reconstructions, colors_bgr)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"efd_order_{order}.png"), canvas_efd)

    # Side-by-side comparison on contour-only canvases
    cmp_left = canvas_orig.copy()
    cmp_right = canvas_efd.copy()
    cv2.putText(cmp_left, "Detected Tree Contours", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(cmp_right, f"EFD Contours order={order}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "efd_comparison.png"),
                np.hstack([cmp_left, cmp_right]))

    _visualize_raster(image_path, raw_contours, reconstructions, order)
    return efd_results


# ═══════════════════════════════════════════════════════════════════════════
# Visualization (raster)
# ═══════════════════════════════════════════════════════════════════════════

def _visualize_raster(image_path, contours, reconstructions, order):
    """
    3-panel matplotlib visualization for raster images:
    Panel 1 — Original image
    Panel 2 — Detected contours filtered by area
    Panel 3 — EFD reconstruction on a standalone contour canvas
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    # One matplotlib colour per contour
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(contours), 1)))
    colors = [tuple(int(c * 255) for c in cmap[i % len(cmap)][:3]) for i in range(len(contours))]

    shape = img_bgr.shape[:2]
    panels = [
        ("Original Image", img_bgr),
        (f"Detected Contours ({len(contours)})", _draw_contours(shape, contours, colors)),
        (f"EFD Contours Only (order={order})", _draw_contours(shape, reconstructions, colors)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "efd_matplotlib_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  ->  Matplotlib overview saved to {out_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)

    # Process a raster image
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    process_image(os.path.join(base_dir, "test_images", "sketches.jpg"))

    print("\n[OK]  All tests complete. Check the 'contour_outputs' folder for results.")
