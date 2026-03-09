"""
Elliptic Fourier Descriptor (EFD) Extraction
=============================================
Supports both raster images (PNG, JPG, BMP) and vector images (SVG).

Raster pipeline:  image → denoise → threshold → findContours → EFD
SVG pipeline:     SVG → svgpathtools parses paths → sample (x,y) points → EFD

Demonstrates:
  1. Loading & binarizing an image, or parsing SVG paths directly
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
# SVG path extraction  (Approach 2 — direct parsing, no rasterization)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_svg_contours(svg_path, num_samples=300, min_points=5):
    """
    Parse an SVG file and extract contour arrays directly from path data.

    Uses svgpathtools to read SVG <path>, <line>, <polyline>, <polygon>,
    and <rect> elements, then samples evenly-spaced points along each
    continuous subpath.

    Args:
        svg_path:    path to the SVG file
        num_samples: number of points to sample per subpath
        min_points:  discard subpaths with fewer points

    Returns:
        contours: list of (N, 2) numpy arrays
        svg_attributes: dict of SVG-level attributes (width, height, viewBox)
    """
    from svgpathtools import svg2paths2

    paths, attributes, svg_attributes = svg2paths2(svg_path)

    contours = []
    for path in paths:
        # Split into continuous subpaths (handles M commands / disconnected segments)
        subpaths = path.continuous_subpaths()
        for subpath in subpaths:
            if len(subpath) == 0:
                continue

            # Sample points along the subpath
            points = []
            for t in np.linspace(0, 1, num_samples, endpoint=False):
                try:
                    pt = subpath.point(t)
                    points.append([pt.real, pt.imag])
                except Exception:
                    continue

            if len(points) >= min_points:
                contours.append(np.array(points, dtype=np.float64))

    return contours, svg_attributes


def _get_svg_dimensions(svg_attributes):
    """Extract canvas width and height from SVG attributes."""
    width, height = 800, 800  # defaults

    # Try viewBox first
    viewbox = svg_attributes.get("viewBox", "")
    if viewbox:
        parts = viewbox.replace(",", " ").split()
        if len(parts) == 4:
            try:
                width = float(parts[2])
                height = float(parts[3])
                return width, height
            except ValueError:
                pass

    # Fall back to width/height attributes
    for attr in ["width", "height"]:
        val = svg_attributes.get(attr, "")
        val = val.replace("px", "").replace("pt", "").strip()
        try:
            if attr == "width":
                width = float(val)
            else:
                height = float(val)
        except ValueError:
            pass

    return width, height


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

    # Find contours
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
# Main entry point — handles both raster and SVG
# ═══════════════════════════════════════════════════════════════════════════

def process_image(image_path, efd_orders=(5, 10, 20, 40), min_contour_area=100):
    """
    Process a raster image or SVG file through the full EFD pipeline.

    For raster images: denoise → threshold → findContours → EFD
    For SVG files:     parse paths → sample points → EFD (no rasterization)

    Args:
        image_path:       path to the input image (PNG, JPG, SVG, etc.)
        efd_orders:       tuple of harmonic orders to compare
        min_contour_area: minimum contour area (raster only)
    """
    is_svg = image_path.lower().endswith(".svg")

    if is_svg:
        _process_svg(image_path, efd_orders)
    else:
        _process_raster(image_path, efd_orders, min_contour_area)


def _process_svg(image_path, efd_orders):
    """SVG processing pipeline: parse paths → EFD → visualize."""
    print(f"\n{'=' * 60}")
    print(f"  SVG Mode  —  Parsing paths directly (no rasterization)")
    print(f"{'=' * 60}")

    try:
        contours, svg_attrs = _extract_svg_contours(image_path)
    except ImportError:
        print("  ⚠  svgpathtools is required for SVG support.")
        print("     Install with:  pip install svgpathtools")
        return
    except Exception as e:
        print(f"  ⚠  Failed to parse SVG: {e}")
        return

    if not contours:
        print("  ⚠  No paths found in the SVG file.")
        return

    print(f"  Found {len(contours)} subpath(s)")
    for i, c in enumerate(contours):
        print(f"    Subpath {i}: {len(c)} points, "
              f"area={_contour_area(c):.1f}, perim={_contour_perimeter(c):.1f}")

    # Build significant list (all SVG contours are kept — no noise filtering)
    significant = [(i, cnt) for i, cnt in enumerate(contours)]

    # Assign colours
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(significant), 1)))
    colors = [cmap[i % len(cmap)] for i in range(len(significant))]

    # Run shared EFD pipeline
    efd_results = _process_contours(
        significant, len(contours), efd_orders, colors
    )

    # Visualize
    _visualize_svg(image_path, contours, significant, colors,
                   efd_orders, svg_attrs)


def _process_raster(image_path, efd_orders, min_contour_area):
    """Raster processing pipeline: denoise → threshold → contours → EFD."""
    contours, hierarchy, img, binary = _extract_raster_contours(
        image_path, min_contour_area
    )
    if contours is None:
        print(f"  ⚠  Could not load image: {image_path}")
        return

    # Filter by minimum area
    significant = [(i, cnt) for i, cnt in enumerate(contours)
                   if cv2.contourArea(cnt) >= min_contour_area]

    print(f"\n{'=' * 60}")
    print(f"  Raster Image  —  {len(contours)} total contours, "
          f"{len(significant)} with area ≥ {min_contour_area}")
    print(f"{'=' * 60}")

    # Draw original contours
    display_bg = cv2.bitwise_not(binary)
    canvas_orig = cv2.cvtColor(display_bg, cv2.COLOR_GRAY2BGR)

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

    # Matplotlib-compatible colours
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(significant), 1)))
    colors = [cmap[i % len(cmap)] for i in range(len(significant))]

    # Run shared EFD pipeline
    efd_results = _process_contours(
        significant, len(contours), efd_orders, colors
    )

    # OpenCV-based EFD reconstruction images
    for order in efd_orders:
        canvas_efd = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)
        for idx, (i, cnt) in enumerate(significant):
            recon, coeffs = reconstruct_contour_efd(cnt, order=order)
            if recon is None:
                continue
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(canvas_efd, [recon_pts], -1, colors_bgr[idx], 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"efd_order_{order}.png"), canvas_efd)

    # Side-by-side comparison
    best_order = efd_orders[-1]
    cmp_left = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)
    cmp_right = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)
    for idx, (i, cnt) in enumerate(significant):
        cv2.drawContours(cmp_left, [cnt], -1, colors_bgr[idx], 2)
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is not None:
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(cmp_right, [recon_pts], -1, colors_bgr[idx], 2)
    cv2.putText(cmp_left, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(cmp_right, f"EFD order={best_order}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "efd_comparison.png"),
                np.hstack([cmp_left, cmp_right]))

    # Matplotlib visualization
    _visualize_raster(image_path, significant, colors, best_order, display_bg)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def _visualize_svg(image_path, contours, significant, colors, efd_orders, svg_attrs):
    """
    3-panel matplotlib visualization for SVG files:
      Panel 1 — Raw SVG paths (plotted from parsed coordinates)
      Panel 2 — EFD reconstruction at lowest order  (coarse)
      Panel 3 — EFD reconstruction at highest order (detailed)
    """
    best_order = efd_orders[-1]
    worst_order = efd_orders[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"SVG EFD Analysis: {os.path.basename(image_path)}", fontsize=14)

    # Panel 1 — Raw SVG paths
    axes[0].set_title("Original SVG Paths")
    for idx, (i, cnt) in enumerate(significant):
        c = np.squeeze(cnt)
        # Close the path for display
        closed = np.vstack([c, c[:1]])
        axes[0].plot(closed[:, 0], closed[:, 1],
                     color=colors[idx % len(colors)], linewidth=1.5,
                     label=f"Path {i} ({len(c)} pts)")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)

    # Panel 2 — EFD at lowest order
    axes[1].set_title(f"EFD Reconstruction (order={worst_order})")
    for idx, (i, cnt) in enumerate(significant):
        recon, _ = reconstruct_contour_efd(cnt, order=worst_order)
        if recon is None:
            continue
        axes[1].plot(recon[:, 0], recon[:, 1],
                     color=colors[idx % len(colors)], linewidth=1.5,
                     label=f"Path {i}")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)

    # Panel 3 — EFD at highest order
    axes[2].set_title(f"EFD Reconstruction (order={best_order})")
    for idx, (i, cnt) in enumerate(significant):
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is None:
            continue
        area = _contour_area(cnt)
        axes[2].plot(recon[:, 0], recon[:, 1],
                     color=colors[idx % len(colors)], linewidth=1.5,
                     label=f"Path {i} (area={area:.0f})")
    axes[2].set_aspect("equal")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "efd_matplotlib_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  ➜  Matplotlib overview saved to {out_path}")
    plt.show()


def _visualize_raster(image_path, significant, colors, best_order, display_bg):
    """
    3-panel matplotlib visualization for raster images:
      Panel 1 — Original image
      Panel 2 — Detected contours on binary
      Panel 3 — EFD reconstruction at best order
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Original
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2 — Detected contours
    contour_vis = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)
    for idx, (i, cnt) in enumerate(significant):
        color_bgr = tuple(int(c * 255) for c in colors[idx][:3])
        cv2.drawContours(contour_vis, [cnt.reshape(-1, 1, 2).astype(np.int32)],
                         -1, color_bgr, 2)
    axes[1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Detected Contours ({len(significant)})")
    axes[1].axis("off")

    # Panel 3 — EFD reconstruction
    axes[2].set_title(f"EFD Reconstruction (order={best_order})")
    for idx, (i, cnt) in enumerate(significant):
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is None:
            continue
        area = _contour_area(cnt)
        axes[2].plot(recon[:, 0], recon[:, 1],
                     color=colors[idx % len(colors)], linewidth=1.5,
                     label=f"Contour {i} (area={area:.0f})")
    axes[2].set_aspect("equal")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=8)

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

    # Process an image (works with both raster and SVG)
    process_image(
        r"C:\Users\Mahmoud\Documents\GitHub\Sketch_Based_Heritage_Restoration\test_images\cloud-rain-alt.svg"
    )

    print("\n✅  All tests complete. Check the 'contour_outputs' folder for results.")
