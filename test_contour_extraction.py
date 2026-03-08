"""
Test OpenCV Contour Extraction from Binary Images
==================================================
Demonstrates:
  1. Creating synthetic binary images (shapes)
  2. Loading & binarizing a real image (if provided)
  3. Finding contours with different retrieval modes
  4. Drawing contours & extracting properties (area, perimeter, bounding box)
  5. Contour hierarchy visualization
  6. Elliptic Fourier Descriptor (EFD) approximation via pyEFD
"""

import cv2
import numpy as np
import pyefd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "contour_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Generate a synthetic binary image with various shapes ──────────────
def create_synthetic_binary_image(width=600, height=400):
    """Create a binary image with rectangles, circles, and nested shapes."""
    img = np.zeros((height, width), dtype=np.uint8)

    # Outer rectangle with an inner rectangle (nested contour)
    cv2.rectangle(img, (30, 30), (200, 180), 255, -1)
    cv2.rectangle(img, (70, 60), (160, 140), 0, -1)  # hole

    # Circle
    cv2.circle(img, (350, 100), 70, 255, -1)

    # Triangle
    pts = np.array([[480, 30], [560, 180], [400, 180]], np.int32)
    cv2.fillPoly(img, [pts], 255)

    # Small dots (to test tiny contours)
    cv2.circle(img, (100, 300), 10, 255, -1)
    cv2.circle(img, (200, 300), 15, 255, -1)
    cv2.circle(img, (300, 300), 20, 255, -1)

    # Nested circles
    cv2.circle(img, (480, 320), 60, 255, -1)
    cv2.circle(img, (480, 320), 35, 0, -1)   # hole
    cv2.circle(img, (480, 320), 15, 255, -1)  # filled center

    return img


# ── 2. Find & draw contours with a given retrieval mode ───────────────────
def extract_and_draw(binary_img, mode, mode_name):
    """
    Find contours using the specified retrieval mode, draw them colour-coded,
    and print properties for each contour.
    """
    # findContours modifies the source in some OpenCV builds, so copy
    contours, hierarchy = cv2.findContours(
        binary_img.copy(), mode, cv2.CHAIN_APPROX_SIMPLE
    )

    # Convert to BGR (inverted: white bg) so we can draw coloured contours
    display_bg = cv2.bitwise_not(binary_img)
    canvas = cv2.cvtColor(display_bg, cv2.COLOR_GRAY2BGR)

    print(f"\n{'=' * 60}")
    print(f"  Retrieval Mode: {mode_name}  —  {len(contours)} contour(s) found")
    print(f"{'=' * 60}")

    for i, cnt in enumerate(contours):
        # Random colour per contour
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.drawContours(canvas, [cnt], -1, color, 2)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        x, y, w, h = cv2.boundingRect(cnt)
        moments = cv2.moments(cnt)

        # Centroid (avoid division by zero)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        # Label the contour on the image
        cv2.putText(canvas, str(i), (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hierarchy info: [Next, Previous, First_Child, Parent]
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]

        print(f"  Contour {i:>2d} | area={area:>8.1f}  perim={perimeter:>7.1f}"
              f"  bbox=({x},{y},{w},{h})  centroid=({cx},{cy})"
              f"  hierarchy(N={hier[0]}, P={hier[1]}, C={hier[2]}, Par={hier[3]})")

    out_path = os.path.join(OUTPUT_DIR, f"contours_{mode_name}.png")
    cv2.imwrite(out_path, canvas)
    print(f"  ➜  Saved to {out_path}")
    return contours, hierarchy


# ── 3. Contour approximation comparison ──────────────────────────────────
def compare_approximations(binary_img):
    """Compare CHAIN_APPROX_NONE vs CHAIN_APPROX_SIMPLE."""
    contours_none, _ = cv2.findContours(
        binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours_simple, _ = cv2.findContours(
        binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    canvas = cv2.cvtColor(cv2.bitwise_not(binary_img), cv2.COLOR_GRAY2BGR)

    total_pts_none = sum(len(c) for c in contours_none)
    total_pts_simple = sum(len(c) for c in contours_simple)

    print(f"\n{'=' * 60}")
    print(f"  Approximation Comparison")
    print(f"{'=' * 60}")
    print(f"  CHAIN_APPROX_NONE   → {total_pts_none:>5d} total points")
    print(f"  CHAIN_APPROX_SIMPLE → {total_pts_simple:>5d} total points")
    print(f"  Reduction ratio     → {total_pts_simple / max(total_pts_none, 1):.2%}")

    # Draw SIMPLE contour points as green dots
    for cnt in contours_simple:
        for pt in cnt:
            cv2.circle(canvas, tuple(pt[0]), 3, (0, 255, 0), -1)

    out_path = os.path.join(OUTPUT_DIR, "approx_comparison.png")
    cv2.imwrite(out_path, canvas)
    print(f"  ➜  Saved to {out_path}")


# ── 4. EFD helper ─────────────────────────────────────────────────────────
def reconstruct_contour_efd(contour, order=10, num_points=300):
    """
    Compute Elliptic Fourier Descriptors for a contour and reconstruct it.

    Args:
        contour:    OpenCV contour (Nx1x2 array)
        order:      number of Fourier harmonics to use
        num_points: number of points in the reconstructed contour

    Returns:
        reconstructed: (num_points, 2) ndarray of (x, y) points
        coeffs:        EFD coefficient array  (order x 4)
    """
    # Flatten OpenCV contour to (N, 2)
    coords = contour.squeeze()
    if coords.ndim != 2 or len(coords) < 5:
        return None, None

    coeffs = pyefd.elliptic_fourier_descriptors(coords, order=order, normalize=False)
    # Compute the DC components (mean offsets)
    a0, c0 = pyefd.calculate_dc_coefficients(coords)
    reconstructed = pyefd.reconstruct_contour(coeffs, locus=(a0, c0),
                                              num_points=num_points)
    return reconstructed, coeffs


# ── 5. Load a real image, extract contours, apply EFD ────────────────────
def process_real_image(image_path, efd_orders=(5, 10, 20, 40), min_contour_area=100):
    """
    Load a real image, denoise, binarize, extract contours, and compute
    Elliptic Fourier Descriptor reconstructions at several harmonic orders.

    Args:
        image_path:       path to the input image
        efd_orders:       tuple of harmonic orders to compare
        min_contour_area: ignore contours smaller than this (reduces noise)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ⚠  Could not load image: {image_path}")
        return

    # Denoise the grayscale image before binarisation
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_denoised.png"), denoised)
    print(f"  ➜  Denoised image saved to {OUTPUT_DIR}/real_denoised.png")

    # Adaptive threshold — black bg / white fg for correct contour detection
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological closing to bridge small gaps / remove noise fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Save the binary image with inverted colours (white bg / black fg)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "real_binary.png"), cv2.bitwise_not(binary))

    # Find only the outermost contours (white objects on black bg)
    contours, hierarchy = cv2.findContours(
        binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # ── Original contours (no labels) ──
    # Use inverted binary (white bg) as canvas so output looks clean
    display_bg = cv2.bitwise_not(binary)
    canvas_orig = cv2.cvtColor(display_bg, cv2.COLOR_GRAY2BGR)

    # Filter by minimum area and keep only significant contours
    significant = [(i, cnt) for i, cnt in enumerate(contours)
                   if cv2.contourArea(cnt) >= min_contour_area]

    print(f"\n{'=' * 60}")
    print(f"  Real Image  —  {len(contours)} total contours, "
          f"{len(significant)} with area ≥ {min_contour_area}")
    print(f"{'=' * 60}")

    # Assign a fixed colour per significant contour
    colors = [tuple(int(c) for c in np.random.randint(60, 255, 3))
              for _ in significant]

    for idx, (i, cnt) in enumerate(significant):
        color = colors[idx]
        cv2.drawContours(canvas_orig, [cnt], -1, color, 2)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        x, y, w, h = cv2.boundingRect(cnt)
        hier = hierarchy[0][i] if hierarchy is not None else [-1, -1, -1, -1]

        print(f"  Contour {i:>2d} | area={area:>8.1f}  perim={perimeter:>7.1f}"
              f"  bbox=({x},{y},{w},{h})"
              f"  hierarchy(N={hier[0]}, P={hier[1]}, C={hier[2]}, Par={hier[3]})")

    out_path = os.path.join(OUTPUT_DIR, "contours_real.png")
    cv2.imwrite(out_path, canvas_orig)
    print(f"  ➜  Saved to {out_path}")

    # ── EFD reconstruction at multiple harmonic orders ──
    for order in efd_orders:
        canvas_efd = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)

        print(f"\n{'-' * 60}")
        print(f"  EFD Reconstruction  —  order = {order}")
        print(f"{'-' * 60}")

        for idx, (i, cnt) in enumerate(significant):
            recon, coeffs = reconstruct_contour_efd(cnt, order=order)
            if recon is None:
                continue

            # Convert reconstructed points to OpenCV contour format
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(canvas_efd, [recon_pts], -1, colors[idx], 2)

            print(f"    Contour {i:>2d} → {len(recon)} pts reconstructed "
                  f"({coeffs.shape[0]} harmonics × {coeffs.shape[1]} coeffs)")

        efd_path = os.path.join(OUTPUT_DIR, f"efd_order_{order}.png")
        cv2.imwrite(efd_path, canvas_efd)
        print(f"  ➜  Saved to {efd_path}")

    # ── Side-by-side comparison: original vs best EFD ──
    best_order = efd_orders[-1]
    canvas_cmp_left = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)
    canvas_cmp_right = cv2.cvtColor(display_bg.copy(), cv2.COLOR_GRAY2BGR)

    for idx, (i, cnt) in enumerate(significant):
        cv2.drawContours(canvas_cmp_left, [cnt], -1, colors[idx], 2)
        recon, _ = reconstruct_contour_efd(cnt, order=best_order)
        if recon is not None:
            recon_pts = recon.astype(np.int32).reshape((-1, 1, 2))
            cv2.drawContours(canvas_cmp_right, [recon_pts], -1, colors[idx], 2)

    # Add labels
    cv2.putText(canvas_cmp_left, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(canvas_cmp_right, f"EFD order={best_order}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    comparison = np.hstack([canvas_cmp_left, canvas_cmp_right])
    cmp_path = os.path.join(OUTPUT_DIR, "efd_comparison.png")
    cv2.imwrite(cmp_path, comparison)
    print(f"\n  ➜  Side-by-side comparison saved to {cmp_path}")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__)

    # Create synthetic test image
    binary_img = create_synthetic_binary_image()
    cv2.imwrite(os.path.join(OUTPUT_DIR, "synthetic_binary.png"), cv2.bitwise_not(binary_img))
    print(f"Synthetic binary image saved to {OUTPUT_DIR}/synthetic_binary.png")

    # Test all four retrieval modes
    modes = [
        (cv2.RETR_EXTERNAL,  "RETR_EXTERNAL"),   # outermost contours only
        (cv2.RETR_LIST,      "RETR_LIST"),        # all contours, flat list
        (cv2.RETR_CCOMP,     "RETR_CCOMP"),       # two-level hierarchy
        (cv2.RETR_TREE,      "RETR_TREE"),        # full hierarchy tree
    ]

    for mode, name in modes:
        extract_and_draw(binary_img, mode, name)

    # Approximation comparison
    compare_approximations(binary_img)

    # If you have a real image, uncomment the line below:
    process_real_image(r"C:\Users\Mahmoud\Documents\GitHub\Sketch_Based_Heritage_Restoration\test_images\davincidataset_preview.png")

    print("\n✅  All tests complete. Check the 'contour_outputs' folder for results.")
