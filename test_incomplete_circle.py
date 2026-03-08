"""
Test: Complete an Incomplete Circle using Elliptic Fourier Descriptors
======================================================================
1. Draw a 1px incomplete circle (arc with a gap)
2. Extract the ordered arc centerline points
3. Compute EFD on those points
4. Reconstruct → EFD produces a CLOSED curve that fills the gap
5. Draw original + completed circle with the infilled part highlighted
"""

import cv2
import numpy as np
import pyefd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "contour_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters ──
CENTER = (200, 200)
RADIUS = 120
START_DEG = 30      # arc starts here
END_DEG = 330       # arc ends here → 60° gap from 330° to 30°
IMG_SIZE = 400


def generate_arc_points(center, radius, start_deg, end_deg, num_points=500):
    """
    Generate ordered (x, y) points along an arc.
    These are the TRUE centerline points — not a contour outline.
    """
    angles = np.linspace(np.radians(start_deg), np.radians(end_deg), num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


def draw_arc_image(center, radius, start_deg, end_deg, size=400):
    """Draw the incomplete circle on a black image."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(img, center, (radius, radius), 0,
                start_deg, end_deg, 255, thickness=1)
    return img


def main():
    # ── 1. Draw the incomplete circle ──
    arc_img = draw_arc_image(CENTER, RADIUS, START_DEG, END_DEG, IMG_SIZE)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "incomplete_circle.png"), arc_img)
    print(f"Drew incomplete circle: {START_DEG}°–{END_DEG}° "
          f"({END_DEG - START_DEG}° arc, {360 - (END_DEG - START_DEG)}° gap)")

    # ── 2. Get ordered arc centerline points ──
    arc_points = generate_arc_points(CENTER, RADIUS, START_DEG, END_DEG,
                                     num_points=500)
    print(f"Arc centerline: {len(arc_points)} ordered points")

    # ── 3. Compute EFD on the arc points ──
    orders = [5, 10, 20, 40]

    for order in orders:
        coeffs = pyefd.elliptic_fourier_descriptors(
            arc_points, order=order, normalize=False)
        a0, c0 = pyefd.calculate_dc_coefficients(arc_points)

        # ── 4. Reconstruct — EFD always produces a CLOSED curve ──
        recon = pyefd.reconstruct_contour(
            coeffs, locus=(a0, c0), num_points=600)
        recon_int = recon.astype(np.int32)

        # ── 5. Classify points: original arc vs infilled gap ──
        # Check each reconstructed point's distance to the original arc
        is_infill = []
        for pt in recon_int:
            # Angular position relative to center
            angle = np.degrees(np.arctan2(pt[1] - CENTER[1],
                                          pt[0] - CENTER[0])) % 360
            # Point is "infilled" if its angle falls in the gap
            # Gap: from END_DEG (330°) to START_DEG (30°), wrapping around 360°
            if END_DEG <= 360 and START_DEG < END_DEG:
                # Gap is from END_DEG → 360 → START_DEG
                in_gap = angle >= END_DEG or angle <= START_DEG
            else:
                in_gap = END_DEG <= angle <= START_DEG
            is_infill.append(in_gap)

        is_infill = np.array(is_infill)

        # ── Draw the result ──
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Draw reconstructed polyline
        for i in range(len(recon_int)):
            j = (i + 1) % len(recon_int)
            pt1 = tuple(recon_int[i])
            pt2 = tuple(recon_int[j])

            # Both points determine line colour
            if is_infill[i] or is_infill[j]:
                color = (0, 255, 255)   # cyan = infilled/completed
            else:
                color = (0, 255, 0)     # green = original arc

            cv2.line(canvas, pt1, pt2, color, 1)

        # Label
        cv2.putText(canvas, f"EFD order={order}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, "green=original  cyan=completed", (10, IMG_SIZE - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        path = os.path.join(OUTPUT_DIR, f"circle_completed_order_{order}.png")
        cv2.imwrite(path, canvas)

        n_infill = int(np.sum(is_infill))
        print(f"  Order {order:>2d}: {len(recon_int)} pts, "
              f"{n_infill} infilled ({n_infill/len(recon_int):.0%} of circumference)")

    # ── Side-by-side comparison at best order ──
    best_order = orders[-1]
    coeffs = pyefd.elliptic_fourier_descriptors(arc_points, order=best_order, normalize=False)
    a0, c0 = pyefd.calculate_dc_coefficients(arc_points)
    recon = pyefd.reconstruct_contour(coeffs, locus=(a0, c0), num_points=600)
    recon_int = recon.astype(np.int32)

    # Left: original incomplete circle
    left = cv2.cvtColor(arc_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(left, "Incomplete Circle", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Right: completed circle
    right = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Draw original arc in green
    cv2.ellipse(right, CENTER, (RADIUS, RADIUS), 0,
                START_DEG, END_DEG, (0, 255, 0), 1)
    # Draw the full EFD reconstruction — only the gap part in cyan
    for i in range(len(recon_int)):
        j = (i + 1) % len(recon_int)
        pt1 = tuple(recon_int[i])
        pt2 = tuple(recon_int[j])
        angle_i = np.degrees(np.arctan2(
            recon_int[i][1] - CENTER[1], recon_int[i][0] - CENTER[0])) % 360
        in_gap = angle_i >= END_DEG or angle_i <= START_DEG
        if in_gap:
            cv2.line(right, pt1, pt2, (0, 255, 255), 1)

    cv2.putText(right, f"Completed (order={best_order})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(right, "green=original  cyan=EFD infill", (10, IMG_SIZE - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    comparison = np.hstack([left, right])
    cv2.imwrite(os.path.join(OUTPUT_DIR, "circle_completion_comparison.png"), comparison)

    print(f"\n✅  Done! Check {OUTPUT_DIR}/")
    print(f"    • circle_completed_order_*.png  (each EFD order)")
    print(f"    • circle_completion_comparison.png  (side-by-side)")


if __name__ == "__main__":
    main()
