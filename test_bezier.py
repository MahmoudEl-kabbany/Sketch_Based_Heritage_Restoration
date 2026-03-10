"""Quick smoke tests for the bezier module."""

from bezier_curves.bezier import (
    extract_from_svg,
    fit_from_image,
    fit_from_contours,
    export_to_svg,
    export_to_json,
    visualize_paths,
    _print_summary,
)
import os
import numpy as np

OUTPUT = os.path.join("bezier_curves", "outputs")
os.makedirs(OUTPUT, exist_ok=True)

# ── Test 1: SVG extraction ──────────────────────────────────────────────
print("\n▶ Test 1: SVG → Bezier extraction")
svg_file = "test_images/cloud-rain-alt.svg"
if os.path.exists(svg_file):
    paths = extract_from_svg(svg_file)
    _print_summary(paths, f"SVG: {svg_file}")

    out_svg = os.path.join(OUTPUT, "svg_roundtrip.svg")
    export_to_svg(paths, out_svg)
    print(f"  SVG export → {out_svg}")

    out_json = os.path.join(OUTPUT, "svg_roundtrip.json")
    export_to_json(paths, out_json)
    print(f"  JSON export → {out_json}")

    visualize_paths(
        paths, title="SVG Bezier Extraction",
        save_path=os.path.join(OUTPUT, "svg_bezier_vis.png"),
        show_controls=True,
    )
else:
    print(f"  ⚠  {svg_file} not found, skipping.")

# ── Test 2: Raster image → contour → Bezier ────────────────────────────
print("\n▶ Test 2: Raster image → Bezier fitting")
raster_files = ["test_images/sketches2.jpg"]
for raster_file in raster_files:
    if not os.path.exists(raster_file):
        print(f"  ⚠  {raster_file} not found, skipping.")
        continue

    paths = fit_from_image(raster_file, min_contour_area=200, max_error=4.0)
    _print_summary(paths, f"Raster: {raster_file}")

    name = os.path.splitext(os.path.basename(raster_file))[0]
    out_svg = os.path.join(OUTPUT, f"{name}_bezier.svg")
    export_to_svg(paths, out_svg)
    print(f"  SVG export → {out_svg}")

    import cv2
    img = cv2.imread(raster_file, cv2.IMREAD_GRAYSCALE)
    visualize_paths(
        paths,
        title=f"Contour Bezier Fit — {name}",
        background=img,
        save_path=os.path.join(OUTPUT, f"{name}_bezier_vis.png"),
        show_controls=False,
    )
