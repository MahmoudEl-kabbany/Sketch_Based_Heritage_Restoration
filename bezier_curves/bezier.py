"""
Bezier Curve Extraction Module
==============================
Two extraction paths:
  1. SVG  → parse raw path segments → cubic Bezier objects
  2. Contours (OpenCV Nx1x2 arrays) → corner detection → Schneider fit → cubic Bezier

All segments are normalized to cubic Bezier for uniformity.

Output:
  - Python objects  (BezierSegment / BezierPath with control points)
  - SVG file export (via svgpathtools)
  - JSON export     (control-point arrays)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import bezier as _bezier_lib
import cv2
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BezierSegment:
    """A single cubic Bezier segment (4 control points)."""

    control_points: np.ndarray  # shape (4, 2) — P0, P1, P2, P3
    source_type: str = "unknown"  # "svg" | "contour"

    # lazily-built bezier.Curve handle
    _curve: object = field(default=None, repr=False, compare=False)

    @property
    def curve(self) -> _bezier_lib.Curve:
        """Return a ``bezier.Curve`` wrapping the control points."""
        if self._curve is None:
            # bezier lib expects shape (2, 4) — rows = dimensions
            nodes = self.control_points.T.astype(np.float64)
            self._curve = _bezier_lib.Curve.from_nodes(nodes)
        return self._curve

    @property
    def start(self) -> np.ndarray:
        return self.control_points[0]

    @property
    def end(self) -> np.ndarray:
        return self.control_points[3]

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the curve at parameter *t* ∈ [0, 1]."""
        return self.curve.evaluate(t).flatten()

    def sample(self, n: int = 50) -> np.ndarray:
        """Return *n* evenly-spaced (x, y) points along the curve."""
        ts = np.linspace(0.0, 1.0, n)
        pts = self.curve.evaluate_multi(ts)  # shape (2, n)
        return pts.T  # (n, 2)


@dataclass
class BezierPath:
    """An ordered sequence of cubic Bezier segments forming a path."""

    segments: List[BezierSegment] = field(default_factory=list)
    is_closed: bool = False
    source_type: str = "unknown"

    @property
    def control_points(self) -> np.ndarray:
        """All control points concatenated — shape (4*N, 2)."""
        if not self.segments:
            return np.empty((0, 2))
        return np.vstack([s.control_points for s in self.segments])

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def sample(self, pts_per_segment: int = 50) -> np.ndarray:
        """Sample all segments and return a single (M, 2) point array."""
        if not self.segments:
            return np.empty((0, 2))
        parts = [s.sample(pts_per_segment) for s in self.segments]
        return np.vstack(parts)


# ═══════════════════════════════════════════════════════════════════════════
# SVG → Bezier  (Phase 1)
# ═══════════════════════════════════════════════════════════════════════════

def _complex_to_xy(c: complex) -> np.ndarray:
    """Convert a complex point to [x, y]."""
    return np.array([c.real, c.imag], dtype=np.float64)


def _line_to_cubic(start: complex, end: complex) -> np.ndarray:
    """Convert an SVG Line to a degenerate cubic (4 control points)."""
    p0 = _complex_to_xy(start)
    p3 = _complex_to_xy(end)
    p1 = p0 + (p3 - p0) / 3.0
    p2 = p0 + 2.0 * (p3 - p0) / 3.0
    return np.vstack([p0, p1, p2, p3])


def _quad_to_cubic(start: complex, control: complex, end: complex) -> np.ndarray:
    """Degree-elevate a quadratic Bezier to cubic (4 control points)."""
    p0 = _complex_to_xy(start)
    q1 = _complex_to_xy(control)
    p3 = _complex_to_xy(end)
    p1 = p0 + (2.0 / 3.0) * (q1 - p0)
    p2 = p3 + (2.0 / 3.0) * (q1 - p3)
    return np.vstack([p0, p1, p2, p3])


def _cubic_controls(seg) -> np.ndarray:
    """Extract control points from an svgpathtools CubicBezier."""
    return np.vstack([
        _complex_to_xy(seg.start),
        _complex_to_xy(seg.control1),
        _complex_to_xy(seg.control2),
        _complex_to_xy(seg.end),
    ])


def _arc_to_cubics(arc_seg) -> List[np.ndarray]:
    """Convert an svgpathtools Arc to one or more cubic Bezier control-point arrays."""
    cubics = arc_seg.as_cubic_curves()
    return [_cubic_controls(c) for c in cubics]


class SVGBezierExtractor:
    """Extract cubic Bezier paths from an SVG file.

    Uses svgpathtools to parse all ``<path>`` elements and converts every
    segment (Line, QuadraticBezier, CubicBezier, Arc) to cubic Bezier.
    """

    def __init__(self, svg_path: str):
        self.svg_path = svg_path
        self.paths: List[BezierPath] = []
        self.svg_attributes: dict = {}

    def extract(self) -> List[BezierPath]:
        from svgpathtools import (
            svg2paths2, Line, QuadraticBezier, CubicBezier, Arc,
        )

        paths, attributes, self.svg_attributes = svg2paths2(self.svg_path)

        self.paths = []
        for path in paths:
            for subpath in path.continuous_subpaths():
                if len(subpath) == 0:
                    continue
                segments: List[BezierSegment] = []
                for seg in subpath:
                    if isinstance(seg, CubicBezier):
                        cps = _cubic_controls(seg)
                        segments.append(BezierSegment(cps, source_type="svg"))
                    elif isinstance(seg, QuadraticBezier):
                        cps = _quad_to_cubic(seg.start, seg.control, seg.end)
                        segments.append(BezierSegment(cps, source_type="svg"))
                    elif isinstance(seg, Line):
                        cps = _line_to_cubic(seg.start, seg.end)
                        segments.append(BezierSegment(cps, source_type="svg"))
                    elif isinstance(seg, Arc):
                        for cps in _arc_to_cubics(seg):
                            segments.append(BezierSegment(cps, source_type="svg"))

                if segments:
                    # Detect closure: does the last endpoint ≈ first start?
                    closed = np.allclose(
                        segments[-1].end, segments[0].start, atol=1e-3
                    )
                    self.paths.append(
                        BezierPath(segments, is_closed=closed, source_type="svg")
                    )

        return self.paths


# ═══════════════════════════════════════════════════════════════════════════
# Schneider's cubic-Bezier fitting algorithm
# (adapted from the Graphics Gems / "An Algorithm for Automatically
#  Fitting Digitized Curves" — Philip J. Schneider, 1990)
# ═══════════════════════════════════════════════════════════════════════════

def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    """Assign parameter values by cumulative chord length."""
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumlen = np.concatenate(([0.0], np.cumsum(diffs)))
    total = cumlen[-1]
    if total < 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return cumlen / total


def _estimate_tangent(points: np.ndarray, end: str) -> np.ndarray:
    """Estimate unit tangent at the start or end of a point sequence."""
    if end == "start":
        tangent = points[1] - points[0]
    else:
        tangent = points[-1] - points[-2]
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([1.0, 0.0])
    return tangent / norm


def _bezier_point(cp: np.ndarray, t: float) -> np.ndarray:
    """De Casteljau evaluation of a cubic Bezier at parameter *t*."""
    u = 1.0 - t
    return (u ** 3) * cp[0] + 3 * (u ** 2) * t * cp[1] + 3 * u * (t ** 2) * cp[2] + (t ** 3) * cp[3]


def _generate_bezier(
    points: np.ndarray,
    params: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
) -> np.ndarray:
    """Solve for the best-fit cubic Bezier control points (Schneider §4)."""
    n = len(points)
    p0 = points[0]
    p3 = points[-1]

    # Build the A matrix (per-point contribution of the two free tangents)
    A = np.zeros((n, 2, 2))
    for i, t in enumerate(params):
        u = 1.0 - t
        A[i, 0] = 3.0 * u * u * t * left_tangent
        A[i, 1] = 3.0 * u * t * t * right_tangent

    # Build C and X matrices for the 2×2 least-squares system
    C = np.zeros((2, 2))
    X = np.zeros(2)
    for i in range(n):
        C[0, 0] += np.dot(A[i, 0], A[i, 0])
        C[0, 1] += np.dot(A[i, 0], A[i, 1])
        C[1, 0] = C[0, 1]
        C[1, 1] += np.dot(A[i, 1], A[i, 1])

        u = 1.0 - params[i]
        t = params[i]
        tmp = (
            points[i]
            - (u ** 3) * p0
            - 3 * (u ** 2) * t * p0
            - 3 * u * (t ** 2) * p3
            - (t ** 3) * p3
        )
        X[0] += np.dot(A[i, 0], tmp)
        X[1] += np.dot(A[i, 1], tmp)

    det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
    if abs(det) < 1e-12:
        dist = np.linalg.norm(p3 - p0) / 3.0
        alpha_l = alpha_r = dist
    else:
        alpha_l = (C[1, 1] * X[0] - C[0, 1] * X[1]) / det
        alpha_r = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det

    seg_len = np.linalg.norm(p3 - p0)
    epsilon = 1e-6 * seg_len
    if alpha_l < epsilon or alpha_r < epsilon:
        dist = seg_len / 3.0
        alpha_l = dist
        alpha_r = dist

    p1 = p0 + alpha_l * left_tangent
    p2 = p3 + alpha_r * right_tangent
    return np.vstack([p0, p1, p2, p3])


def _reparameterize(
    points: np.ndarray, params: np.ndarray, cp: np.ndarray
) -> np.ndarray:
    """Newton-Raphson reparameterization to improve parameter values."""
    new_params = params.copy()
    for i in range(len(points)):
        t = params[i]
        u = 1.0 - t
        # Bezier and first two derivatives
        q = _bezier_point(cp, t)
        q1 = 3.0 * (u * u * (cp[1] - cp[0]) + 2 * u * t * (cp[2] - cp[1]) + t * t * (cp[3] - cp[2]))
        q2 = 6.0 * (u * (cp[2] - 2 * cp[1] + cp[0]) + t * (cp[3] - 2 * cp[2] + cp[1]))

        diff = q - points[i]
        num = np.dot(diff, q1)
        den = np.dot(q1, q1) + np.dot(diff, q2)
        if abs(den) > 1e-12:
            new_params[i] = t - num / den
        new_params[i] = np.clip(new_params[i], 0.0, 1.0)
    return new_params


def _max_error(
    points: np.ndarray, cp: np.ndarray, params: np.ndarray
) -> Tuple[float, int]:
    """Return (max squared error, index of worst point)."""
    max_err = 0.0
    split_idx = len(points) // 2
    for i in range(len(points)):
        err = np.sum((_bezier_point(cp, params[i]) - points[i]) ** 2)
        if err > max_err:
            max_err = err
            split_idx = i
    return max_err, split_idx


def _fit_cubic_single(
    points: np.ndarray,
    left_tangent: np.ndarray,
    right_tangent: np.ndarray,
    max_error: float,
    max_iterations: int = 4,
) -> List[np.ndarray]:
    """Fit a set of points with one or more cubic Bezier curves (recursive).

    Returns a list of (4, 2) control-point arrays.
    """
    if len(points) == 2:
        dist = np.linalg.norm(points[1] - points[0]) / 3.0
        cp = np.vstack([
            points[0],
            points[0] + dist * left_tangent,
            points[1] + dist * right_tangent,
            points[1],
        ])
        return [cp]

    params = _chord_length_parameterize(points)
    cp = _generate_bezier(points, params, left_tangent, right_tangent)
    err, split_idx = _max_error(points, cp, params)

    if err < max_error:
        return [cp]

    # Try iterative reparameterization
    if err < max_error * 4.0:
        for _ in range(max_iterations):
            params = _reparameterize(points, params, cp)
            cp = _generate_bezier(points, params, left_tangent, right_tangent)
            err, split_idx = _max_error(points, cp, params)
            if err < max_error:
                return [cp]

    # Split at the point of maximum error and recurse
    split_idx = max(1, min(split_idx, len(points) - 2))
    center_tangent = points[split_idx + 1] - points[split_idx - 1]
    norm = np.linalg.norm(center_tangent)
    if norm < 1e-12:
        center_tangent = np.array([1.0, 0.0])
    else:
        center_tangent = center_tangent / norm

    left_curves = _fit_cubic_single(
        points[: split_idx + 1], left_tangent, center_tangent, max_error
    )
    right_curves = _fit_cubic_single(
        points[split_idx:], -center_tangent, right_tangent, max_error
    )
    return left_curves + right_curves


# ═══════════════════════════════════════════════════════════════════════════
# Contour → Bezier  (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class ContourBezierFitter:
    """Fit cubic Bezier curves to OpenCV contours.

    Pipeline:
      1. Reshape contour to (N, 2)
      2. Simplify with RDP (``cv2.approxPolyDP``) to find corners
      3. Split contour at corners
      4. Fit each sub-segment with Schneider's algorithm
    """

    def __init__(
        self,
        corner_threshold: float = 2.0,
        max_error: float = 5.0,
    ):
        self.corner_threshold = corner_threshold
        self.max_error_sq = max_error ** 2  # Schneider uses squared error

    def fit(self, contours: list) -> List[BezierPath]:
        """Fit Bezier paths to a list of OpenCV contours (Nx1x2 arrays)."""
        paths: List[BezierPath] = []
        for contour in contours:
            path = self._fit_single_contour(contour)
            if path is not None:
                paths.append(path)
        return paths

    def _fit_single_contour(self, contour: np.ndarray) -> Optional[BezierPath]:
        points = np.squeeze(contour).astype(np.float64)
        if points.ndim != 2 or len(points) < 2:
            return None

        # Detect if contour is closed
        is_closed = np.allclose(points[0], points[-1], atol=1.0)
        if is_closed and len(points) > 2:
            points = points[:-1]  # remove duplicate closing point

        # Find corner indices via RDP simplification
        corner_indices = self._find_corners(points, is_closed)

        # Split into segments at the corners and fit each
        segments: List[BezierSegment] = []
        n = len(corner_indices)

        if n < 2:
            # No meaningful corners — fit the entire contour as one
            cps_list = self._fit_segment(points, is_closed)
            for cps in cps_list:
                segments.append(BezierSegment(cps, source_type="contour"))
        else:
            for i in range(n - 1):
                start_idx = corner_indices[i]
                end_idx = corner_indices[i + 1]
                seg_points = points[start_idx: end_idx + 1]
                if len(seg_points) < 2:
                    continue
                cps_list = self._fit_segment(seg_points, closed=False)
                for cps in cps_list:
                    segments.append(BezierSegment(cps, source_type="contour"))

            # Close the loop if needed
            if is_closed and n >= 2:
                start_idx = corner_indices[-1]
                end_idx = corner_indices[0]
                seg_points = np.vstack([points[start_idx:], points[: end_idx + 1]])
                if len(seg_points) >= 2:
                    cps_list = self._fit_segment(seg_points, closed=False)
                    for cps in cps_list:
                        segments.append(BezierSegment(cps, source_type="contour"))

        if not segments:
            return None
        return BezierPath(segments, is_closed=is_closed, source_type="contour")

    def _find_corners(self, points: np.ndarray, is_closed: bool) -> List[int]:
        """Use RDP to find corner indices in the contour."""
        # approxPolyDP needs (N, 1, 2)
        contour_cv = points.reshape(-1, 1, 2).astype(np.float32)
        approx = cv2.approxPolyDP(contour_cv, self.corner_threshold, is_closed)
        approx_pts = np.squeeze(approx).astype(np.float64)
        if approx_pts.ndim == 1:
            approx_pts = approx_pts.reshape(1, -1)

        # Map each approximate vertex back to the nearest original point index
        corner_indices = []
        for pt in approx_pts:
            dists = np.linalg.norm(points - pt, axis=1)
            corner_indices.append(int(np.argmin(dists)))

        # Sort and deduplicate
        corner_indices = sorted(set(corner_indices))
        return corner_indices

    def _fit_segment(
        self, points: np.ndarray, closed: bool = False
    ) -> List[np.ndarray]:
        """Fit cubic Beziers to *points* using Schneider's method."""
        if len(points) < 2:
            return []

        left_tangent = _estimate_tangent(points, "start")
        right_tangent = _estimate_tangent(points, "end")

        if closed and len(points) > 2:
            # For closed segments make tangents consistent at junction
            wrap_tangent = points[1] - points[-2]
            norm = np.linalg.norm(wrap_tangent)
            if norm > 1e-12:
                wrap_tangent = wrap_tangent / norm
                left_tangent = wrap_tangent
                right_tangent = -wrap_tangent

        return _fit_cubic_single(
            points, left_tangent, right_tangent, self.max_error_sq
        )


# ═══════════════════════════════════════════════════════════════════════════
# Raster image → contour extraction  (mirrors efd.py approach)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_raster_contours(
    image_path: str, min_contour_area: float = 100.0
) -> Tuple[list, Optional[np.ndarray]]:
    """Load a raster image, binarize, and return OpenCV contours."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Filter by area
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    return contours, img


# ═══════════════════════════════════════════════════════════════════════════
# Export — SVG
# ═══════════════════════════════════════════════════════════════════════════

def export_to_svg(
    paths: List[BezierPath],
    output_path: str,
    width: float = 800,
    height: float = 800,
    stroke: str = "black",
    stroke_width: float = 1.5,
) -> str:
    """Write Bezier paths to an SVG file.

    Returns the output file path.
    """
    from svgpathtools import CubicBezier as SvgCubic, Path as SvgPath, wsvg

    svg_paths = []
    svg_attrs = []
    for bp in paths:
        segs = []
        for s in bp.segments:
            cp = s.control_points
            segs.append(SvgCubic(
                complex(cp[0, 0], cp[0, 1]),
                complex(cp[1, 0], cp[1, 1]),
                complex(cp[2, 0], cp[2, 1]),
                complex(cp[3, 0], cp[3, 1]),
            ))
        if segs:
            svg_paths.append(SvgPath(*segs))
            svg_attrs.append({
                "fill": "none",
                "stroke": stroke,
                "stroke-width": str(stroke_width),
            })

    wsvg(
        svg_paths,
        attributes=svg_attrs,
        svg_attributes={"width": str(width), "height": str(height)},
        filename=output_path,
    )
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Export — JSON
# ═══════════════════════════════════════════════════════════════════════════

def export_to_json(paths: List[BezierPath], output_path: str) -> str:
    """Serialize Bezier paths to a JSON file. Returns the output file path."""
    data = []
    for bp in paths:
        path_data = {
            "is_closed": bp.is_closed,
            "source_type": bp.source_type,
            "segments": [],
        }
        for s in bp.segments:
            path_data["segments"].append({
                "control_points": s.control_points.tolist(),
                "source_type": s.source_type,
            })
        data.append(path_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def extract_from_svg(svg_path: str) -> List[BezierPath]:
    """Parse an SVG file and return its paths as cubic Bezier objects."""
    extractor = SVGBezierExtractor(svg_path)
    return extractor.extract()


def fit_from_contours(
    contours: list,
    corner_threshold: float = 2.0,
    max_error: float = 5.0,
) -> List[BezierPath]:
    """Fit cubic Bezier paths to a list of OpenCV contours."""
    fitter = ContourBezierFitter(corner_threshold=corner_threshold, max_error=max_error)
    return fitter.fit(contours)


def fit_from_image(
    image_path: str,
    min_contour_area: float = 100.0,
    corner_threshold: float = 2.0,
    max_error: float = 5.0,
) -> List[BezierPath]:
    """End-to-end: raster image → contours → fitted cubic Bezier paths."""
    contours, _ = _extract_raster_contours(image_path, min_contour_area)
    return fit_from_contours(contours, corner_threshold, max_error)


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_paths(
    paths: List[BezierPath],
    title: str = "Bezier Curves",
    background: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show_controls: bool = True,
    pts_per_segment: int = 80,
) -> None:
    """Plot Bezier paths with optional control-point handles.

    Args:
        paths:           list of BezierPath objects
        title:           figure title
        background:      optional grayscale or BGR image to show behind curves
        save_path:       if given, save figure to this file
        show_controls:   draw control points and handle lines
        pts_per_segment: sampling density per segment
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(title)
    ax.set_aspect("equal")

    if background is not None:
        if background.ndim == 2:
            ax.imshow(background, cmap="gray", alpha=0.4)
        else:
            ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), alpha=0.4)

    colors = plt.cm.tab10.colors
    for idx, bp in enumerate(paths):
        color = colors[idx % len(colors)]
        for seg in bp.segments:
            pts = seg.sample(pts_per_segment)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)
            if show_controls:
                cp = seg.control_points
                # Draw handle lines P0–P1 and P2–P3
                ax.plot(
                    [cp[0, 0], cp[1, 0]], [cp[0, 1], cp[1, 1]],
                    color=color, linewidth=0.6, linestyle="--", alpha=0.5,
                )
                ax.plot(
                    [cp[2, 0], cp[3, 0]], [cp[2, 1], cp[3, 1]],
                    color=color, linewidth=0.6, linestyle="--", alpha=0.5,
                )
                # Control points
                ax.plot(
                    cp[:, 0], cp[:, 1], "o",
                    color=color, markersize=3, alpha=0.6,
                )

    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  ➜  Saved visualization → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# Quick-run CLI
# ═══════════════════════════════════════════════════════════════════════════

def _print_summary(paths: List[BezierPath], label: str) -> None:
    total_segs = sum(p.num_segments for p in paths)
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  Paths: {len(paths)}  |  Total segments: {total_segs}")
    for i, p in enumerate(paths):
        status = "closed" if p.is_closed else "open"
        print(f"    Path {i}: {p.num_segments} segments ({status})")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bezier.py <image_or_svg_path> [output.svg]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_svg = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUTPUT_DIR, "bezier_output.svg")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".svg":
        paths = extract_from_svg(input_path)
        label = f"SVG extraction: {input_path}"
    else:
        paths = fit_from_image(input_path)
        label = f"Contour fitting: {input_path}"

    _print_summary(paths, label)

    if paths:
        export_to_svg(paths, output_svg)
        print(f"  ➜  SVG written → {output_svg}")

        json_path = os.path.splitext(output_svg)[0] + ".json"
        export_to_json(paths, json_path)
        print(f"  ➜  JSON written → {json_path}")

        visualize_paths(paths, title=label, save_path=os.path.join(OUTPUT_DIR, "bezier_vis.png"))
