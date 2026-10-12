"""Phase 1 — Extraction.

Skeleton-based Bezier path extraction with endpoint tangents and curvatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from bezier_curves.bezier import BezierPath, fit_from_image_skeleton, stroke_geometry, stroke_radius
from restoration.geometry import safe_normalize, visual_order_key

# The direction of a stroke at a free end is that of the longest stretch behind
# the end, up to this many stroke widths, that follows a single circular arc to
# within this many stroke radii (never asking for better than a pixel).
DIRECTION_WINDOW_WIDTHS = 6.0
ARC_FIT_RADII = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EndpointInfo:
    """Geometric descriptor for one free end of a BezierPath."""

    endpoint_id: int        # stable endpoint identity within one extraction
    path_index: int          # index into ExtractionResult.paths
    end: str                 # "start" | "end"
    position: np.ndarray     # (x, y)
    tangent: np.ndarray      # unit tangent pointing *outward* from the path
    curvature: float         # unsigned curvature κ of the stroke as it reaches this end
    radius: float = 1.0      # radius (half the width) of the stroke this end belongs to
    # The same curvature with a sign: positive when the stroke, followed towards
    # this end, turns towards (-tangent.y, tangent.x).
    signed_curvature: float = 0.0


@dataclass
class ExtractionResult:
    """Everything extracted from one image."""

    paths: List[BezierPath]
    endpoints: List[EndpointInfo]  # free ends only: where a stroke stops
    image_shape: Tuple[int, int]   # (h, w)
    diagonal: float                # sqrt(h² + w²)
    dist_map: Optional[np.ndarray] = None  # distance of every pixel to the nearest stroke
    line_width: float = 2.0        # typical stroke width of the drawing, in pixels


# ═══════════════════════════════════════════════════════════════════════════
# Direction of a stroke at its end
# ═══════════════════════════════════════════════════════════════════════════

def _arc_through(far: np.ndarray, mid: np.ndarray, tip: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """The circular arc from *far* through *mid* to *tip*.

    Returns (unit tangent at tip in the direction of travel, signed curvature,
    largest distance of *points* from the arc). Three points in line give a
    straight line.
    """
    # Work relative to the tip, which keeps the arithmetic well conditioned.
    far, mid, points = far - tip, mid - tip, points - tip
    chord = safe_normalize(-far)
    bulge = float(mid[0] * chord[1] - mid[1] * chord[0])    # signed distance of mid from the line far-tip
    if abs(bulge) < 0.01:
        return chord, 0.0, float(np.max(np.abs(points[:, 0] * chord[1] - points[:, 1] * chord[0])))

    # Centre of the circle through far, mid and the tip (the origin).
    denominator = 2.0 * float(far[0] * mid[1] - far[1] * mid[0])
    centre = np.array([
        float(mid[1] * (far @ far) - far[1] * (mid @ mid)),
        float(far[0] * (mid @ mid) - mid[0] * (far @ far)),
    ]) / denominator
    radius = float(np.linalg.norm(centre))
    tangent = np.array([centre[1], -centre[0]]) / radius    # perpendicular to the radius at the tip
    if float(tangent @ -mid) < 0.0:
        tangent = -tangent
    turning_left = float(-(mid - far)[0] * mid[1] + (mid - far)[1] * mid[0]) > 0.0   # (mid - far) x (tip - mid)
    deviation = float(np.max(np.abs(np.linalg.norm(points - centre, axis=1) - radius)))
    return tangent, (1.0 if turning_left else -1.0) / radius, deviation


def _end_direction(path: BezierPath, end: str, stroke_radius: float) -> Tuple[np.ndarray, float]:
    """Outward unit tangent and signed curvature of a stroke at one of its ends.

    Both come from the arc through the end and two points further back along
    the stroke. The stretch used is DIRECTION_WINDOW_WIDTHS long; if the
    stroke does not follow one arc over that length (it has a corner, say),
    the stretch is halved, at most twice.
    """
    samples = path.sample(pts_per_segment=24)
    into = samples if end == "end" else samples[::-1]           # running into the end
    steps = np.linalg.norm(np.diff(into, axis=0), axis=1)
    into = into[np.concatenate((steps > 1e-9, [True]))]
    if len(into) < 2:
        return np.array([1.0, 0.0]), 0.0
    steps = np.linalg.norm(np.diff(into, axis=0), axis=1)
    back = np.concatenate((np.cumsum(steps[::-1])[::-1], [0.0]))   # arc length from each sample to the end
    tip = into[-1]

    def point_at(distance: float) -> np.ndarray:
        return np.array([np.interp(distance, back[::-1], into[::-1, axis]) for axis in (0, 1)])

    tolerance = max(ARC_FIT_RADII * stroke_radius, 1.0)
    window = min(DIRECTION_WINDOW_WIDTHS * 2.0 * stroke_radius, float(back[0]))
    for attempt in range(3):
        tangent, curvature, deviation = _arc_through(
            point_at(window), point_at(window / 2.0), tip, into[back <= window],
        )
        if deviation <= tolerance:
            break
        window /= 2.0
    return tangent, curvature


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint extraction
# ═══════════════════════════════════════════════════════════════════════════

def _extract_endpoints(
    paths: List[BezierPath],
    radius_map: np.ndarray,
    fallback_radius: float,
) -> List[EndpointInfo]:
    """Build an EndpointInfo for every free end of every open path.

    An end that sits on a junction is connected to other strokes and is not a
    place where a stroke could be missing.
    """
    endpoints: List[EndpointInfo] = []
    for idx, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue
        radius = stroke_radius(path.sample(pts_per_segment=24), radius_map, fallback_radius)
        for end, is_free in (("start", path.start_free), ("end", path.end_free)):
            if not is_free:
                continue
            pos = (
                path.segments[0].control_points[0]
                if end == "start"
                else path.segments[-1].control_points[3]
            ).astype(np.float64)
            tangent, curvature = _end_direction(path, end, radius)
            endpoints.append(EndpointInfo(
                endpoint_id=len(endpoints),
                path_index=idx,
                end=end,
                position=pos,
                tangent=tangent,
                curvature=abs(curvature),
                radius=radius,
                signed_curvature=curvature,
            ))
    return endpoints


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_paths(
    image_path: str,
    max_error: float = 2.0,
) -> ExtractionResult:
    """Extract Bezier paths, free endpoints, and stroke measurements from a raster image."""
    geometry = stroke_geometry(image_path)
    binary, radius_map, line_width = geometry
    h, w = binary.shape[:2]

    # Bezier path extraction via skeleton
    paths = fit_from_image_skeleton(image_path, max_error=max_error, geometry=geometry)

    # Ensure visual discovery order (top-down, left-right)
    paths.sort(key=visual_order_key)

    return ExtractionResult(
        paths=paths,
        endpoints=_extract_endpoints(paths, radius_map, line_width / 2.0),
        image_shape=(h, w),
        diagonal=float(np.hypot(h, w)),
        # Distance of every pixel to the nearest stroke (0 on strokes)
        dist_map=cv2.distanceTransform((255 - binary).astype(np.uint8), cv2.DIST_L2, 5),
        line_width=line_width,
    )
