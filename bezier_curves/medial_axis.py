"""
Geometric Medial Axis Extraction Module
=======================================
Replaces raster morphological skeletonization with a purely vector approach.

Pipeline:
  1. Image -> Potrace -> GeoJSON Polygons (perfect Bezier boundaries)
  2. Polygons -> pygeoops.centerline -> Voronoi MultiLineString
  3. MultiLineString -> networkx Graph -> sknw chain builder -> Bezier Paths

This bypasses all pixel-grid aliasing and junction tearing artifacts
by performing topology resolution in continuous mathematical space.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import cv2
import networkx as nx
import numpy as np
import pygeoops
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString

from .bezier import (
    BezierPath, 
    BezierSegment, 
    _fit_cubic_single, 
    _estimate_tangent,
    _build_skeleton_chains,
    _merge_connected_paths,
    _prune_skeleton_spurs
)

def get_potrace_path() -> str:
    """Resolve the path to the standalone potrace.exe."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe_path = os.path.join(base_dir, "potrace.exe")
    if not os.path.exists(exe_path):
        return "potrace"
    return exe_path


def _clean_polygon_holes(poly: Polygon, min_hole_area: float = 50.0) -> Polygon:
    """Remove tiny interior rings (noise holes) that cause Voronoi loops."""
    if not poly.interiors:
        return poly
    new_interiors = []
    for ring in poly.interiors:
        if Polygon(ring).area >= min_hole_area:
            new_interiors.append(ring)
    return Polygon(poly.exterior, new_interiors)


def _detect_corners(points: np.ndarray, angle_threshold_deg: float = 70.0) -> Set[int]:
    """Detect sharp corner indices in a polyline."""
    corners: Set[int] = set()
    if len(points) < 3:
        return corners
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_theta)))
        if angle > angle_threshold_deg:
            corners.add(i)
    return corners


def _chaikin_smooth(points: np.ndarray, iterations: int = 2,
                    corner_threshold_deg: float = 65.0) -> np.ndarray:
    """Chaikin corner-cutting smoothing with sharp-corner pinning.

    Applies Chaikin subdivision to smooth noisy Voronoi centerline polylines
    while preserving sharp corners (detected by angle threshold). Endpoints
    are always preserved.
    """
    if len(points) < 3:
        return points

    for _ in range(iterations):
        pinned = _detect_corners(points, corner_threshold_deg)
        # Always pin first and last
        pinned.add(0)
        pinned.add(len(points) - 1)

        new_pts = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            if i in pinned:
                new_pts.append(p0)
                if (i + 1) in pinned:
                    # Both pinned — keep the segment as-is
                    pass
                else:
                    # Only start pinned: add the 3/4 point
                    new_pts.append(0.25 * p0 + 0.75 * p1)
            else:
                if (i + 1) in pinned:
                    # Only end pinned: add the 1/4 point
                    new_pts.append(0.75 * p0 + 0.25 * p1)
                else:
                    # Neither pinned: standard Chaikin subdivision
                    new_pts.append(0.75 * p0 + 0.25 * p1)
                    new_pts.append(0.25 * p0 + 0.75 * p1)

        # Always add the last point
        new_pts.append(points[-1])
        points = np.array(new_pts, dtype=np.float64)

    return points


def _estimate_stroke_half_width(polygon: Polygon) -> float:
    """Estimate the stroke half-width from the polygon area/perimeter ratio.

    For a long thin stroke: area ≈ width × length, perimeter ≈ 2 × (width + length).
    So width ≈ area / (perimeter/2 - width) ≈ 2 × area / perimeter for thin strokes.
    """
    area = polygon.area
    perim = polygon.length
    if perim < 1e-6:
        return 1.0
    return max(0.5, float(area / perim))


def _extract_vector_boundaries(
    image_path: str, image_height: int, min_area: float = 50.0
) -> List[Polygon]:
    """Convert raster sketch to smooth vector polygons using Potrace."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Standardize and clean the image
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    with tempfile.TemporaryDirectory() as tmpdir:
        bmp_path = os.path.join(tmpdir, "input.bmp")
        json_path = os.path.join(tmpdir, "output.json")
        
        cv2.imwrite(bmp_path, binary)
        
        potrace_exe = get_potrace_path()
        # -i inverts the image. 
        # -a 0.0 prevents Potrace from preemptively rounding off sharp corners
        # -t turdsize: suppress speckles.
        turdsize = max(2, int(image_height * 0.002))
        cmd = [potrace_exe, bmp_path, "-i", "-b", "geojson", "-a", "0.0", "-t", str(turdsize), "-o", json_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Potrace failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(f"Potrace executable not found at {potrace_exe}. Please download potrace.exe and place it in the project root.")

        if not os.path.exists(json_path):
            return []

        with open(json_path, "r") as f:
            geojson = json.load(f)

    polygons: List[Polygon] = []
    features = geojson.get("features", [])
    
    # Transformation matrix to flip Y-axis
    # x' = x, y' = -y + image_height
    flip_matrix = [1.0, 0.0, 0.0, -1.0, 0.0, float(image_height)]
    
    for feat in features:
        geom = shape(feat["geometry"])
        transformed_geom = affine_transform(geom, flip_matrix)
        
        if isinstance(transformed_geom, MultiPolygon):
            for poly in transformed_geom.geoms:
                if poly.area >= min_area:
                    clean_poly = _clean_polygon_holes(poly)
                    polygons.append(clean_poly)
        elif isinstance(transformed_geom, Polygon):
            if transformed_geom.area >= min_area:
                clean_poly = _clean_polygon_holes(transformed_geom)
                polygons.append(clean_poly)

    smoothed = [p.simplify(0.15, preserve_topology=True) for p in polygons]
    return smoothed


def _compute_voronoi_centerlines(
    polygons: List[Polygon], 
    densify_factor: float = -1.0,
    min_branch_factor: float = 0.0
) -> List[MultiLineString]:
    """Calculate the Medial Axis centerlines for a list of polygons."""
    centerlines = []
    for poly in polygons:
        try:
            cl = pygeoops.centerline(
                poly,
                densify_distance=densify_factor,      
                min_branch_length=min_branch_factor,  
                simplifytolerance=0.0,  # Explicitly disabled to avoid self-intersecting loops
            )
            if cl is not None and not cl.is_empty:
                if isinstance(cl, MultiLineString):
                    centerlines.append(cl)
                elif isinstance(cl, LineString):
                    centerlines.append(MultiLineString([cl]))
        except Exception as e:
            print(f"Warning: Voronoi centerline extraction failed for a polygon: {e}")
            continue

    return centerlines


def _multilinestring_to_sknw_graph(centerlines: Union[MultiLineString, List[LineString]], image_height: float = 1000.0) -> nx.MultiGraph:
    """Convert fragmented PyGeoOps linestrings into a connected NetworkX graph.
    Uses a spatial grid to robustly merge nearby endpoints.
    """
    graph = nx.MultiGraph()
    graph.is_multigraph = lambda: True
    node_coords_to_id = {}
    next_node_id = 0

    grid_tolerance = max(1.5, image_height * 0.002)
    grid_size = grid_tolerance * 2.0  # Grid-based spatial lookup to fix fragmentation
    
    def get_node_id(pt):
        nonlocal next_node_id
        if not node_coords_to_id:
            node_coords_to_id[next_node_id] = pt
            graph.add_node(next_node_id, o=np.array([pt[1], pt[0]], dtype=np.float64))
            next_node_id += 1
            return 0
            
        # Find nearest
        existing_pts = np.array(list(node_coords_to_id.values()))
        dists = np.linalg.norm(existing_pts - pt, axis=1)
        best_idx = np.argmin(dists)
        if dists[best_idx] <= grid_tolerance:
            return list(node_coords_to_id.keys())[best_idx]
            
        node_coords_to_id[next_node_id] = pt
        graph.add_node(next_node_id, o=np.array([pt[1], pt[0]], dtype=np.float64))
        ret_id = next_node_id
        next_node_id += 1
        return ret_id

    for cl_multi in centerlines:
        for line in cl_multi.geoms:
            if line.length < 1e-3:
                continue
            
            pts = np.array(line.coords, dtype=np.float64)
            u = get_node_id(pts[0])
            v = get_node_id(pts[-1])
            
            # Create "pts" array in [y, x] format for `_extract_skeleton_edges`
            pts_yx = np.column_stack((pts[:, 1], pts[:, 0]))
            
            graph.add_edge(u, v, pts=pts_yx)
            
    return graph


def _split_at_corners(points: np.ndarray, threshold_deg: float = 75.0) -> List[np.ndarray]:
    """Split a polyline into segments at sharp corners to preserve geometry."""
    if len(points) < 3:
        return [points]
    
    splits = [0]
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
            
        cos_theta = np.dot(v1, v2) / (n1 * n2)
        angle = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
        
        if angle > threshold_deg:
            splits.append(i)
            
    splits.append(len(points) - 1)
    
    segments = []
    for i in range(len(splits) - 1):
        seg = points[splits[i] : splits[i+1] + 1]
        if len(seg) >= 2:
            segments.append(seg)
    return segments


def fit_from_image_geometric(
    image_path: str,
    image_height: int,
    min_area: float = 50.0,
    max_error: float = 2.0,
    tangent_lookahead: int = 8,
    straightness_scale: float = 0.75,
    merge_radius: float = 5.0,
    follow_junction_continuation: bool = True,
    junction_min_alignment: float = -0.15,
    spur_threshold: float = 12.0,
    simplification_tolerance: float = 0.5,
) -> Tuple[List[BezierPath], Dict[int, set]]:
    """End-to-end geometric extraction: raster -> boundary -> voronoi -> bezier."""
    polygons = _extract_vector_boundaries(image_path, image_height=image_height, min_area=min_area)
    
    # Per-polygon adaptive densification: smaller polygons need denser Voronoi sampling
    # to produce accurate centerlines for thin/short strokes.
    diagonal = float(np.hypot(image_height, image_height))  # approximate
    min_chain_length = max(3.0, diagonal * 0.003)

    all_centerlines: List[MultiLineString] = []
    stroke_widths: Dict[int, float] = {}  # polygon index -> estimated half-width
    for poly_idx, poly in enumerate(polygons):
        half_w = _estimate_stroke_half_width(poly)
        stroke_widths[poly_idx] = half_w
        # Adaptive densify: smaller polygons get finer sampling
        adaptive_densify = max(2.0, min(half_w * 1.5, float(np.sqrt(poly.area)) * 0.01))
        # But never exceed a safe upper bound to prevent hangs
        adaptive_densify = min(adaptive_densify, max(3.0, image_height * 0.003))
        cls = _compute_voronoi_centerlines(
            [poly],
            densify_factor=adaptive_densify,
            min_branch_factor=0.0,
        )
        all_centerlines.extend(cls)

    # Reconstruct the graph to maintain continuous strokes through junctions
    graph = _multilinestring_to_sknw_graph(all_centerlines, image_height=image_height)
    
    # Prune topological spurs to prevent path fragmentation
    _prune_skeleton_spurs(graph, threshold_length=spur_threshold)
    
    # Extract continuous chains using the existing sknw logic
    chains = _build_skeleton_chains(
        graph,
        follow_junction_continuation=follow_junction_continuation,
        junction_min_alignment=junction_min_alignment,
    )

    paths: List[BezierPath] = []
    lookahead = max(1, int(tangent_lookahead))
    for chain in chains:
        pts = chain.points
        if len(pts) < 2:
            continue

        # Discard chains shorter than the minimum length threshold
        chain_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if chain_len < min_chain_length:
            continue

        # Adaptive simplification tolerance based on estimated stroke width.
        # Thin strokes get tighter tolerance; thick strokes get more aggressive simplification.
        avg_half_width = np.mean(list(stroke_widths.values())) if stroke_widths else 2.0
        adaptive_simp = max(0.2, min(simplification_tolerance, avg_half_width * 0.3))

        # Downsample extremely dense Voronoi points to prevent deep recursion/hangs
        if len(pts) > 10 and adaptive_simp > 0.0:
            ls = LineString(pts).simplify(adaptive_simp, preserve_topology=True)
            if ls.is_simple:
                pts = np.array(ls.coords)
        
        if len(pts) < 2:
            continue

        # Phase A: Chaikin smoothing with corner pinning to clean Voronoi noise
        # while preserving sharp corners (critical for bolt, star, shape geometries)
        if len(pts) >= 4:
            pts = _chaikin_smooth(pts, iterations=2, corner_threshold_deg=65.0)

        if len(pts) < 2:
            continue

        left_tangent = _estimate_tangent(pts, "start", lookahead=lookahead)
        right_tangent = _estimate_tangent(pts, "end", lookahead=lookahead)
        cps_list = _fit_cubic_single(
            pts,
            left_tangent,
            right_tangent,
            max_error ** 2,
            straightness_scale=straightness_scale,
        )

        segments = [BezierSegment(cps, source_type="geometric_centerline") for cps in cps_list]
        if not segments:
            continue

        paths.append(
            BezierPath(
                segments=segments,
                is_closed=chain.is_closed,
                source_type="geometric_centerline",
            )
        )

    merged_paths = _merge_connected_paths(paths, merge_radius=merge_radius)

    return merged_paths, {}
