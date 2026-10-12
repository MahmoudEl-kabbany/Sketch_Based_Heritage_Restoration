# Phase 1 — Extraction

> **Source file:** [`restoration/extraction.py`](../restoration/extraction.py)
> **Supporting module:** [`bezier_curves/bezier.py`](../bezier_curves/bezier.py)

---

## 1. Purpose

Phase 1 turns a raster sketch into the geometry the later phases reason about:

- **Bézier paths** — every stroke as a chain of cubic Bézier curves, each knowing whether its two ends are *free* (the stroke stops there) or sit on a *junction* with other strokes
- **Free ends** — for every place a stroke stops: position, outward direction, curvature and stroke radius. These are the only places a missing stroke can be attached
- **Stroke measurements** — the line width of the drawing and a distance map

Every size used here is a multiple of the stroke radius, measured from the image itself, so a drawing with 3 px lines and one with 30 px lines are treated alike.

---

## 2. Pipeline Overview

```
Raster Image (grayscale)
    │
    ├─► Otsu binarization, speck removal
    ├─► Stroke radius map (distance transform) and line width
    │
    ├─► Skeletonization (scikit-image → 1-pixel medial axis)
    ├─► Graph construction (sknw → NetworkX graph)
    ├─► Spur pruning (terminal branches that are thinning artefacts)
    │
    ├─► Chain traversal (junction-continuation; ends marked free / junction)
    ├─► End trimming (the stretch where the skeleton leaves the centre line)
    ├─► Schneider fitting (cubic Bézier curves) → BezierPath objects
    │
    ├─► Free-end extraction
    │       ├─► stroke radius
    │       └─► direction and curvature from an arc fitted behind the end
    │
    └─► Distance map (cv2.distanceTransform)
```

---

## 3. Detailed Steps

### 3.1 Binarization and Stroke Measurement

```python
binary, radius_map, line_width = stroke_geometry(image_path)
```

- **Otsu thresholding** with `BINARY_INV`: strokes (dark pixels) become foreground.
- **Radius map:** the distance of every stroke pixel to the background. Along the skeleton this is the local stroke radius.
- **Line width:** twice the median radius along the skeleton.
- **Speck removal:** blobs smaller than half a squared line width are deleted. There is no blur: blurring closes small gaps and moves stroke ends.

### 3.2 Skeletonization

The binary image is reduced to a 1-pixel-wide skeleton using `scikit-image`'s morphological skeletonization:

```python
from skimage.morphology import skeletonize
skeleton = skeletonize(binary // 255)
```

**Why skeletonize?** Thick lines would produce ambiguous graph structures. The skeleton extracts the medial axis — the centerline of every stroke — giving a topologically clean representation regardless of line width.

### 3.3 Graph Construction (sknw)

The skeleton is converted into a NetworkX graph using `sknw`:

```python
import sknw
graph = sknw.build_sknw(skeleton)
```

- **Nodes** = junction points (where 3+ branches meet) and endpoint pixels (degree-1 nodes)
- **Edges** = sequences of skeleton pixels connecting nodes
- Each edge stores a `pts` attribute: an (N, 2) array of pixel coordinates along the skeleton segment

### 3.4 Spur Pruning

Thinning leaves short branches that are not strokes: at corners, bumps and blunt stroke ends. A terminal branch (one end free, the other on a junction) is removed when either holds:

| Rule | Condition | Why |
|---|---|---|
| It does not leave the stroke | length ≤ 1.5 × stroke radius at its junction | a real stub sticks out of the body of the stroke it hangs off |
| It runs into a corner | radius at its tip < 0.5 × radius at its junction, and length ≤ 3 radii | a real stub keeps its width to its end; an obliquely cut stroke end gives such a branch |

A real stub left by damage next to a junction survives both rules, which a fixed pixel length cannot guarantee.

### 3.5 Chain Traversal

The graph edges are traversed to form continuous chains of pixels:

1. **Starting priority:** Endpoints and junction nodes are processed first (degree ≠ 2)
2. **Degree-2 pass-through:** At degree-2 nodes, the chain continues to the next edge automatically
3. **Junction continuation:** At junctions (degree ≥ 3), the system selects the edge with the best directional continuation:
   - Incoming direction is computed from the last ~4 pixels of the current edge
   - Each candidate outgoing edge's direction is computed similarly
   - The edge with the highest `cos(angle)` alignment is chosen
   - Minimum alignment threshold: `cos(angle) ≥ 0.5` (60° deviation max)
4. **Cycle collection:** Any unused edges after the main traversal are collected as cycles

**Output:** A list of `_SkeletonChain` objects, each containing an ordered array of (x, y) points.

Each chain records for both of its ends whether it is **free** (a degree-1 node of the graph: the stroke stops) or on a **junction**. A chain that returns to its first node is a closed loop.

### 3.6 End Trimming

At a stroke's end the skeleton leaves the centre line. At a blunt end it forks into the two corners of the outline, 1.4 radii of skeleton for one radius of stroke. Each free end is therefore trimmed by 1.5 stroke radii (never by less than 3 px, since pixel quantization does not shrink with the stroke). The stroke radius is the median along the whole chain; the radius at the tip itself says nothing, it is zero in a corner. Junction ends are not trimmed.

### 3.7 Schneider's Bézier Fitting Algorithm

Each chain of pixel coordinates is fitted with one or more cubic Bézier curves using Schneider's algorithm (Philip J. Schneider, "An Algorithm for Automatically Fitting Digitized Curves", Graphics Gems I, 1990).

#### Core Algorithm

1. **Chord-length parameterization:** Assigns parameter values `t ∈ [0, 1]` to each point based on cumulative arc length
2. **Tangent estimation:** Weighted tangent at each endpoint using exponential decay and curvature down-weighting:
   - Lookahead window: 10 points
   - Skip tip: 2 pixels (to avoid noisy terminal pixels)
   - Turning angle > 25° → weight reduced by 70%
3. **Least-squares fit:** Solves a 2×2 linear system to find optimal control point positions P₁ and P₂ that minimize the sum of squared distances
4. **Newton-Raphson reparameterization:** Iteratively improves parameter values (up to 4 iterations)
5. **Error evaluation:** Computes max and mean squared error between the fitted curve and original points

#### Splitting Strategy

If the fit error exceeds the tolerance (default: `max_error = 2.0` pixels²):

- **Corner detection first:** If a dominant turning angle > 72° exists, split there
- **Near-straight detection:** If the points are nearly linear (max deviation ≤ √max_error × 0.75, mean deviation ≤ 70% of that, turning angle ≤ 32°), use an exactly straight cubic
- **Recursive splitting:** Split at the point of maximum error and recursively fit each half

#### Mathematical Details

**Cubic Bézier evaluation:**
```
B(t) = (1-t)³·P₀ + 3(1-t)²t·P₁ + 3(1-t)t²·P₂ + t³·P₃
```

**Control point generation (Schneider §4):**
```
A[i,0] = 3(1-tᵢ)²tᵢ · T_left
A[i,1] = 3(1-tᵢ)tᵢ² · T_right

C = Aᵀ·A  (2×2 matrix)
X = Aᵀ·(points - baseline)

[α_left, α_right] = C⁻¹ · X

P₁ = P₀ + α_left · T_left
P₂ = P₃ + α_right · T_right
```

**Newton-Raphson reparameterization:**
```
t_new = t - (B(t) - point) · B'(t) / (B'(t) · B'(t) + (B(t) - point) · B''(t))
```

### 3.8 Free Ends: Direction and Curvature

Only free ends become endpoints. An end on a junction is connected to other strokes and is not a place where a stroke could be missing; in the test drawings such ends were 43% of all path ends and under 1% of them lay at a true gap.

The direction of a stroke at a free end is taken from the stroke *behind* the end, not from the last few pixels:

1. Take the stretch of the path within **6 stroke widths** of the end (or the whole path if it is shorter).
2. Fit the circular arc through three points: the end, the middle of the stretch and its far end. Three points in line give a straight line.
3. If the stretch does not follow that arc to within **half a stroke radius** (it contains a corner, say), halve the stretch and fit again, at most twice.
4. The **tangent** is the arc's tangent at the end, pointing outward. The **signed curvature** is the arc's curvature: positive when the stroke, followed towards the end, turns towards `(-tangent.y, tangent.x)`.

This is exact for straight strokes and circular arcs, and an error of a pixel in one of the three points turns the tangent by a fraction of a degree, because the points are several stroke widths apart.

### 3.9 Distance Map

```python
dist_map = cv2.distanceTransform((255 - binary), cv2.DIST_L2, 5)
```

For each pixel, the distance to the nearest stroke pixel (0 on strokes). Phase 3 uses it to measure how much existing stroke a candidate bridge would run over.

---

## 4. Key Data Structures

### EndpointInfo

```python
@dataclass
class EndpointInfo:
    endpoint_id: int         # Stable ID within one extraction
    path_index: int          # Index into ExtractionResult.paths
    end: str                 # "start" | "end"
    position: np.ndarray     # (x, y) pixel coordinates
    tangent: np.ndarray      # Unit tangent pointing outward from the path
    curvature: float         # Unsigned curvature of the stroke as it reaches this end
    radius: float            # Radius (half the width) of the stroke
    signed_curvature: float  # The same curvature with its sign
```

### ExtractionResult

```python
@dataclass
class ExtractionResult:
    paths: List[BezierPath]          # All extracted Bézier paths
    endpoints: List[EndpointInfo]    # Free ends only: where a stroke stops
    image_shape: Tuple[int, int]     # (height, width)
    diagonal: float                  # √(h² + w²)
    dist_map: Optional[np.ndarray]   # Distance of every pixel to the nearest stroke
    line_width: float                # Typical stroke width of the drawing, in pixels
```

### BezierSegment & BezierPath

```python
@dataclass
class BezierSegment:
    control_points: np.ndarray  # (4, 2) — P₀, P₁, P₂, P₃
    source_type: str            # "skeleton" | "bridge" | "efd_closure"

@dataclass
class BezierPath:
    segments: List[BezierSegment]  # Ordered sequence of cubic segments
    is_closed: bool                # Whether the path forms a loop
    source_type: str               # "skeleton" | "restored"
    start_free: bool               # The stroke stops at its start (not on a junction)
    end_free: bool                 # ... and at its end
```

---

## 5. Entry Point

```python
def extract_paths(
    image_path: str,
    max_error: float = 2.0,       # Schneider fitting tolerance (pixels)
) -> ExtractionResult:
```

---

## 6. Constants

All are multiples of the stroke radius or width, set from the geometry described above, not fitted to the test images. The gap benchmark moves little when any of them is set clearly lower or higher (see the README).

| Constant | Value | Meaning |
|---|---|---|
| `SPUR_RADIUS_FACTOR` | 1.5 radii | longest terminal branch that counts as not leaving its stroke |
| `CORNER_TIP_RADIUS_RATIO` | 0.5 | tip radius / junction radius below which a branch runs into a corner |
| `CORNER_BRANCH_MAX_RADII` | 3 radii | longest such corner branch |
| `END_TRIM_RADII` | 1.5 radii | stretch trimmed from a free end |
| `END_TRIM_MIN_PX` | 3 px | least trim, whatever the stroke width |
| `SPECK_AREA_WIDTHS2` | 0.5 widths² | blobs below this area are specks |
| `DIRECTION_WINDOW_WIDTHS` | 6 widths | stretch behind an end its direction is taken from |
| `ARC_FIT_RADII` | 0.5 radii | how closely that stretch must follow one arc |

---

## 7. Dependencies

| Library | Usage |
|---|---|
| `opencv-python` | Image I/O, binarization, distance transform |
| `scikit-image` | Morphological skeletonization |
| `sknw` | Skeleton-to-graph conversion |
| `numpy` | Array operations, linear algebra |
