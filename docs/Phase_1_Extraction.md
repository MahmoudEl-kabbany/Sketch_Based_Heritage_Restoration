# Phase 1 — Extraction

> **Source file:** [`restoration/extraction.py`](../restoration/extraction.py)  
> **Supporting module:** [`bezier_curves/bezier.py`](../bezier_curves/bezier.py), [`eliptic_fourier_descriptors/efd.py`](../eliptic_fourier_descriptors/efd.py)

---

## 1. Purpose

Phase 1 transforms a raw raster sketch image into a structured geometric representation suitable for gap detection and restoration. It produces:

- **Bézier paths** — cubic Bézier curve representations of every line in the sketch
- **Endpoint descriptors** — position, outward tangent, curvature, and tangent confidence for each open path endpoint
- **EFD contour data** — Elliptic Fourier Descriptor coefficients for closed contours
- **Distance map** — pixel-wise distance from the nearest sketch line (for collision detection in later phases)

---

## 2. Pipeline Overview

```
Raster Image (grayscale)
    │
    ├─► Otsu Binarization (lines = foreground)
    │
    ├─► Skeletonization (scikit-image → 1-pixel medial axis)
    │
    ├─► Graph Construction (sknw → NetworkX graph)
    │
    ├─► Chain Traversal (junction-continuation heuristics)
    │
    ├─► Schneider Fitting (cubic Bézier curves)
    │       └─► BezierPath objects
    │
    ├─► Endpoint Extraction
    │       ├─► Derivative tangent from control points
    │       ├─► Multi-scale PCA context tangent
    │       ├─► Blended tangent + confidence score
    │       └─► EndpointInfo objects
    │
    ├─► EFD Contour Extraction
    │       └─► Fourier coefficients (order=40)
    │
    └─► Distance Map Computation
            └─► cv2.distanceTransform
```

---

## 3. Detailed Steps

### 3.1 Image Loading & Binarization

```python
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

- Grayscale loading ensures color images are handled uniformly
- **Otsu thresholding** automatically selects the optimal binarization threshold
- `BINARY_INV` means lines (dark pixels) become foreground (255)

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

### 3.4 Chain Traversal

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

### 3.5 Schneider's Bézier Fitting Algorithm

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

### 3.6 Endpoint Tangent Estimation

For each open path, tangent vectors and curvature values are computed at both endpoints:

#### Derivative Tangent

Directly from the Bézier control points:
```
B'(0) = 3(P₁ - P₀)    → tangent at start
B'(1) = 3(P₃ - P₂)    → tangent at end
```
The tangent is negated to point **outward** (away from the path interior).

#### Multi-Scale PCA Context Tangent

A more robust tangent estimate using local path samples:

1. **Sample the path** at 24 points per segment
2. **Remove duplicate points** (distance < 1e-6)
3. **Run PCA at 4 window sizes:** 10%, 15%, 20%, 25% of path length
4. For each window:
   - Extract local points near the endpoint
   - Center points (subtract mean)
   - Compute covariance matrix and eigenvectors
   - Principal eigenvector = dominant direction
   - Orient to point outward (dot product with reference direction)
   - Confidence = `0.5 × linearity + 0.5 × directional_alignment`
     - Linearity = `eigenvalue_max / sum(eigenvalues)`
     - Directional alignment = `max(0, dot(principal, outward_ref))`
5. **Circular mean:** Weighted average of all 4 directional estimates
6. **Final confidence:** `R × mean(weights)` where R is the resultant length

#### Tangent Blending

The final tangent is a blend of derivative and context tangents:

```python
if confidence < 0.40:
    blend = clip(confidence, 0.05, 0.35)  # Low confidence → favor derivative
else:
    blend = clip(confidence, 0.40, 0.88)  # High confidence → favor context

tangent = normalize((1 - blend) × derivative_tangent + blend × context_tangent)
```

**Safety check:** If the blended tangent points into the path interior (dot with interior direction > 0.50), it is flipped and confidence is halved.

#### Curvature

Computed from Bézier first and second derivatives:
```
κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
```

### 3.7 EFD Contour Extraction

Closed contours are extracted and their Elliptic Fourier Descriptor coefficients computed:

1. **Binarize** with Otsu + morphological closing (5×5 elliptical kernel)
2. **Find contours** using `cv2.findContours` with `RETR_TREE` hierarchy
3. **Filter** by minimum area (default: 100 pixels²)
4. **Compute EFD coefficients** using `pyefd` with order=40 harmonics
5. **Store** contour points, coefficients, and DC locus (a₀, c₀)

### 3.8 Distance Map

```python
dist_map = cv2.distanceTransform((255 - binary), cv2.DIST_L2, 5)
```

The distance map stores, for each pixel, the Euclidean distance to the nearest foreground (line) pixel. This is used in Phase 4 for bridge collision detection.

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
    tangent: np.ndarray      # Unit tangent pointing outward from path
    curvature: float         # Unsigned curvature κ at endpoint
    tangent_confidence: float  # 0.0–1.0 confidence in tangent direction
```

### ExtractionResult

```python
@dataclass
class ExtractionResult:
    paths: List[BezierPath]          # All extracted Bézier paths
    endpoints: List[EndpointInfo]    # Endpoints of open paths only
    efd_contours: List[dict]         # EFD data for closed contours
    image_shape: Tuple[int, int]     # (height, width)
    diagonal: float                  # √(h² + w²)
    dist_map: Optional[np.ndarray]   # Distance map from lines
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
```

---

## 5. Entry Point

```python
def extract_paths(
    image_path: str,
    max_error: float = 2.0,       # Schneider fitting tolerance (pixels²)
    spur_threshold: float = 12.0, # Minimum path length to keep
    merge_radius: float = 5.0,    # Endpoint merge radius
) -> ExtractionResult:
```

---

## 6. Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_error` | 2.0 | Maximum squared error for Schneider fitting (pixels²) |
| `spur_threshold` | 12.0 | Minimum path arc length to keep (removes noise spurs) |
| `merge_radius` | 5.0 | Radius for merging nearby endpoints |
| `min_contour_area` | 100 | Minimum area for EFD contour extraction |
| `efd_order` | 40 | Number of Fourier harmonics for EFD |

---

## 7. Dependencies

| Library | Usage |
|---|---|
| `opencv-python` | Image I/O, binarization, morphology, distance transform, contour detection |
| `scikit-image` | Morphological skeletonization |
| `sknw` | Skeleton-to-graph conversion |
| `numpy` | Array operations, linear algebra |
| `pyefd` | Elliptic Fourier Descriptor computation |
