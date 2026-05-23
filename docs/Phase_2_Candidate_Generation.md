# Phase 2 — Candidate Generation

> **Source file:** [`restoration/candidates.py`](../restoration/candidates.py)

---

## 1. Purpose

Phase 2 identifies all geometrically plausible connections between open endpoints detected in Phase 1. For every open endpoint, it searches for partner endpoints and generates **ConnectionCandidate** objects — each representing a potential bridge with a preliminary curve and diagnostic metadata.

The goal is to be **generous in generation and conservative in filtering** — generating enough candidates to avoid missing valid connections, while applying geometric gates to reject clearly implausible pairings before they reach the scoring and ASP phases.

---

## 2. Pipeline Overview

```
EndpointInfo list  +  BezierPath list  +  ExtractionResult
    │
    ├─► Build KD-tree from all endpoint positions
    │
    ├─► Same-Path Self-Closure Search
    │       ├─► Gap distance / path length ratio check (≤ 22%)
    │       ├─► Direction gate (relaxed thresholds)
    │       └─► Continuation or extension-intersection bridge
    │
    ├─► Cross-Path Endpoint Pairing (Tier 1: 15% diagonal)
    │       ├─► Direction gates (forward ≥ 0.05, bilateral ≥ 0.20)
    │       ├─► Good-continuation test (tangent ray proximity)
    │       ├─► Extension-intersection test (ray convergence)
    │       └─► Bridge-body crossing filter
    │
    ├─► Cross-Path Endpoint Pairing (Tier 2: 30% diagonal)
    │       ├─► Direction gates (forward ≥ 0.15, bilateral ≥ 0.25)
    │       ├─► Stricter misalignment cap (≤ 72°)
    │       └─► Same tests as Tier 1
    │
    └─► Per-endpoint cap (max 5 candidates)
            └─► List[ConnectionCandidate]
```

---

## 3. Spatial Search Strategy

### KD-Tree Construction

A `scipy.spatial.cKDTree` is built from all endpoint positions for efficient nearest-neighbor queries:

```python
positions = np.array([ep.position for ep in endpoints])
kdtree = cKDTree(positions)
```

### Two-Tier Radius System

| Tier | Radius | Purpose | Thresholds |
|---|---|---|---|
| **Tier 1** | 15% of image diagonal | High-confidence nearby connections | Relaxed direction gates |
| **Tier 2** | 30% of image diagonal | Extended reach for isolated endpoints | Stricter direction gates |

Tier 2 only activates for endpoints that have **fewer than 2 Tier 1 candidates**, ensuring that well-connected endpoints don't accumulate excessive candidates.

---

## 4. Direction Gates

Before any candidate is generated, its endpoint pair must pass directional compatibility checks. These gates are the primary filter against geometrically implausible connections.

### Alignment Metrics

For each endpoint pair (A, B), four metrics are computed:

```python
direction_ab = ep_b.position - ep_a.position  # Vector from A to B
dir_ab = direction_ab / ||direction_ab||       # Normalized

forward_a = dot(ep_a.tangent, dir_ab)          # A's tangent alignment toward B
forward_b = dot(ep_b.tangent, -dir_ab)         # B's tangent alignment toward A
bilateral = min(forward_a, forward_b)           # Worst-case alignment
misalignment = arccos(dot(tangent_a, -tangent_b))  # Angle between opposing tangents
```

### Gate Thresholds

| Parameter | Same-Path | Tier 1 | Tier 2 |
|---|---|---|---|
| Min forward (each) | -0.05 | 0.05 | 0.15 |
| Min bilateral | -0.10 | 0.20 | 0.25 |
| Max misalignment | 120° | 90° | 72° |

**Special relaxation:** When misalignment < 25° (nearly anti-parallel tangents), forward and bilateral thresholds drop to -0.80, allowing connections even when tangents don't point exactly toward the partner.

**Same-path high-confidence override:** For same-path closures, misalignment up to 150° is allowed if bilateral ≥ -0.02 and both tangent confidences ≥ 0.55.

---

## 5. Candidate Scenarios

### 5.1 Good Continuation

**Trigger:** The tangent ray from endpoint A passes within tolerance of endpoint B's position.

```python
perp_dist, ray_param = _ray_point_distance(ep_a.position, ep_a.tangent, ep_b.position)
# Accepted if perp_dist < proximity_threshold and ray_param > 0
```

**Bridge construction:** Single G1-continuous cubic Bézier from A to B:
```
P₀ = A.position
P₁ = A.position + α · A.tangent
P₂ = B.position + α · B.tangent  (tangent points outward, so this creates correct incoming direction)
P₃ = B.position

α = chord_length / 3.0
```

### 5.2 Extension-Intersection

**Trigger:** The tangent rays from both endpoints converge at an intersection point.

**Two sub-methods:**

#### Linear Intersection (Ray-Ray)
```python
intersection = _line_line_intersection(
    ep_a.position, ep_a.tangent,
    ep_b.position, ep_b.tangent,
)
```
- Both ray parameters must be positive (intersection ahead of both rays)
- Returns `None` if rays are parallel (determinant < 1e-10)

#### Curved Intersection (Bézier Extrapolation)
When linear intersection fails, the system extrapolates the terminal Bézier segment beyond its [0, 1] domain:

```python
extrapolated_a = _extrapolate_bezier(cp_a, "forward", steps=10, dt=0.05)
extrapolated_b = _extrapolate_bezier(cp_b, "backward", steps=10, dt=0.05)
intersection = _nearest_approach(extrapolated_a, extrapolated_b, threshold)
```

- Evaluates the cubic Bézier at t = 1.05, 1.10, ..., 1.50 (or t = -0.05, -0.10, ..., -0.50)
- Finds the nearest approach between the two extrapolated curves
- Returns the midpoint if minimum distance < threshold

**Bridge construction:** Two-segment bridge through the intersection point:
```
Segment 1: A.position → Intersection
Segment 2: Intersection → B.position
```

### 5.3 Self-Closure

**Trigger:** Both endpoints belong to the same path (the gap closes a loop).

**Additional qualification:**
- Gap distance ≤ 22% of path arc length
- Short paths (< 0.8% of diagonal or ≤ 2 segments) are flagged as `spur_involved`

Self-closure candidates use the same continuation or extension-intersection bridge construction, but with relaxed direction gates and boosted scoring in Phase 3.

---

## 6. Filtering & Post-Processing

### Bridge-Body Crossing Detection

For Tier 2 candidates and extension-intersection bridges, the system checks whether the bridge path passes through existing sketch lines:

```python
# Sample the bridge at 20 points
# For each sample point, check distance to all non-participating paths
# If any point is within 3 pixels of another path → penalize or reject
```

### Per-Endpoint Cap

Maximum 5 candidates per endpoint (configurable via `max_per_endpoint`). When exceeded, candidates are ranked by a preliminary quality metric and the lowest-scoring are dropped.

### Spur Detection

Paths shorter than 0.8% of the image diagonal or with ≤ 2 segments are flagged:
```python
spur_involved = (path_length < diagonal * 0.008) or (num_segments <= 2)
```
Spur flags propagate to scoring (Phase 3) where they apply a -0.22 penalty.

---

## 7. Key Data Structure

### ConnectionCandidate

```python
@dataclass
class ConnectionCandidate:
    id: int                                # Unique candidate ID
    ep_a: EndpointInfo                     # Source endpoint
    ep_b: EndpointInfo                     # Target endpoint
    scenario: str                          # "continuation" | "extension_intersection" | "self_closure"
    bridge_points: np.ndarray              # (N, 2) sampled preliminary bridge
    bridge_bezier: List[BezierSegment]     # Preliminary bridge segments
    distance: float                        # Euclidean distance between endpoints
    tier: int                              # 1 = normal radius, 2 = extended radius
    bilateral_alignment: float             # min(forward_a, forward_b)
    misalignment_deg: float                # Tangent misalignment in degrees
    spur_involved: bool                    # One or both paths are short spurs
    same_path_closure: bool                # Both endpoints on same path
    intersection_point: Optional[np.ndarray]  # Convergence point (if extension)
    extension_quality: float               # Quality metric for extension bridges
    relaxed_tier2_extension: bool          # Extended Tier 2 with relaxed gates
    score: float                           # Set in Phase 3
```

---

## 8. Entry Point

```python
def generate_candidates(
    extraction: ExtractionResult,
    lookahead_fraction: float = 0.15,        # Tier 1 radius as fraction of diagonal
    max_per_endpoint: int = 5,               # Max candidates per endpoint
    diagnostics: Optional[dict] = None,      # Optional diagnostics output
) -> List[ConnectionCandidate]:
```

---

## 9. Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `lookahead_fraction` | 0.15 | Tier 1 search radius as fraction of image diagonal |
| `max_per_endpoint` | 5 | Maximum candidates generated per endpoint |
| `self_closure_gap_ratio` | 0.22 | Maximum gap/path-length ratio for self-closure |
| `spur_diagonal_fraction` | 0.008 | Paths shorter than this fraction are spurs |
| `tier2_multiplier` | 2.0 | Tier 2 radius = Tier 1 × this multiplier |

---

## 10. Diagnostics Output

When `diagnostics` is provided, the function populates detailed rejection statistics:

```json
{
  "radius_px": { "tier1": 150.2, "tier2": 300.4 },
  "same_path": {
    "evaluated_paths": 12,
    "eligible_pairs": 8,
    "generated_pairs": 5,
    "rejection_reason_counts": {
      "direction_gate_failed": 3,
      "gap_ratio_exceeded": 2
    }
  },
  "cross_path": {
    "pairs_within_tier2": 45,
    "eligible_pairs": 28,
    "generated_pairs": 18,
    "rejection_reason_counts": {
      "direction_gate_failed": 12,
      "no_scenario_matched": 5
    }
  }
}
```

---

## 11. Dependencies

| Library | Usage |
|---|---|
| `scipy.spatial.cKDTree` | Efficient spatial nearest-neighbor queries |
| `scipy.spatial.distance.cdist` | Distance matrix for nearest-approach detection |
| `numpy` | Vector math, array operations |
