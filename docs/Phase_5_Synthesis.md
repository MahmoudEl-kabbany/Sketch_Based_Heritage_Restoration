# Phase 5 — Synthesis

> **Source file:** [`restoration/synthesis.py`](../restoration/synthesis.py)

---

## 1. Purpose

Phase 5 takes the list of `ConnectionCandidate` objects accepted by the ASP solver and generates the final, production-quality cubic Bézier curves (bridges) that connect them. It then merges these new bridge segments with the original extraction segments to form complete, contiguous paths.

---

## 2. Pipeline Overview

```
Accepted Candidates
    │
    ├─► For each candidate:
    │       ├─► Determine scenario (continuation vs extension)
    │       ├─► Build straight bridge if geometrically straight
    │       ├─► Build G1-continuous bridge if smooth curve
    │       ├─► Build 2-segment intersection bridge if corner
    │       └─► Return List[BezierSegment]
    │
    ├─► Merge Path Topologies:
    │       ├─► Track endpoints as connectable tokens
    │       ├─► For each bridge:
    │       │       ├─► Connect Source Path + Bridge + Target Path
    │       │       ├─► Reverse paths if necessary (start-to-start)
    │       │       └─► Detect if merged path forms a closed loop
    │       └─► Output updated List[BezierPath]
    │
    └─► Final Output: Restored Paths + Independent Bridge Segments
```

---

## 3. Bridge Construction Methods

All generated bridges are defined as `BezierSegment` objects containing 4 control points `(P0, P1, P2, P3)`. To prevent kinks, the handles (P1 and P2) are clamped by a stabilization function.

### 3.1 Control Point Stabilization

`_stabilize_bridge_control_points(cp)` ensures handle geometry doesn't create "hooks" or backtracking curves:
- Forces minimum forward projection along the chord (at least 2% of chord length, rescued to 8% if below)
- Clamps maximum lateral deviation to 60% of chord length

### 3.2 Straight Bridges

Used when both endpoint curvatures are near zero (< 0.005) and their tangents are perfectly aligned (> 95% anti-parallel).

```python
# P1 and P2 are placed exactly 1/3 and 2/3 along the straight line from P0 to P3
unit = (P3 - P0) / norm(P3 - P0)
step = norm(P3 - P0) / 3.0
P1 = P0 + unit * step
P2 = P3 - unit * step
```

### 3.3 Smooth G1-Continuous Bridges

The default bridge for organic continuation. It creates a single cubic Bézier segment that maintains first-derivative (G1) continuity with the incoming paths, while scaling the handle lengths based on local curvature to prevent overshooting.

```python
chord = norm(P3 - P0)
alpha_a = chord / 3.0
alpha_b = chord / 3.0

# Curvature-aware scaling: high curvature = shorter handles
if curvature_a > 0: alpha_a = min(alpha_a, 1.0 / (3.0 * curvature_a))
if curvature_b > 0: alpha_b = min(alpha_b, 1.0 / (3.0 * curvature_b))

P1 = P0 + alpha_a * tangent_a
P2 = P3 + alpha_b * tangent_b  (tangent_b points outward, so + is correct for incoming)
```

### 3.4 Intersection Bridges (Corners)

Used for `extension_intersection` candidates to create sharp corners. Two Bézier segments are generated, meeting at the intersection point `I`.

**Linear Intersection (Sharp Corners):**
Used when both endpoints are straight lines (< 0.008 curvature).
- Segment 1: `P0` to `I`
- Segment 2: `I` to `P3`
- The handles going toward and away from `I` follow the direct straight-line vectors to `I`.

**Curved Intersection (Organic Convergence):**
- Segment 1: `P0` to `I`. `P1` follows the source tangent. `P2` approaches `I` from the chord direction.
- Segment 2: `I` to `P3`. `P1` departs `I` along the chord direction. `P2` follows the target tangent.

**Smooth Fallback:**
If the calculated intersection point `I` is very close to the chord midpoint (distance < 45% of chord), drawing a 2-segment bridge through it creates an unnatural kink. In this case, the system discards the intersection and falls back to a single smooth **G1 bridge**.

---

## 4. Path Merging Topology

After the bridge segments are created, the discrete paths must be chained together.

1. **Tokenization:** Every start and end of every path is assigned a stable token `(type, ID)`.
2. **Path Storage:** All paths are loaded into an `active` dictionary.
3. **Bridge Iteration:** For each accepted connection A ↔ B:
   - Look up the active path containing endpoint A.
   - Look up the active path containing endpoint B.
   - **Orientation:** Ensure Path A ends at the bridge, and Path B starts at the bridge. If they are backwards, the entire path is reversed using `_reverse_segments()` (reversing the segment list and swapping P0/P3 and P1/P2 for every segment).
   - **Concatenation:** `Merged Segments = Path A Segments + Bridge Segment(s) + Path B Segments`.
   - Update the `active` dictionary with the new merged path and delete Path B.
   - If Path A and Path B were the *same* active path, the merge creates a closed loop. The path is updated and its start/end tokens are set to `None`.
4. **Closure Check:** Finally, all paths are iterated. If the distance between the first and last point of the path is less than 1% of the total path length (or 5 pixels), `is_closed` is set to `True`.

---

## 5. Entry Points

```python
def synthesize_bridges(
    accepted: List[ConnectionCandidate],
) -> List[BezierSegment]:
    # Returns just the new bridge segments

def merge_restored_paths(
    original_paths: List[BezierPath],
    bridges: List[BezierSegment],
    accepted: List[ConnectionCandidate],
) -> List[BezierPath]:
    # Returns the combined final topology
```

---

## 6. Dependencies

| Library | Usage |
|---|---|
| `numpy` | Vector math, Euclidean distance |
