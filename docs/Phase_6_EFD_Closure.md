# Phase 6 — EFD Gap Closure

> **Source file:** [`restoration/efd_closure.py`](../restoration/efd_closure.py)

---

## 1. Purpose

Phase 6 addresses open paths that form nearly-closed shapes with a **single remaining gap**. Phase 2/3 candidate generation often ignores these gaps because the endpoints are too far apart for reliable tangent-based extrapolation. Phase 6 analyzes the macro-structure of the whole contour using symmetry detection and curvature heuristics to synthesize a closing arc.

---

## 2. Pipeline Overview

```
List[BezierPath] (Restored)
    │
    ├─► For each open path:
    │       ├─► Skip if gap distance / perimeter ratio > 30% (threshold)
    │       ├─► Skip if gap is already bridged by a Phase 5 connection
    │       │
    │       ├─► Attempt Symmetry Detection
    │       │       ├─► Test 24 reflection axes
    │       │       ├─► Find axis with lowest Hausdorff distance
    │       │       └─► Target: Mirror opposite arc across axis
    │       │
    │       ├─► Plausibility Validation
    │       │       ├─► Check tangent continuation & misalignment
    │       │       ├─► Check curvature coherence
    │       │       └─► Accept/Reject closure
    │       │
    │       ├─► Synthesis
    │       │       ├─► Symmetry Mirroring (if valid axis found)
    │       │       │       └─► Resample and convert to Bezier
    │       │       └─► Curvature-Aware Arc (Fallback)
    │       │
    │       └─► Mark path is_closed=True
    │
    └─► Final List[BezierPath]
```

---

## 3. Qualification Criteria

For an open path to be considered a "single-gap" contour:

1. **Path length:** Must have at least 3 segments and a measurable perimeter.
2. **Gap Ratio:** `Distance(start, end) / ArcLength` must be ≤ 30% (`gap_threshold`).
   - Exception: Small gaps (≤ 20px) on large perimeters (≥ 80px) are forced through regardless of ratio.
3. **No Overlapping Bridges:** If a Phase 5 bridge already exists within proximity of the gap endpoints, the path is skipped.

---

## 4. Symmetry Detection & Mirroring

### 4.1 Axis Detection

If the gap ratio is between 8% and 30%, the algorithm looks for reflectional symmetry.

1. Compute contour centroid.
2. Subsample contour to max 500 points.
3. Test 24 rotational angles (0 to 180° in 7.5° increments) as reflection axes passing through the centroid.
4. For each axis, reflect the points and compute the **directed Hausdorff distance** between the original and reflected set.
5. If the minimum Hausdorff distance is below threshold (default 8.0px, scales with perimeter), the axis is selected.

### 4.2 Mirroring the Opposite Arc

If a valid symmetry axis is found:

1. Reflect the gap endpoints (`gap_start`, `gap_end`) across the axis to find the "opposite" region of the contour.
2. Extract the continuous arc of points on the contour between those opposite endpoints.
3. Reflect that extracted arc across the axis. It should perfectly span the original gap.
4. Snap the first and last points exactly to `gap_start` and `gap_end`.

### 4.3 Conversion to Bézier

The mirrored polyline is resampled (max 80 points) and converted to a chain of cubic Bézier segments (roughly 4 points per segment) via chord-based tangent fitting.

If the generated mirrored curve is excessively long (> 3× the gap distance or > 45% of total perimeter), it is discarded as an artifact.

---

## 5. Curvature-Aware Arc (Fallback)

If symmetry is not detected, or mirroring fails, the gap is closed using a single curvature-aware Bézier segment.

Unlike standard G1 continuation, the handles are pulled inward into the gap to create an arc:

```python
chord = norm(gap_end - gap_start)

# Tangents point INTO the gap
t_start = tangent at path end
t_end = tangent at path start (reversed)

# Handles constrained by local curvature
alpha_0 = min(chord / 3.0, 1.0 / (3.0 * k_start))
alpha_3 = min(chord / 3.0, 1.0 / (3.0 * k_end))

P1 = gap_start + alpha_0 * t_start
P2 = gap_end + alpha_3 * t_end
```

---

## 6. Semantic Plausibility Validation

Because closing macro-gaps is risky, every proposed EFD closure must pass a plausibility check before synthesis.

**Metrics Computed:**
- `bilateral_alignment`: How well gap tangents point toward each other
- `continuation_score`: Overall forward flow into the gap
- `misalignment_deg`: Angle between opposing tangents
- `curvature_coherence`: `1.0 - |k_start - k_end| / (k_start + k_end)`
- `plausibility_score`: Weighted sum of continuation (45%), bilateral (35%), and curvature coherence (20%), modulated by tangent confidence.

**Rejection Gates:**
- **Tiny Gap Bypass:** Gaps ≤ 3px automatically pass.
- **Long Gap Risk:** Gaps > 90px with ratio > 16% and poor tangents are hard-rejected.
- **Symmetry Rescue:** If symmetry was detected, lower scores (≥ ~0.40) are accepted.
- **Standard Threshold:** Defaults to 0.50 score.
- **Weak Support:** Hard rejection if bilateral < 0.05 and misalignment > 120°.

If the closure is rejected, the path remains open.

---

## 7. Entry Point

```python
def close_single_gaps(
    paths: List[BezierPath],
    efd_contours: List[dict],
    gap_threshold: float = 0.30,
    validity_check_enabled: bool = True,
    plausibility_threshold: float = 0.50,
    return_diagnostics: bool = False,
) -> Union[List[BezierPath], Tuple[List[BezierPath], List[Dict[str, Any]]]]:
```

---

## 8. Dependencies

| Library | Usage |
|---|---|
| `scipy.spatial.distance.directed_hausdorff` | Assessing reflection symmetry |
| `numpy` | Polyline manipulation, rotation matrices |
| `math` | Trig functions |
