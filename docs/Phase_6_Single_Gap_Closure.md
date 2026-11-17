# Phase 6 — Symmetry-Guided Single-Gap Closure

> **Source file:** [`restoration/efd_closure.py`](../restoration/efd_closure.py)

---

> **Optional, off by default.** Run it with `restore(..., efd_closure=True)`.
> On the 32 test drawings it repaired 5 more gaps of 784 (+0.6 points of gap recall) and added 5% to the line
> drawn off the drawing; in nearby settings its closures were the longest wrong strokes of the whole pipeline
> (in one such run two closures drew 137 line widths off the drawing, all 773 bridges together 58). Its plausibility
> check is the one part of the system still built on hand-set thresholds. Since a wrong stroke is worse than an
> open gap, it is not run unless asked for. A single-gap contour whose two ends face each other is in any case
> a candidate of Phases 2 to 4 like any other pair of ends.

> **About the name.** This phase used to be called "EFD Gap Closure". It computes no Elliptic Fourier
> Descriptors: it closes a gap by mirroring the contour across a reflection-symmetry axis, or with a
> curvature-aware G1 arc. The module and its parameters (`efd_closure.py`, `efd_closure=...`) still carry the
> old name. The standalone demo in `eliptic_fourier_descriptors/efd.py` does compute EFDs and is unrelated to this phase.

## 1. Purpose

Phase 6 addresses open paths that form nearly-closed shapes with a **single remaining gap**: the path's own start and
end. Phases 2–4 treat such a pair of ends like any other, but may leave it open when the ends are far apart or do not
face each other. Phase 6 instead looks at the shape of the whole contour, through its reflection symmetry and the
curvature at its ends, to synthesize a closing curve.

---

## 2. Pipeline Overview

```
List[BezierPath] (after Phase 5)
    │
    ├─► For each open path whose two ends are both free and which has ≥ 3 segments:
    │       ├─► Skip if a Phase 5 bridge on the path already spans this gap
    │       ├─► Skip if gap / perimeter > gap_threshold (30%), unless it is a small gap (§3)
    │       │
    │       ├─► Symmetry detection (only for 8% ≤ gap ratio ≤ 30%)
    │       │       └─► 24 reflection axes; keep the one with the lowest symmetric Hausdorff distance
    │       │
    │       ├─► Plausibility validation (skipped for small gaps)
    │       │
    │       ├─► Synthesis
    │       │       ├─► Symmetry mirroring (if an axis was found and the mirrored arc is not too long)
    │       │       └─► Curvature-aware G1 arc (otherwise)
    │       │
    │       └─► Path marked is_closed=True
    │
    └─► Final List[BezierPath] + one diagnostics entry per validation decision
```

---

## 3. Qualification Criteria

For an open path to be considered a single-gap contour:

1. **Shape:** it is open, both of its ends are free (not on a junction), and it has at least 3 segments and a non-zero perimeter (arc length of the path sampled at 30 points per segment).
2. **Not already bridged:** the path is skipped if one of its `bridge` segments has endpoints whose summed distance to the two gap ends is ≤ 2 × max(8 px, 2% of the perimeter).
3. **Gap ratio:** `gap = |end − start|`, `ratio = gap / perimeter` must be ≤ `gap_threshold` (default 0.30).
   - **Small-gap exception:** a gap above the threshold is still closed if gap ≤ 20 px, ratio ≤ 0.45 and perimeter ≥ 80 px. Such a gap skips symmetry detection *and* the plausibility check and is always closed with the curvature-aware arc.

---

## 4. Symmetry Detection and Mirroring

### 4.1 Axis Detection (`_detect_symmetry`)

Attempted only when 8% ≤ ratio ≤ 30% (and not for the small-gap exception).

1. Compute the centroid of the sampled contour.
2. Subsample the contour to at most 500 points.
3. Test 24 axes through the centroid, 0° to 180° in 7.5° steps.
4. For each axis, reflect the points and compute the **symmetric Hausdorff distance**: the larger of the two directed Hausdorff distances (original → reflected and reflected → original).
5. The axis with the smallest distance is selected if that distance is below max(8 px, 1.5% of the perimeter).

### 4.2 Mirroring the Opposite Arc

1. Reflect the gap ends across the axis to find the "opposite" region of the contour.
2. Take the contour points between the samples nearest to those two reflected points.
3. Reflect that arc back across the axis; it spans the gap. Orient it from gap start to gap end and snap its first and last points onto them.

### 4.3 Conversion to Bézier

The mirrored polyline is resampled to at most 80 points and converted to at most 10 cubic segments (`max(6, 80 // 8)`), each spanning an equal run of points, with handles a third of the chord along the local direction.

The mirrored closure is discarded, and the arc of §5 used instead, when it is longer than **both** 3 × the gap and 45% of the perimeter, i.e. when its length exceeds `max(3 × gap, 0.45 × perimeter)`.

---

## 5. Curvature-Aware Arc (Fallback)

If no symmetry axis was found, or mirroring failed, the gap is closed by one cubic built by `geometry.g1_control_points`:

```python
chord = norm(gap_end - gap_start)

# Outward tangents of the two ends, so both handles point into the gap.
# Each is a blend of the end segment's derivative and the mean direction of the
# last ~12% of the path, weighted by that direction's consistency (0.15–0.90).
alpha_0 = min(chord / 3, 1 / (3 * k_start))   # k = curvature at the end (ignored if ≈ 0)
alpha_3 = min(chord / 3, 1 / (3 * k_end))

P1 = gap_start + alpha_0 * t_start
P2 = gap_end   + alpha_3 * t_end
```

---

## 6. Plausibility Validation

When `validity_check_enabled` (default) and the gap is not a small-gap exception, each proposed closure is checked before synthesis.

**Metrics** (`_closure_plausibility_metrics`):
- `forward_start`, `forward_end`: how far each end's tangent points across the gap
- `bilateral_alignment`: the smaller of the two
- `continuation_score`: mean of their positive parts
- `misalignment_deg`: angle between one tangent and the reverse of the other
- `curvature_coherence`: `1 − |k_start − k_end| / (k_start + k_end)` (1 when both are ≈ 0)
- `context_confidence`: the lower consistency of the two end directions
- `plausibility_score`: `(0.45·continuation + 0.35·bilateral + 0.20·curvature_coherence) × (0.60 + 0.40·context_confidence)`
- `long_gap_semantic_risk`: gap > 90 px, ratio > 0.16, score < 0.62, and (bilateral < 0.12 or misalignment > 95°)

**Decision** (`_evaluate_closure_validity`), first match wins:

| Reason | Condition | Result |
|---|---|---|
| `tiny_gap_bypass` | gap ≤ `min_gap_for_validity_check` (3 px) | accept |
| `long_gap_semantic_risk` | the flag above | reject |
| — | ratio ≤ 0.035, gap ≤ 45 px, continuation ≥ 0.10, misalignment ≤ 35° | the threshold below is lowered to min(threshold, 0.24) |
| `symmetry_supported` | an axis was found, bilateral ≥ 0.20, misalignment ≤ 110°, score ≥ max(0.45, 0.8 × threshold) | accept |
| `plausibility_pass` | score ≥ threshold (default 0.50) | accept |
| `borderline_continuation_pass` | ratio ≤ 0.16, score ≥ max(0.42, 0.85 × plausibility_threshold), bilateral ≥ 0.16, continuation ≥ 0.35, misalignment ≤ 132° | accept |
| `weak_continuation_support` | bilateral < 0.05 and misalignment > 120° | reject |
| `low_plausibility_score` | otherwise | reject |

A rejected closure leaves the path open.

---

## 7. Entry Point

```python
def close_single_gaps(
    paths: List[BezierPath],
    gap_threshold: float = 0.30,
    validity_check_enabled: bool = True,
    plausibility_threshold: float = 0.50,
    min_gap_for_validity_check: float = 3.0,
) -> Tuple[List[BezierPath], List[Dict[str, Any]]]:  # (paths, diagnostics)
```

Each diagnostics entry holds `path_index`, `accepted`, `closure_method` (`symmetry_mirroring`, `curvature_arc` or `none`), `reason`, `gap_distance_px`, `gap_ratio`, `plausibility_threshold` and the metrics above. Closure segments have `source_type = "efd_closure"`.

---

## 8. Dependencies

| Library | Usage |
|---|---|
| `scipy.spatial.distance.directed_hausdorff` | Reflection symmetry (both directions) |
| `numpy` | Polyline manipulation, reflections |
| `math` | Trig functions |
