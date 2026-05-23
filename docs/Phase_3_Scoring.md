# Phase 3 — Scoring

> **Source file:** [`restoration/scoring.py`](../restoration/scoring.py)

---

## 1. Purpose

Phase 3 assigns a composite quality score to each ConnectionCandidate produced in Phase 2. The score combines:

- **Gestalt perceptual heuristics** (proximity, good continuation, closure)
- **Smoothness metrics** (Dynamic Time Warping, mean squared jerk)
- **Contextual adjustments** (spur suppression, self-closure priority, shape affinity)

The final score determines each candidate's competitiveness in Phase 4's ASP optimization.

---

## 2. Pipeline Overview

```
List[ConnectionCandidate]  +  ExtractionResult
    │
    ├─► Compute raw MDTW for each candidate
    ├─► Compute raw MJerk for each candidate
    ├─► Min-max normalize MDTW and MJerk across all candidates
    │
    ├─► For each candidate:
    │       ├─► Gestalt proximity score
    │       ├─► Gestalt continuation score
    │       ├─► Gestalt closure score
    │       ├─► Weighted composite score
    │       └─► Context adjustments (alignment, spur, extension)
    │
    ├─► Self-closure suppression of cross-path competitors
    ├─► Shape affinity boost for same-shape fragments
    │
    └─► Sort candidates by score (best first)
```

---

## 3. Core Metrics

### 3.1 MDTW — Modified Dynamic Time Warping

**Purpose:** Measures how much the bridge deviates from a direct straight-line connection. Lower MDTW = more direct, natural bridge.

**Computation:**

1. **Sample context:** Take the last 20 points from the source path (tail) and first 20 points from the target path (head)
2. **Combine:** `combined = [source_tail, bridge_points, target_head]`
3. **Baseline:** A straight line from the first to last point of `combined`, sampled at the same number of points
4. **DTW distance:** Computed independently for x and y coordinates using the `dtaidistance` library
5. **Final metric:** `MDTW = √(dtw_x² + dtw_y²)`

```python
def _compute_mdtw(bridge_pts, source_path, target_path, tail_len=20):
    combined = vstack([source_tail, bridge_pts, target_head])
    baseline = linspace(combined[0], combined[-1], len(combined))
    dtw_x = dtw.distance(combined[:, 0], baseline[:, 0])
    dtw_y = dtw.distance(combined[:, 1], baseline[:, 1])
    return hypot(dtw_x, dtw_y)
```

**Interpretation:**
- MDTW ≈ 0: Bridge goes almost perfectly straight
- High MDTW: Bridge meanders, deviates, or takes an indirect path

### 3.2 MJerk — Mean Squared Jerk

**Purpose:** Quantifies the smoothness of the combined curve. Human-drawn strokes have characteristically low jerk. Lower MJerk = smoother, more natural transition.

**Computation:**

Jerk is the third derivative of position with respect to arc length, computed via finite differences:

```
Arc-length parameterization:
    ds = ||Δposition||

Velocity (1st derivative):
    v = Δposition / ds

Acceleration (2nd derivative):
    a = Δv / ds

Jerk (3rd derivative):
    j = Δa / ds

MJerk = mean(||j||²)
```

```python
def _compute_mjerk(bridge_pts, source_path, target_path, tail_len=20):
    combined = vstack([source_tail, bridge_pts, target_head])
    diffs = diff(combined, axis=0)
    ds = norm(diffs, axis=1)
    vel = diffs / ds[:, newaxis]
    acc = diff(vel) / ds[:-1, newaxis]
    jerk = diff(acc) / ds[:-2, newaxis]
    return mean(sum(jerk ** 2, axis=1))
```

**Interpretation:**
- Low MJerk: Smooth, flowing transition (like human drawing)
- High MJerk: Abrupt curvature changes, sharp kinks

### 3.3 Normalization

Both MDTW and MJerk are min-max normalized across all candidates to [0, 1]:

```python
normalized[i] = (raw[i] - min(raw)) / (max(raw) - min(raw))
```

This ensures both metrics contribute proportionally regardless of their absolute scale.

---

## 4. Gestalt Heuristics

### 4.1 Proximity

```python
proximity = max(0, 1 - distance / radius)
```
- **1.0** when endpoints are touching
- **0.0** at the edge of the search radius
- Radius is Tier 1 (15% diagonal) or Tier 2 (30% diagonal)

### 4.2 Good Continuation

```python
direction_ab = ep_b.position - ep_a.position
continuation = dot(ep_a.tangent, normalize(direction_ab))
```
- **1.0** when the tangent points exactly toward the partner
- **0.0** when perpendicular
- **-1.0** when pointing away (clamped to 0 in score formula)

### 4.3 Closure

```python
closure = 1.0 if connecting would close a loop, else 0.0
```

Loop detection heuristic: If the *other* endpoints of both participating paths are within 15 pixels of each other, connecting these endpoints would complete a closed shape.

---

## 5. Composite Score Formula

### Weight Tables

| Metric | Tier 1 Weight | Tier 2 Weight |
|---|---|---|
| Proximity | 0.20 | 0.05 |
| Continuation | 0.30 | 0.40 |
| Closure | 0.15 | 0.15 |
| DTW (cost) | 0.20 | 0.25 |
| Jerk (cost) | 0.15 | 0.15 |

**Rationale:** Tier 2 candidates are farther away, so proximity matters less. Continuation and DTW become more important because longer bridges must demonstrate stronger directional evidence.

### Base Score

```python
score = (
    weights["proximity"] × proximity
    + weights["continuation"] × max(0, continuation)
    + weights["closure"] × closure
    - weights["dtw"] × normalized_dtw
    - weights["jerk"] × normalized_jerk
)
```

Note: DTW and jerk are **subtracted** because they are costs (lower = better).

---

## 6. Context Adjustments

### 6.1 Bilateral Alignment Bonus
```python
score += 0.16 × max(0, bilateral_alignment)
```
Rewards candidates where **both** tangents point toward each other.

### 6.2 Self-Closure Bonus
```python
if same_path_closure:
    score += 0.20
```
Same-path closures receive a significant boost because closing a loop is geometrically well-defined.

### 6.3 Spur Penalty
```python
if spur_involved:
    score -= 0.22
```
Connections involving very short paths (potential noise spurs) are penalized.

### 6.4 High Misalignment Penalty
```python
if misalignment_deg > 70:
    score -= 0.08 × min(1, (misalignment_deg - 70) / 60)
```
Gradual penalty for tangent misalignment beyond 70°.

### 6.5 Weak Tier 2 Bilateral Penalty
```python
if tier == 2 and bilateral_alignment < 0.25:
    score -= 0.06
```

### 6.6 Extension-Intersection Adjustments

For extension-intersection candidates, additional logic:

**Same-path extension:**
```python
corner_factor = clip((misalignment - 55) / 60, 0, 1)
extension_need = 1 - continuation_strength
context_conf = min(tangent_confidence_a, tangent_confidence_b)
score += (0.26 + 0.10 × corner_factor) × ext_quality × extension_need² × context_conf
```
- Favors extension bridges at sharp corners where continuation is weak
- Modulated by tangent confidence to prevent false extensions with unreliable tangents
- Suppressed when continuation strength > 0.60 (standard continuation suffices)

**Cross-path extension:**
```python
score += 0.04 × extension_quality
```
- Plus a corner-recovery bonus for short, high-quality Tier 1 extensions with 125–165° misalignment
- Minus a crossing penalty for Tier 2 extensions that pass within 3 pixels of other paths

### 6.7 Self-Closure Suppression

When a path has a strong self-closure option (score ≥ 0.60), cross-path candidates touching that path are penalized:

```python
if self_closure_score ≥ 0.60 and cross_distance / self_distance ≥ 0.95:
    cross_score -= 0.18
elif self_closure_score ≥ 0.45 and distance_ratio ≥ 1.02:
    cross_score -= 0.12
```

This prevents cross-path bridges from "stealing" endpoints that should self-close.

### 6.8 Shape Affinity Boost

Paths that are likely fragments of the same shape receive a connection bonus:

```python
affinity = bbox_overlap_ratio × centroid_proximity_factor
if affinity ≥ 0.18 and distance ≤ diagonal × 0.22:
    score += 0.16 × affinity
```

Conversely, when high-affinity options exist for a path, low-affinity competitors are suppressed by -0.10.

---

## 7. Final Sorting

Candidates are sorted by a multi-key comparator (best first):

```python
candidates.sort(key=lambda c: (
    -c.score,                              # Primary: highest score first
    int(not c.same_path_closure),           # Prefer self-closures
    -c.extension_quality,                   # Prefer higher extension quality
    c.distance,                            # Prefer shorter distance
    c.misalignment_deg,                    # Prefer lower misalignment
    -c.bilateral_alignment,                # Prefer higher bilateral
    c.id,                                  # Stable tie-break
))
```

---

## 8. Entry Point

```python
def score_candidates(
    candidates: List[ConnectionCandidate],
    result: ExtractionResult,
) -> List[ConnectionCandidate]:
    # Scores are set in-place on each candidate's .score attribute
    # Returns the list sorted best-first
```

---

## 9. Dependencies

| Library | Usage |
|---|---|
| `dtaidistance` | Dynamic Time Warping distance computation |
| `scipy.spatial.cKDTree` | Path-body proximity checking |
| `numpy` | All vector math, normalization |
