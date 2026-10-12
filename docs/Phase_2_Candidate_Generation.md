# Phase 2 — Candidate Generation

> **Source file:** [`restoration/candidates.py`](../restoration/candidates.py)

---

## 1. Purpose

Phase 2 proposes every bridge that could plausibly redraw a missing stroke: a connection between two free stroke ends found in Phase 1. It proposes generously and judges nothing. Whether a candidate is any good is decided in Phase 3 by a fitted scorer, not here by geometric gates.

Earlier versions of this phase rejected pairs with hand-set direction gates, spur rules and extension guards. Each gate fixed some drawings and broke others, and a pair rejected here could never be recovered. They are gone.

---

## 2. Pipeline Overview

```
ExtractionResult (free ends)
    │
    ├─► Pair the free ends
    │       ├─► each end with its nearest partners within reach (KD-tree)
    │       └─► the two ends of one stroke, whenever they are within reach
    │
    └─► For each pair
            ├─► smooth bridge  (always)
            └─► corner bridge  (when the two tangent rays meet ahead of both ends)
                    │
                    └─► ConnectionCandidate, unscored
```

---

## 3. Pairing

```python
endpoint_pairs(result, reach_fraction=0.30, partners_per_endpoint=6)
```

Two free ends are a pair when one is among the other's **6 nearest** free ends and they are no farther apart than **30% of the image diagonal**. The two ends of a single stroke are always a pair if they are within reach, so a broken contour can be closed however many other ends lie nearer.

Ends on junctions are not free ends and are never paired (see Phase 1).

With 4 or 8 partners instead of 6 the benchmark results do not change materially (see the README), so this is a limit on work, not a tuned gate.

---

## 4. Bridge Shapes

A pair of ends is proposed in up to two shapes. Each is a separate candidate; a stroke end can be used by one accepted bridge only, so at most one of them is drawn.

### 4.1 Smooth bridge

One cubic Bézier that leaves end A along its outward tangent and enters end B against B's outward tangent (G1 continuity at both ends):

```python
cp = g1_control_points(ep_a.position, ep_a.tangent, ep_a.curvature,
                       ep_b.position, ep_b.tangent, ep_b.curvature)
```

- Handles are a third of the gap long, shortened to a third of the radius of curvature at an end whose stroke is curved.
- `_stabilize_bridge_control_points` keeps each handle pointing forward along the gap and limits its sideways part to 60% of the gap, which prevents hooks and backtracking.

### 4.2 Corner bridge

Two straight segments: from end A to the point where the two outward tangent rays meet, and from there to end B.

```python
corner = tangent_ray_intersection(ep_a, ep_b, max_leg=2.0 * distance)
```

The corner is proposed only when the rays meet ahead of both ends, no farther from either end than twice the gap length. This is the right shape where two straight strokes met at an angle and the corner was erased; a smooth bridge would cut such a corner.

---

## 5. Key Data Structure

### ConnectionCandidate

```python
@dataclass
class ConnectionCandidate:
    id: int
    ep_a: EndpointInfo
    ep_b: EndpointInfo
    shape: str                    # "smooth" | "corner"
    bridge_points: np.ndarray     # (N, 2) sampled bridge, from ep_a to ep_b
    bridge_bezier: List[BezierSegment]
    distance: float               # straight distance between the two ends, in pixels
    same_path_closure: bool       # the two ends belong to one stroke
    corner_point: Optional[np.ndarray]   # where a corner bridge turns
    # Filled in by Phase 3:
    features: Dict[str, float]         # its measurements, in natural units
    contributions: Dict[str, float]    # log-odds each measurement adds
    probability: float                 # chance that it redraws a missing stroke
    score: float                       # log-odds above the acceptance threshold
```

The bridge is built here once. What Phase 3 scores is exactly what Phase 5 draws.

---

## 6. Entry Point

```python
def generate_candidates(
    result: ExtractionResult,
    reach_fraction: float = REACH_DIAGONAL_FRACTION,        # 0.30
    partners_per_endpoint: int = PARTNERS_PER_ENDPOINT,     # 6
) -> List[ConnectionCandidate]:
```

---

## 7. Constants

| Constant | Value | Meaning |
|---|---|---|
| `REACH_DIAGONAL_FRACTION` | 0.30 | farthest partner of a free end, as a share of the image diagonal |
| `PARTNERS_PER_ENDPOINT` | 6 | nearest partners each free end is paired with |
| `CORNER_MAX_LEG_GAPS` | 2.0 | a corner farther from either end than this many gap lengths is not proposed |

---

## 8. Dependencies

| Library | Usage |
|---|---|
| `scipy.spatial.cKDTree` | nearest free ends |
| `numpy` | geometry |
