# Phase 5 — Synthesis

> **Source file:** [`restoration/synthesis.py`](../restoration/synthesis.py)

---

## 1. Purpose

Phase 5 puts the bridges the solver accepted into the drawing: it merges each bridge with the two paths it joins, so that the result is a set of complete, contiguous paths.

It no longer builds any geometry. A candidate's bridge is built once, when the candidate is generated (Phase 2), so that the bridge Phase 3 scored is exactly the bridge that is drawn. Earlier versions scored a preliminary bridge and then drew a different, refined one.

---

## 2. Pipeline Overview

```
Accepted candidates (each with its bridge segments)
    │
    ├─► Collect the bridge segments                 synthesize_bridges()
    │
    └─► Merge path topologies                       merge_restored_paths()
            ├─► Track the free ends of every path as connectable tokens
            ├─► For each bridge:
            │       ├─► Source path + bridge + target path
            │       ├─► Reverse a path if necessary (start-to-start)
            │       └─► Detect if the merged path forms a closed loop
            └─► Output the updated List[BezierPath]
```

---

## 3. Bridge Shapes

Both shapes are built in [`restoration/candidates.py`](../restoration/candidates.py) and described in [Phase 2](Phase_2_Candidate_Generation.md):

- **Smooth bridge** — one cubic Bézier, G1-continuous with both strokes, handles a third of the gap long and shortened where a stroke is curved, stabilized against hooks.
- **Corner bridge** — two straight segments through the point where the two tangent rays meet.

A corner bridge is a run of two `bridge` segments; a smooth bridge is one.

---

## 4. Path Merging Topology

After the bridge segments are created, the discrete paths must be chained together.

1. **Tokenization:** Every free end of every open path is identified by the token `(path_index, end)`. An end on a junction has no token: it cannot be bridged.
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
    # The bridge segments of the accepted candidates, in order

def merge_restored_paths(
    original_paths: List[BezierPath],
    accepted: List[ConnectionCandidate],
) -> List[BezierPath]:
    # Returns the combined final topology
```

---

## 6. Dependencies

| Library | Usage |
|---|---|
| `numpy` | Vector math, Euclidean distance |
