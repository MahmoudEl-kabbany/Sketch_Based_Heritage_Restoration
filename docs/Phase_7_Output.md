# Phase 7 — Output & Reporting

> **Source file:** [`restoration/pipeline.py`](../restoration/pipeline.py)

---

## 1. Purpose

Phase 7 is the final stage of the restoration pipeline. It orders the final paths, explains every change the restoration made and every stroke end it left open, saves three images, and writes a JSON report.

---

## 2. Pipeline Overview

```
Final paths + accepted candidates + Phase 6 diagnostics
    │
    ├─► Sort final paths (top-down, then left-right)
    │
    ├─► Explain each restoration change           _change_events()
    │       └─► one entry per run of bridge / EFD-closure segments: R1, R2, ...
    ├─► Explain each free end left open           _open_ends()
    │
    ├─► Save labeled overlay                      _save_overlay(labeled=True)
    ├─► Save unlabeled overlay                    _save_overlay(labeled=False)
    ├─► Save 3-panel comparison                   _save_visualization()
    │
    ├─► Build and save the JSON report            _build_report()
    │
    └─► Return RestorationResult
```

A *change* is a contiguous run of segments of one restoration type (`bridge` or `efd_closure`) inside a final path. `_iter_changes()` yields them in path order; the same numbering is used for the labels in the image and the entries in the report.

---

## 3. Post-Processing & Sorting

So that labels and indices are stable, the final paths are sorted top-down, then left-right.

```python
# restoration/geometry.py
def visual_order_key(path: BezierPath):
    pts = path.sample(5)
    # Group by 12-pixel vertical bands, then sort horizontally
    return (round(np.min(pts[:, 1]) / 12.0), np.min(pts[:, 0]))

final_paths.sort(key=visual_order_key)
```

---

## 4. Output Files

For an input `name.png`, the pipeline writes four files to the output directory (`restoration/outputs/` by default):

| File | Written by | Content |
|---|---|---|
| `name_labeled_overlay.png` | `_save_overlay(labeled=True)` | The input image with every restoration change drawn in green and labeled `R1`, `R2`, ... Labels are placed beside the change and moved if they would overlap an earlier label. |
| `name_overlay.png` | `_save_overlay(labeled=False)` | The same green changes on the input image, without labels. |
| `name_restoration.png` | `_save_visualization()` | Three panels: the input image, the extracted paths (red), and the final paths (green for paths that were restored, teal for untouched ones). |
| `name_report.json` | `restore()` | The report described below. |

The black-on-white `name_restored.png` used for PSNR/SSIM is **not** written by the pipeline. `evaluate_restoration.py` renders it by drawing the restoration changes onto the damaged image (`render_bridges_on_damaged`).

---

## 5. Report

The report is the dictionary returned by `_build_report()`. It is saved as `name_report.json` and is also available as `RestorationResult.report`.

### Structure

```json
{
  "schema_version": "3.0",
  "image": { "name": "da9_gt_damaged.png", "shape": [224, 310], "diagonal_px": 382.5, "line_width_px": 4.0 },
  "summary": {
    "processing_time_s": 1.741,
    "stage_timings_s": { "phase1_extraction_s": 0.39, "...": "one entry per phase", "total_pipeline_s": 1.741 },
    "total_restoration_events": 18,
    "open_paths_before": 38,
    "open_paths_after": 20,
    "closed_paths_before": 0,
    "closed_paths_after": 0,
    "net_open_path_change": -18,
    "free_ends_left_open": 7
  },
  "analysis": {
    "extraction": { "total_paths": 38, "open_paths": 38, "closed_paths": 0, "endpoints": 43 },
    "candidate_generation": {
      "total": 182,
      "shape_distribution": { "smooth": 155, "corner": 27 },
      "probability_summary": { "min": 0.0, "max": 0.9923, "mean": 0.2043 },
      "above_threshold": 28
    },
    "scoring_model": {
      "threshold": 0.6667,
      "scorecards": {
        "smooth": {
          "intercept": -5.5766,
          "features": {
            "gap_length": { "meaning": "gap length in stroke widths", "transform": "log2(x)",
                            "weight": -0.7892, "mean": 3.7627, "scale": 1.1899 },
            "...": "one entry per measurement on the scorecard"
          }
        },
        "corner": { "...": "the same for corner bridges" }
      }
    },
    "selection": {
      "accepted_total": 18,
      "acceptance_rate_percent": 9.9,
      "accepted_candidate_ids": [59, 73, 17],
      "accepted_endpoint_pairs": ["id:10|id:11", "id:13|id:17"],
      "accepted_shapes": { "smooth": 8, "corner": 10 },
      "accepted_probability_summary": { "min": 0.9258, "max": 0.9923, "mean": 0.9734 }
    },
    "asp_solver": {
      "fact_lines": 28,
      "fact_bytes": 668,
      "accepted_id_count_before_decode": 18,
      "candidate_count_input": 182,
      "candidate_count_above_threshold": 28,
      "encoding_time_s": 0.00002,
      "solving_time_s": 0.019,
      "total_solve_s": 0.02,
      "control_args": ["--models=1", "--opt-mode=optN", "--opt-strategy=usc"]
    },
    "restoration_outcome": {
      "final_paths": 20,
      "final_open_paths": 20,
      "final_closed_paths": 0,
      "bridges_created": 18,
      "bridge_events_logged": 18,
      "efd_closure_events_logged": 0
    },
    "efd_closure_phase": { "enabled": false, "attempted": 0, "accepted": 0, "rejected": 0, "...": "..." }
  },
  "detailed_events": [ "... one entry per change, see below ..." ],
  "open_ends": [ "... one entry per free stroke end left open, see below ..." ]
}
```

The report carries the scorecards it was scored with, so every probability in it can be recomputed by hand from the measurements listed for a bridge.

### Detailed events

Each entry of `detailed_events` explains one change and carries the same `id` as its label in the labeled overlay.

```json
{
  "id": "R2",
  "source": "asp_bridge",
  "type": "ASP Junction Bridge",
  "coordinates": [46.7, 64.1],
  "segment_count": 2,
  "geometry": { "approx_length_px": 16.73, "sample_points": 60 },
  "candidate": {
    "id": 73,
    "shape": "corner",
    "endpoint_a": { "path_index": 14, "end": "end", "endpoint_id": 13 },
    "endpoint_b": { "path_index": 18, "end": "start", "endpoint_id": 17 },
    "probability": 0.9902,
    "threshold": 0.6667,
    "log_odds_above_threshold": 3.918,
    "distance_px": 16.64,
    "same_path_closure": false,
    "in_favour": [
      { "feature": "crossed_ink", "meaning": "existing stroke the bridge runs over between its ends, in stroke widths",
        "value": 0.0, "log_odds": 1.83 },
      { "feature": "gap_length", "meaning": "gap length in stroke widths", "value": 3.788, "log_odds": 1.72 },
      { "feature": "corner_turn", "meaning": "change of direction at the corner, in degrees", "value": 12.13, "log_odds": 1.7 }
    ],
    "against": [],
    "measurements": {
      "gap_length": { "value": 3.788, "log_odds": 1.723 },
      "crossed_ink": { "value": 0.0, "log_odds": 1.831 },
      "...": "every measurement on the scorecard of this shape"
    },
    "alternatives": [
      { "id": 72, "shape": "smooth", "endpoint_a": { "...": "..." }, "endpoint_b": { "...": "..." }, "probability": 0.9748 },
      { "id": 53, "shape": "smooth", "endpoint_a": { "...": "..." }, "endpoint_b": { "...": "..." }, "probability": 0.4766 }
    ],
    "corner_point": [46.7, 64.11]
  },
  "explanation": "R2 joins the end of path 14 to the start of path 18 with a corner bridge (candidate C73) across a gap of 16.6px, 3.8 stroke widths, drawing about 16.7px of line. The estimated probability that it redraws a missing stroke is 0.99; bridges are drawn above 0.67. In its favour: existing stroke the bridge runs over between its ends, in stroke widths = 0 (+1.8); gap length in stroke widths = 3.79 (+1.7); change of direction at the corner, in degrees = 12.1 (+1.7). Against it: no single measurement stands out. It was preferred to 12 other candidate(s) for these two stroke ends; the most likely of them, C72, has probability 0.97."
}
```

- `source` is `asp_bridge` for a bridge chosen in Phase 4 and `efd_gap_closure` for a Phase 6 closure.
- `probability` is the scorecard's estimate that the bridge redraws a missing stroke; `log_odds_above_threshold` is what the solver maximized.
- `measurements` lists every measurement of the bridge in its natural unit with the log-odds it adds relative to an average candidate. Their sum plus the scorecard's intercept is the bridge's log-odds.
- `in_favour` and `against` are the up to three measurements that move the log-odds most in each direction (by at least 0.25).
- `alternatives` are the three most probable other candidates for the same two stroke ends: what the bridge was preferred to.
- An `efd_gap_closure` event has `efd_closure_validity` (closure method, reason, gap size and plausibility metrics) instead of `candidate`.

### Open ends

Each entry of `open_ends` is a free stroke end that no bridge was attached to, with its most probable candidate and the reason it was not drawn.

```json
{
  "endpoint": { "path_index": 0, "end": "end", "endpoint_id": 0 },
  "coordinates": [10.0, 153.0],
  "best_candidate": { "id": 1, "shape": "corner", "endpoint_a": { "...": "..." }, "endpoint_b": { "...": "..." }, "probability": 0.1582 },
  "against": [
    { "feature": "gap_length", "meaning": "gap length in stroke widths", "value": 15.98, "log_odds": -0.7 },
    { "feature": "corner_turn", "meaning": "change of direction at the corner, in degrees", "value": 106.3, "log_odds": -0.46 }
  ],
  "reason": "Its most likely bridge, C1 to the end of path 18, has probability 0.16, not above the 0.67 needed to draw it. Against it: gap length in stroke widths = 16 (-0.7); change of direction at the corner, in degrees = 106 (-0.5); ..."
}
```

The reason is one of:

- no other free stroke end lies within reach;
- the end's most probable bridge is not above the acceptance threshold (with the measurements that count against it);
- that bridge needs a stroke end which a more plausible bridge uses.

Most open ends of a drawing are simply where its strokes end. The list is there so that a missed repair can be traced to its cause.

---

## 6. Return Object

`restore()` returns a `RestorationResult` holding the Python objects directly, so downstream code does not need to parse the JSON or the images.

```python
@dataclass
class RestorationResult:
    image_path: str
    original_paths: List[BezierPath]
    restored_paths: List[BezierPath]
    bridges: List[BezierSegment]
    report: dict
```

`restore_batch()` processes a list of images, in parallel if `workers` is given, and returns a `List[RestorationResult]`.

---

## 7. Dependencies

| Library | Usage |
|---|---|
| `matplotlib` | Overlay and comparison images (non-interactive `Agg` backend) |
| `opencv-python` | Reading the input image, drawing the path panels |
| `json` | Report dumping |
| `time.perf_counter` | Per-phase timing |
