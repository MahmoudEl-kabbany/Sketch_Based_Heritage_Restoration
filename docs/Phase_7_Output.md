# Phase 7 — Output & Reporting

> **Source file:** [`restoration/pipeline.py`](../restoration/pipeline.py)

---

## 1. Purpose

Phase 7 is the final stage of the restoration pipeline. It orders the final paths, describes every change the restoration made, saves three images, and writes a JSON report.

---

## 2. Pipeline Overview

```
Final paths + accepted candidates + Phase 6 diagnostics
    │
    ├─► Sort final paths (top-down, then left-right)
    │
    ├─► Describe each restoration change          _change_events()
    │       └─► one entry per run of bridge / EFD-closure segments: R1, R2, ...
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
  "schema_version": "2.1",
  "image": { "name": "ankh_damaged.png", "shape": [1024, 1024], "diagonal_px": 1448.2 },
  "summary": {
    "processing_time_s": 4.21,
    "stage_timings_s": {
      "phase1_extraction_s": 0.9,
      "phase2_candidate_generation_s": 0.3,
      "phase3_scoring_s": 1.1,
      "phase4_asp_fact_encoding_s": 0.001,
      "phase4_asp_solving_s": 0.02,
      "phase4_asp_decode_s": 0.0,
      "phase4_asp_total_s": 0.021,
      "phase5_synthesis_s": 0.002,
      "phase6_efd_closure_s": 0.01,
      "phase7_output_s": 1.8,
      "total_pipeline_s": 4.21
    },
    "total_restoration_events": 7,
    "open_paths_before": 9,
    "open_paths_after": 2,
    "closed_paths_before": 0,
    "closed_paths_after": 6,
    "net_open_path_change": -7
  },
  "analysis": {
    "extraction": { "total_paths": 9, "open_paths": 9, "closed_paths": 0, "endpoints": 18 },
    "candidate_generation": {
      "total": 33,
      "tier_distribution": { "tier1": 20, "tier2": 13 },
      "scenario_distribution": { "self_closure": 2, "continuation": 19, "extension_intersection": 12 },
      "score_summary": { "min": -0.4, "max": 0.71, "mean": 0.12 },
      "alignment_summary": { "min": -0.2, "max": 1.0, "mean": 0.55 },
      "misalignment_deg_summary": { "min": 0.4, "max": 164.0, "mean": 61.3 },
      "diagnostics": { "...": "rejection statistics from Phase 2, see Phase_2_Candidate_Generation.md" }
    },
    "selection": {
      "accepted_total": 7,
      "dropped_after_endpoint_sanitize": 0,
      "acceptance_rate_percent": 21.2,
      "accepted_candidate_ids": [4, 9, 12],
      "accepted_endpoint_pairs": ["id:0|id:3"],
      "accepted_scenarios": { "continuation": 7 },
      "accepted_score_summary": { "min": 0.31, "max": 0.71, "mean": 0.52 },
      "accepted_alignment_summary": { "min": 0.61, "max": 1.0, "mean": 0.9 },
      "accepted_misalignment_deg_summary": { "min": 0.4, "max": 31.0, "mean": 9.8 }
    },
    "asp_solver": {
      "fact_lines": 132,
      "fact_bytes": 2870,
      "accepted_id_count_before_decode": 7,
      "candidate_count_input": 33,
      "candidate_count_after_pruning": 33,
      "candidate_count_pruned": 0,
      "encoding_time_s": 0.001,
      "solving_time_s": 0.02,
      "total_solve_s": 0.03,
      "control_args": ["--models=1", "--opt-mode=optN", "--opt-strategy=usc"]
    },
    "restoration_outcome": {
      "final_paths": 8,
      "final_open_paths": 2,
      "final_closed_paths": 6,
      "bridges_created": 7,
      "bridge_events_logged": 7,
      "efd_closure_events_logged": 0
    },
    "efd_closure_phase": {
      "validity_check_enabled": true,
      "plausibility_threshold": 0.5,
      "attempted": 1,
      "accepted": 0,
      "rejected": 1,
      "rejection_reasons": { "low_plausibility_score": 1 },
      "plausibility_score_summary": { "min": 0.2, "max": 0.2, "mean": 0.2 },
      "accepted_path_indices": [],
      "rejected_path_indices": [3]
    }
  },
  "detailed_events": [ "... one entry per change, see below ..." ]
}
```

The numbers above are illustrative.

### Detailed events

Each entry of `detailed_events` describes one change and carries the same `id` as its label in the labeled overlay.

```json
{
  "id": "R1",
  "source": "asp_bridge",
  "type": "ASP Junction Bridge",
  "coordinates": [412.5, 96.0],
  "segment_count": 1,
  "geometry": { "approx_length_px": 38.2, "sample_points": 30 },
  "candidate": {
    "id": 12, "scenario": "continuation", "tier": 1,
    "distance_px": 37.9, "score": 0.5312, "extension_quality": 0.0,
    "bilateral_alignment": 0.97, "misalignment_deg": 4.1,
    "same_path_closure": false, "spur_involved": false,
    "endpoint_a": { "path_index": 2, "end": "end", "endpoint_id": 5 },
    "endpoint_b": { "path_index": 4, "end": "start", "endpoint_id": 8 }
  },
  "explanation": "R1 is an ASP-selected bridge from candidate C12. ..."
}
```

- `source` is `asp_bridge` for a bridge chosen in Phase 4 and `efd_gap_closure` for a Phase 6 closure.
- `candidate` is present for bridges whose segments can be traced back to their candidate. A bridge loses that link when the path it sits in is reversed during merging; its explanation says so.
- `candidate.intersection_point` is added for extension-intersection candidates.
- An `efd_gap_closure` event has `efd_closure_validity` (closure method, reason, gap size and plausibility metrics) instead of `candidate`.

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

`restore_batch()` processes a list of images one after another and returns a `List[RestorationResult]`.

---

## 7. Dependencies

| Library | Usage |
|---|---|
| `matplotlib` | Overlay and comparison images (non-interactive `Agg` backend) |
| `opencv-python` | Reading the input image, drawing the path panels |
| `json` | Report dumping |
| `time.perf_counter` | Per-phase timing |
