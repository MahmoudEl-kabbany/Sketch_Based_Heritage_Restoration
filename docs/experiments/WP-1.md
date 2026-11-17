# WP-1 — Docs/code consistency and the Phase 6 rename

Branch `wp1-docs-and-rename` (from `wp0-safety-net`). Two commits: docs + display text, then identifiers.

## Doc items (first-draft table, checked against `accuracy_overhaul`)

| # | Item | Finding | Action |
|---|---|---|---|
| 1 | Junction continuation threshold | Doc: `cos ≥ 0.5`. Code: `junction_min_alignment = -0.15` (≈ 99°). Leftover edges are walked in a second pass with no threshold | `Phase_1` §3.5 fixed |
| 2 | Unit of `max_error` | Doc: pixels². Code: pixels, squared before `_fit_cubic_single` | `Phase_1` §3.7 fixed; fitting steps rewritten in code order (near-straight → corner → straight preference → accept/reparameterize → split) with the real bounds |
| 3 | Tangent lookahead | Doc: 10. Pipeline: 8 (`fit_from_image_skeleton` default) | `Phase_1` §3.7 fixed |
| 4 | Spur threshold | Already documented correctly (radius-based rules, arc length) | none |
| 5 | Free-end trim | Documented; the cap at a quarter of the chain was missing | `Phase_1` §3.6 completed |
| 6–10, 13 | Tier-2 rule, crossing filter, spur definition, per-endpoint cap, rejection names, utility pseudocode | Code removed on this branch (verified with `git grep`) | closed as obsolete |
| 11 | Phase 6 | Hausdorff is symmetric (max of both directions), threshold max(8 px, 1.5% perimeter). The rescue threshold is max(0.45, 0.8 × threshold). Also wrong in the doc: the mirrored-arc length rule (rejected only above **both** 3× gap and 45% perimeter), the segment count (≤ 10, not "4 points each"), and the small-gap exception (skips symmetry and validation). Missing from the doc: the tiny-smooth-gap threshold relief and `borderline_continuation_pass` | `Phase_6` doc rewritten from the code |
| 12 | `restore_batch` workers | Already documented | none |
| — | `candidate_id` on `BezierSegment` | `_change_events` matches segments by endpoints in either order (`_segment_key`). Over all 64 benchmark images: 804 bridge events, 0 without a candidate, with Phase 6 off and on | not needed, not added |
| — | Plan WP-3 H1 | `stroke_geometry()` does not blur at all (speck removal only) | plan corrected |

## Rename: "Symmetry-guided single-gap closure"

- **Docs:** `docs/Phase_6_EFD_Closure.md` → `docs/Phase_6_Single_Gap_Closure.md`, plus README, Phase 1 and Phase 7.
- **Display:** module docstring, event `type` `"Single-Gap Closure"`, explanation text, console prints, the `restore()` docstring.
- **Identifiers** (approved):
  - `restoration/single_gap_closure.py`. `restoration/efd_closure.py` is a re-export shim that emits a `DeprecationWarning`.
  - `restore()` arguments `gap_closure`, `gap_closure_threshold`, `gap_closure_validity_check`,
    `gap_closure_plausibility_threshold` and `gap_closure_min_gap_for_validity_check`. The old `efd_*` names are mapped
    by `_accept_renamed_kwargs` with a `DeprecationWarning`. Passing both an old and a new name is a `TypeError`.
  - Segment `source_type` `"gap_closure"` (`CHANGE_SOURCES`, `render_restored_sketch`).
  - Report keys: see the mapping in [Phase 7](../Phase_7_Output.md#schema-40-renamed-keys). `schema_version` is now `4.0`.
  - No evaluation or training script read the old keys. `evaluate_gaps.py` passes the source type through, so it
    now writes `gap_closure`.
- **New test:** `test_old_phase6_names_still_work` (aliases, warning, conflict error, shim).

## Verification

| Check | Result |
|---|---|
| `pytest test_restoration.py test_regression.py` | 12 passed |
| Snapshot (64 images × default/Phase 6) | identical, no rebless |
| `check_changes.py` | UNCHANGED: 87.2% recall, 89.4 line widths off the drawing, 5 wrong strokes, 23 on undamaged, PSNR 26.06 dB, SSIM 0.9830 |

Timings are unaffected (text and naming only).
