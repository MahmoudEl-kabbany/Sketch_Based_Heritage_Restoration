# WP-2 — Performance

Branch `wp2-performance` (from `wp1-docs-and-rename`). The output is unchanged: the snapshot is identical for all
64 images × 2 variants, and `check_changes.py` reports UNCHANGED (87.2% recall, 89.4 line widths off the drawing,
5 wrong strokes, 23 on undamaged, PSNR 26.06 dB, SSIM 0.9830).

## What WP-0 showed

No stage grows super-linearly: P2 and P3 have k ≈ 0.9–1.0, and P1 depends on image size rather than the number of
ends. The goal of the first draft ("near-linear runtime") was therefore already met. WP-2 only removes overhead
that the profile showed.

## Changes

| Change | Where | Effect | Identical? |
|---|---|---|---|
| MJerk spline: `CubicSpline(x, y)(at)` replaced by `not_a_knot_spline`, the same computation (same LAPACK `gtsv` call, same coefficient formulas, same evaluation order) without scipy's input checks and object construction | `scoring.py` | 336 → 118 µs per call | **bit-identical**: 2,167 recorded and random curves before the change; `test_spline_matches_scipy_bit_for_bit` guards it |
| Report written with one `f.write(json.dumps(...))` instead of `json.dump` (thousands of small writes) | `pipeline.py` | 24.9 → 12.6 ms on `grid` (258 KB report) | byte-identical file. The write happens after `total_pipeline_s` is measured, so it does not show in the table below |

## Before/after (32 damaged images, serial, warm, `save_images=False`, same machine, back to back)

| Stage | Before (ms, sum) | After (ms, sum) | Change |
|---|---|---|---|
| P1 extract | 12,307 | 12,311 | 0.0% |
| P2 candidates | 951 | 1,036 | +8.9% (noise, code unchanged) |
| **P3 scoring** | **8,738** | **7,079** | **−19.0%** |
| P4 ASP | 136 | 133 | −2% |
| P5 synthesis | 329 | 326 | −1% |
| P7 output | 437 | 466 | +7% (noise, code unchanged) |
| **total** | **22,920** | **21,375** | **−6.7%** |

Per image: `grid` 687 → 602 ms, `geometric_art` 1,372 → 1,286 ms, `da37_gt` 1,304 → 1,265 ms.
Growth exponents after: P2 0.91, P3 0.87, total 0.40.

## Checked and left alone

- **sknw numba cache works.** The first `build_sknw` in a fresh process takes 0.32 s with the cache, against 3.47 s
  compiling without it (`.nbi` files present in `site-packages/sknw/__pycache__`).
- **P1 (Schneider fitting, 54–58% of the time).** `bezier_eval`, `_generate_bezier` and `_turning_angles_deg` are
  already vectorized. What remains is per-call NumPy overhead on short arrays. Speeding it up means reordering
  floating-point operations inside the recursive fit, and a last-bit change can flip a split decision
  (`err < max_error²`) and change the extracted paths. That is not worth about 0.2 s on the largest image.
  A future option is a numba/compiled fitter, verified against the snapshot.
- **`merge_restored_paths.component_with`** (linear scan) does not appear in the profile. P5 is ≤ 26 ms per image.
- **`_detect_symmetry` coarse-to-fine:** Phase 6 is off by default, and it would change results. Not done.
- **Compute vs I/O:** with figures, matplotlib adds about 1.4 s per image (more than all compute stages). The figures
  stay optional (`save_images`). `scripts/profile_pipeline.py` profiles both settings, so no new report key was added.
