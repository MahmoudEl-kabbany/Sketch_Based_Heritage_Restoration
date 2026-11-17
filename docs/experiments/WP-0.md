# WP-0 — Safety net

Branch `wp0-safety-net` (from `accuracy_overhaul` @ `bcfd0d4`). No pipeline code was changed.

## What was added

| File | Purpose |
|---|---|
| `regression_snapshot.py` | Records and compares an exact snapshot of the output for the 64 benchmark images (32 damaged + 32 undamaged originals), under two variants: default settings and `efd_closure=True`. It records the accepted endpoint pairs, each final path (closed/open, segment count, control points rounded to 1e-9), and the Phase 6 summary. Comparison: structure exact, numbers within `1e-6`. `--rebless` records a new snapshot. It uses the shipped `scoring_model.json` and writes nothing to `restoration/outputs/`. |
| `regression/snapshot/` | The recorded snapshot: 64 JSON files, 3.2 MB. Paths drop the `test_images/` prefix, because `.gitignore` ignores that folder name at any depth. |
| `test_regression.py` | `test_output_matches_snapshot` (all 64 images in parallel, `REBLESS=1` to re-record). `test_restoration_is_deterministic` (one image restored twice in-process and once in a fresh subprocess, comparing the full report without timing keys). Both are skipped without `test_images/`. |
| `scripts/profile_pipeline.py` | Scaling probe (serial run over the 32 damaged images: per-stage time against free ends, and the growth exponent k in time ~ ends^k), plus cProfile on one image (default: the most free ends), with and without figures, after a warm-up run. |

`check_changes.py` still runs only `test_restoration.py`, so its runtime is unchanged.

## Verification on the untouched pipeline

- `python -m pytest test_restoration.py test_regression.py`: **11 passed** (37 s).
- `python regression_snapshot.py`: **64/64 identical** (26 s, 8 workers).
- `python check_changes.py`: **UNCHANGED**, 0 of 32 drawings changed (143 s):

| Metric (held-out models) | Value |
|---|---|
| Gaps repaired | 87.2% |
| Line off the drawing | 89.4 line widths |
| Wrong strokes | 5 |
| Strokes on undamaged originals | 23 |
| Placement, median | 0.12 line widths |
| Mean PSNR / SSIM | 26.06 dB / 0.9830 |

## Scaling probe (32 damaged images, `save_images=False`, serial, warm)

Times in ms, sorted by the number of free ends.

| image | ends | cands | P1 extract | P2 candidates | P3 scoring | P4 ASP | P5 synthesis | P7 output | total |
|---|---|---|---|---|---|---|---|---|---|
| ankh_damaged.png | 12 | 50 | 479 | 6 | 49 | 2 | 4 | 5 | 546 |
| restoration_test_big_damaged.png | 17 | 80 | 458 | 10 | 84 | 2 | 7 | 8 | 569 |
| eye_damaged.png | 25 | 108 | 755 | 13 | 111 | 3 | 11 | 10 | 903 |
| da58_gt_damaged.png | 28 | 118 | 144 | 12 | 116 | 3 | 5 | 6 | 287 |
| scarab_damaged.png | 30 | 136 | 714 | 13 | 136 | 3 | 14 | 14 | 894 |
| djed_pillar_damaged.png | 32 | 126 | 530 | 13 | 141 | 3 | 8 | 9 | 705 |
| star_junction_damaged.png | 32 | 164 | 30 | 30 | 270 | 2 | 1 | 4 | 338 |
| simple_lotus_damaged.png | 36 | 150 | 514 | 17 | 187 | 3 | 5 | 8 | 735 |
| flail_damaged.png | 40 | 159 | 617 | 17 | 148 | 4 | 6 | 10 | 803 |
| da9_gt_damaged.png | 43 | 182 | 119 | 18 | 159 | 3 | 4 | 7 | 311 |
| da61_gt_damaged.png | 44 | 179 | 145 | 16 | 207 | 4 | 7 | 8 | 387 |
| da10_gt_damaged.png | 45 | 199 | 115 | 28 | 358 | 4 | 5 | 9 | 519 |
| da6_gt_damaged.png | 45 | 192 | 102 | 20 | 170 | 3 | 4 | 6 | 307 |
| geometric_damaged.png | 45 | 216 | 165 | 21 | 179 | 5 | 5 | 11 | 386 |
| da57_gt_damaged.png | 48 | 195 | 182 | 20 | 188 | 3 | 5 | 7 | 406 |
| simple_cat_damaged.png | 48 | 214 | 203 | 21 | 194 | 3 | 4 | 7 | 433 |
| feather_damaged.png | 51 | 216 | 663 | 24 | 210 | 3 | 14 | 15 | 929 |
| grid_square_damaged.png | 52 | 219 | 54 | 21 | 228 | 4 | 4 | 8 | 321 |
| da22_gt_damaged.png | 56 | 233 | 201 | 29 | 258 | 6 | 8 | 10 | 513 |
| butterfly_damaged.png | 61 | 287 | 273 | 29 | 257 | 4 | 7 | 12 | 584 |
| da5_gt_damaged.png | 72 | 327 | 263 | 33 | 311 | 4 | 10 | 12 | 633 |
| da79_gt_damaged.png | 84 | 381 | 539 | 38 | 344 | 4 | 20 | 22 | 969 |
| ra_damaged.png | 88 | 356 | 375 | 39 | 325 | 4 | 14 | 19 | 776 |
| shape_set_damaged.png | 93 | 404 | 449 | 41 | 354 | 5 | 9 | 15 | 874 |
| enclosed_shapes_damaged.png | 96 | 419 | 458 | 40 | 468 | 4 | 9 | 19 | 999 |
| islamic_art_1_damaged.png | 96 | 446 | 618 | 46 | 384 | 14 | 16 | 26 | 1105 |
| da8_gt_damaged.png | 97 | 395 | 375 | 41 | 370 | 5 | 12 | 19 | 824 |
| geometric_art_damaged.png | 108 | 481 | 814 | 51 | 484 | 6 | 26 | 28 | 1409 |
| da12_gt_damaged.png | 110 | 502 | 422 | 59 | 428 | 5 | 11 | 19 | 946 |
| da78_gt_damaged.png | 117 | 500 | 750 | 51 | 457 | 5 | 24 | 30 | 1318 |
| da37_gt_damaged.png | 119 | 533 | 689 | 56 | 490 | 5 | 21 | 26 | 1286 |
| grid_damaged.png | 136 | 552 | 108 | 58 | 467 | 8 | 7 | 13 | 662 |

| Stage | Growth exponent k (time ~ ends^k) | Share of total time |
|---|---|---|
| P1 extract | 0.20 | 54.4% |
| P2 candidates | 0.93 | 4.1% |
| P3 scoring | 0.90 | 37.6% |
| P4 ASP | 0.48 | 0.6% |
| P5 synthesis | 0.62 | 1.4% |
| P7 output | 0.71 | 1.9% |
| total | 0.40 | — |

## Profile highlights (cProfile, warm; times inflated by the profiler)

**`grid_damaged.png`** (most free ends: 136, 552 candidates, 512×512), `save_images=False`, 1.23 s total:
- `score_candidates` 0.78 s, all in `_measure` (552 calls). Inside it: `compute_mjerk` 0.32 s, of which
  `scipy CubicSpline.__init__` is 0.23 s; `_resample` 0.14 s; `compute_mdtw` 0.12 s; `_crossed_ink` 0.11 s.
- `extract_paths` 0.17 s, `generate_candidates` 0.10 s, `json.dump` of the report 0.13 s.

**`grid_damaged.png`, `save_images=True`:** matplotlib adds about 1.4 s (text layout, image resampling and PNG
encoding). That is more than all compute stages together.

**`geometric_art_damaged.png`** (slowest extraction, 1240×1240), 2.17 s total:
- `extract_paths` 1.16 s, of which `_fit_cubic_single` (recursive Schneider split, 1236 calls) is 0.81 s,
  `bezier_eval` 0.18 s and `_turning_angles_deg` 0.17 s.
- `score_candidates` 0.72 s (same breakdown as above).

## Interpretation

- **No stage grows super-linearly.** P2 and P3 are linear in the number of free ends (k ≈ 0.9). P1 depends on the image
  size and stroke length rather than on the number of free ends (grid: 136 ends, 108 ms; geometric_art: 108 ends, 814 ms).
  ASP is negligible. The O(n³) loops the first-draft plan suspected no longer exist on this branch.
- **WP-2 should therefore be small.** The worthwhile targets are:
  1. P3: `compute_mjerk` builds a `CubicSpline` per candidate. A cheaper exact equivalent, or a vectorized batch,
     would cut about 30% of P3.
  2. P1: the Schneider recursion in `_fit_cubic_single` on large images.
  3. I/O: figures cost more than all compute stages, so keep `save_images=False` for benchmarks (already the default
     in the evaluation scripts).

  `component_with` (Phase 5) does not appear in the profile. P5 is ≤ 26 ms, so it can be left as is.
- The snapshot is 3.2 MB. A rebless rewrites it, so rebless only for intended changes, and commit the snapshot in the
  same commit as the change.
