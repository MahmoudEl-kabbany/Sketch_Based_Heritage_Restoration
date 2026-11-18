# Binarization on stained, unevenly lit pages (the synthetic "hard" level)

Branch `wp-hard-binarization` (from `wp-junction-measurement`). **Adopted, inference only** (Mahmoud, 2026-11-18): `PreprocessConfig().background` is now `"auto"`. The scorer is still trained on images binarized without flattening (`train_scoring.training_preprocess`).

| after adoption | before | now |
|---|---|---|
| real pairs (`check_changes.py`) | 86.3% / 39.3 lw off | **UNCHANGED**; shipped model current |
| exact snapshot (64 images) | — | identical |
| synthetic test, overall recall | 83.3% | **86.8%** |
| synthetic test, easy | 91.9% / 6 wrong | 91.9% / 6 wrong |
| synthetic test, medium | 87.4% / 7 wrong / 139 off | 87.5% / 6 wrong / 138 off |
| synthetic test, hard | 70.6% / 30,531 wrong | **81.2% / 11 wrong** |

## Problem

WP-4 found that global Otsu fails on the hard level. Where stains and a light gradient overlap, the paper gets as
dark as faded ink, and the mask fills with noise: about 600 endpoints per image, tens of thousands of wrong strokes,
70% recall.

## What was added (`restoration/preprocess.py`)

- `background="divide"`: estimate the paper's brightness with a grey-level closing (a disk of `background_size` = 6%
  of the diagonal, wider than any stroke or filled detail; on a reduced image, then smoothed), and divide the image by
  it before Otsu. Stains and gradients become flat white.
- `background="auto"`: the same, but only when the paper estimate varies by more than `UNEVEN_PAPER` = 0.05
  (1st to 99th percentile, relative). Measured values:

  | images | unevenness (closing 6% of diagonal) |
  |---|---|
  | 64 benchmark images | ≤ 0.028 |
  | synthetic easy | ≤ 0.028 |
  | synthetic medium | ≥ 0.080 |
  | synthetic hard | ≥ 0.168 |

  With `auto`, all 64 benchmark masks are bit-identical to Otsu's.
- `window_relative`: a local-threshold window as a share of the diagonal (scale-invariant, unlike WP-3's fixed pixels).

A 3% disk was tried first. On `simple_cat` the filled pupils are wider than 43 px, so the paper estimate went dark
there (unevenness 0.216) and the drawing changed. 6% fixes this.

## Screening: hard level, synthetic val split, shipped model (no refit)

| variant | recall % | wrong strokes | off-drawing (lw) | endpoints |
|---|---|---|---|---|
| Otsu (current) | 66.4 | 30,388 | 27,812 | 98,834 |
| background divide (3%) | 79.7 | 8 | 177 | 3,284 |
| divide + median 3 | 81.0 | 10 | 200 | 3,260 |
| Sauvola, window 3% of diagonal | 79.5 | 9 | 189 | 3,359 |
| Sauvola, 51 px (WP-3) | 80.3 | 7 | 184 | 3,300 |
| **auto (6%)** | **79.5** | **8** | **176** | — |

## Full check (refitted per variant; real pairs via `check_changes.py`, synthetic test split at all levels)

| variant | real recall % | real off-drawing (lw) | real drawings changed | easy | medium | hard (recall / wrong) |
|---|---|---|---|---|---|---|
| Otsu (shipped) | 86.3 | 39.3 | — | 91.9% / 123 off | 87.4% / 139 off | 70.6% / 30,531 |
| divide (3%) | 86.2 | 41.7 | 2 | 91.8% / 118 | 87.4% / 127 | **81.2% / 11** |
| divide + median | 86.1 | 57.0 | 20 | 93.4% / 135 | 90.1% / 188 | 83.7% / 9 |
| auto (6%) | 86.6 | 52.1 | 3 | 91.8% / 117 | 87.7% / 136 | **81.2% / 11** |

## Interpretation

- **Flattening the paper fixes the hard level.** Recall goes from 70.6 to 81.2% and wrong strokes from about 30,500
  to 11. Easy and medium are unchanged or slightly better.
- **On the real pairs the preprocessing itself changes nothing with `auto`** (identical masks). The +12.8 line widths
  in the `auto` row come from **retraining**: the synthetic *medium* training images are now normalized, so the
  scorecard's weights shift slightly, and 3 real drawings change. The keep rule rejects that trade-off.
- Therefore there are two ways to adopt it:
  1. Inference-only: `auto` at restoration time, the shipped model unchanged, and training images binarized as before.
     The real pairs are bit-identical, and stained pages get the full fix (val hard: 79.5% / 8 wrong).
  2. Everywhere, including training: consistent, but the retrained model costs 12.8 line widths on the real pairs.
