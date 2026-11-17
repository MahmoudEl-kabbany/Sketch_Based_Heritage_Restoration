# WP-3 — Preprocessing

Branch `wp3-preprocessing` (from `wp2-performance`). **Defaults unchanged.** The snapshot is identical for 64/64
images, and the default run of `check_changes.py` reports UNCHANGED.

## What was added

- **`restoration/preprocess.py`** holds the binarization, speck removal and skeleton. `PreprocessConfig()` reproduces the
  original behaviour bit for bit. The skeleton is now computed once and shared by the line-width measurement and the
  graph. `bezier.stroke_geometry()` delegates to it.
- **Variants** (fields of `PreprocessConfig`): `threshold` (`otsu` / `sauvola` / `niblack` / `adaptive_gaussian`),
  `denoise` (`median` / `nlmeans`, on the gray image), `close_px`, `skeleton` (`zhang` / `lee`), and
  `junction_aware_pruning`. With junction-aware pruning, a spur is removed only if its junction keeps two other
  branches longer than 3 junction radii.
- **How a variant is selected:** `extract_paths(..., preprocess=cfg)` and `restore(..., preprocess=cfg)` per call, or
  the environment variable `SKETCH_PREPROCESS` (JSON) for whole benchmark runs. `check_changes.py --preprocess '{...}'`
  sets it.
- **`check_changes.py`** now also stores the per-gap stage shares (`stages`) in `check_results.json`.
- **`scripts/compare_preprocessing.py`** runs `check_changes.py` once per variant, each with its **own refitted
  scorer** (held out by drawing), and prints the table below.
- **Unit test** `test_preprocessing_variants_are_validated_and_run`.

## Results (32 benchmark pairs, held-out refit per variant)

| variant | hypothesis | recall % | off-drawing (lw) | wrong strokes | strokes on undamaged | PSNR | SSIM | endpoints missing % | no candidate % | not selected % | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **default** | Otsu, no blur, Zhang | **87.24** | **89.4** | 5 | 23 | 26.06 | 0.9830 | 1.15 | 0.26 | 9.69 | — |
| median3 | H1: gray median 3 before Otsu | 85.97 | 78.5 | 2 | 22 | 26.02 | 0.9827 | 1.79 | 0.38 | 10.84 | trade-off, not kept (−12% / −1.3 pt) |
| sauvola | H2: Sauvola w25 | 84.44 | 74.0 | 4 | 17 | 25.63 | 0.9828 | 1.28 | 0.38 | 11.86 | trade-off, not kept (−17% / −2.8 pt) |
| sauvola51 | H2: Sauvola w51 | 85.33 | 80.6 | 2 | 20 | 25.92 | 0.9828 | 1.02 | 0.38 | 11.73 | trade-off, not kept (−10% / −1.9 pt) ¹ |
| adaptive | H2: adaptive Gaussian w25 | 77.42 | 184.9 | 16 | 29 | 24.90 | 0.9826 | 2.17 | 0.13 | 11.48 | **worse** ¹ |
| nlm_adaptive_close | H3: nlmeans + adaptive + close 3 | 76.79 | 61.7 | 2 | 12 | 24.93 | 0.9827 | 2.42 | 0.51 | 11.73 | trade-off, not kept (−31% / −10.4 pt) ¹ |
| **h4** | H4: junction-aware pruning | 87.12 | 70.8 | 7 | 17 | 26.05 | 0.9832 | **0.77** | 0.26 | 10.08 | trade-off, not kept (**−21% / −0.1 pt**) |
| lee | Lee skeleton | 85.59 | 82.8 | 4 | 20 | 25.70 | 0.9821 | 0.89 | 0.38 | 11.86 | trade-off, not kept (−7% / −1.6 pt) |
| h4_median3 | H4 + H1 | 87.12 | 86.0 | 4 | 19 | 26.04 | 0.9828 | 1.79 | 0.26 | 9.82 | trade-off, not kept (−4% / −0.1 pt) |
| h4_sauvola51 | H4 + Sauvola w51 | 86.73 | 72.3 | 4 | 19 | 26.04 | 0.9831 | 0.89 | 0.26 | 10.33 | trade-off, not kept (−19% / −0.5 pt) ¹ |

Verdicts use the keep rule. A variant is kept if the off-drawing line falls and recall does not (or the reverse). If
they move in opposite directions, it is kept only for ≥ 25% less off-drawing line at ≤ 2 points of recall lost.

¹ One unit test failed under this variant (the env variable also reaches the unit tests):
`test_extraction_does_not_depend_on_scale`, or the line-width check next to it. Fixed-pixel windows (Sauvola 51,
adaptive 25) are not scale-invariant, unlike the rest of Phase 1, which is measured in stroke radii.

## Interpretation

- **Nothing passes the keep rule, so the default stays Otsu / no blur / Zhang / radius-based pruning.**
- **H4 (junction-aware pruning) is the closest and the most promising.** It cuts the off-drawing line by 21% and
  extraction misses from 1.15% to 0.77% of gaps, at only 0.1 points of recall. It has fewer strokes on undamaged
  originals (17 vs 23) but more wrong strokes (7 vs 5). It falls 4 points short of the 25% rule. Worth revisiting with
  a tuned `JUNCTION_LONG_BRANCH_RADII` and on the WP-4 data.
- **H1 (median blur):** moot as the first draft stated it (the pipeline never blurred). A gray median trades recall for
  less off-drawing line. It increases extraction misses (1.15 → 1.79%), as the docs predicted: blurring moves stroke ends.
- **H2/H3 (local thresholds, denoising):** they lose recall on this benchmark, and adaptive Gaussian is clearly worse.
  This is expected: the 32 benchmark images are clean digital line art with even backgrounds, where Otsu is ideal.
  These methods target stained, unevenly lit scans, which the benchmark does not contain. **Re-test them on the WP-4
  degraded splits** (stains, fading, uneven light) before ruling them out. If they are pursued, make their window
  scale with the measured line width.
- **Lee skeleton:** fewer extraction misses (0.89%), but more candidates are left unselected. Not better overall.

## Recommendation to Mahmoud

Keep the defaults. Re-run `scripts/compare_preprocessing.py` once WP-4's degraded data exists, with H4, Sauvola
(line-width-scaled window) and nlmeans as the main candidates.
