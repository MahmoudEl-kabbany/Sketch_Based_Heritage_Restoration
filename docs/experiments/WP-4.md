# WP-4 — Synthetic damage dataset

Branch `wp4-synthetic-data` (from `wp3-preprocessing`). **Defaults unchanged.** The snapshot is identical, the default
`check_changes.py` reports UNCHANGED, and the shipped model is untouched.

## What was added

| File | Purpose |
|---|---|
| `synthetic_damage.py` | Typed gaps on the clean skeleton (sknw): straight (< 10° turn), curved (15–50°), corner (> 50°), junction (disk over a degree-≥3 node). Erasure is a capsule along the skeleton (radius ≥ half the stroke width + 1). Gaps stay apart, and stretch gaps are at most half their edge long. Degradation: fading, paper texture, stains, illumination gradient, blur, Gaussian noise, salt-and-pepper, JPEG, at three levels (`DIFFICULTY`). Deterministic per seed. Splits are by source drawing (hash: 70/15/15). |
| `scripts/fetch_quickdraw.py` | Streams the first 2,000 lines of 20 Quick, Draw! categories and keeps 25 seeded, recognized drawings with ≥ 3 strokes per category (500). Rasterized at 512 px, 4 px lines, to `data/quickdraw/` with `ATTRIBUTION.txt` (CC BY 4.0). |
| `scripts/make_synthetic.py` | Benchmark originals + Quick, Draw! → `data/synthetic/<split>/<drawing>/<difficulty>_<n>/{gapped,damaged}.png, gaps.json` + `manifest.json`. |
| `scripts/evaluate_synthetic.py` | `evaluate_gaps` per variant. Ground truth comes from (clean, gapped) and the pipeline runs on damaged. Each true gap is matched to its recorded type. Results by difficulty, gap type and source. Held-out models (`--models`) are used where available; Quick, Draw! drawings, which no model has seen, use the shipped model. |
| `evaluate_gaps.evaluate_pair` | New optional `reference_path` (ground truth from the undegraded gapped image) and `include_undamaged`. Defaults are unchanged. |
| `train_scoring.py` / `check_changes.py` | `--extra-data DIR` (train split), `--extra-split`, `--extra-difficulties`. Extra drawings get folds after the benchmark drawings, so the benchmark's CV folds do not move. |
| `regression/synthetic_samples/` | 21 committed samples (7 Quick, Draw! drawings at 256 px × 3 levels, 1.2 MB, with attribution). |
| Tests | `test_synthetic_damage_is_deterministic_and_matches_ground_truth`, `test_committed_synthetic_samples_regenerate_exactly`. |

**Dataset built:** 532 drawings (32 benchmark originals + 500 Quick, Draw!) × 3 levels × 2 variants = 3,192 variants
with 18,341 gaps (train 361 / val 82 / test 89 drawings). 722 MB in `data/synthetic/` (git-ignored), about 6.5 min to
generate.

**Check of the ground truth:** the gaps found by `build_ground_truth(clean, gapped)` match the erased ones (e.g.
25 erased → 25–27 components). Extra components are collateral clipping of a nearby stroke and are labelled `other`
(under 3% of gaps).

## Baseline: current pipeline on the synthetic test split

534 variants of 89 drawings. Models held out by drawing (`train_scoring.py --cv-dir`) for the benchmark drawings, and
the shipped model for Quick, Draw! drawings (never seen).

Overall: 3,321 gaps, **recall 83.2%**.

| difficulty | images | gaps | recall % | wrong strokes | off-drawing (lw) | bridged by extraction % | candidate not selected % | no candidate % | endpoints missing % |
|---|---|---|---|---|---|---|---|---|---|
| easy | 178 | 1103 | **91.8** | 8 | 146 | 2.9 | 4.8 | 0.1 | 0.4 |
| medium | 178 | 1109 | **88.1** | 7 | 150 | 2.4 | 9.2 | 0.0 | 0.3 |
| hard | 178 | 1109 | **69.8** | 26,182 | 25,692 | 12.4 | 13.9 | 1.9 | 2.0 |

| source | images | gaps | recall % |
|---|---|---|---|
| own (benchmark originals) | 30 | 603 | 85.4 |
| Quick, Draw! | 504 | 2718 | 82.7 |

| gap type | gaps | recall % | candidate not selected % | no candidate % | endpoints missing % |
|---|---|---|---|---|---|
| straight | 1469 | 90.3 | 4.8 | 0.2 | 0.3 |
| curved | 916 | 87.0 | 5.8 | 0.4 | 0.7 |
| corner | 395 | 75.4 | 14.4 | 0.5 | 0.8 |
| junction | 514 | **64.6** | **23.2** | 2.5 | 2.7 |
| other | 27 | 37.0 | 33.3 | 0.0 | 3.7 |

## Does synthetic training data help on the real benchmark?

`check_changes.py --extra-data data/synthetic` (train split only; the benchmark's held-out folds are unchanged):

| training data | candidate separation | recall % | off-drawing (lw) | wrong strokes | strokes on undamaged | PSNR | verdict |
|---|---|---|---|---|---|---|---|
| current (benchmark + its circle/box variants) | 96.8% | **87.2** | 89.4 | 5 | 23 | 26.06 | — |
| + synthetic, all levels | 82.9% | 30.0 | 69.7 | 0 | 3 | 22.98 | worse in practice (−57 pt recall) |
| + synthetic, easy + medium | **98.4%** | 84.4 | **30.4** | **2** | **18** | **26.11** | trade-off, not kept (−66% off-drawing for −2.8 pt; the rule allows ≤ 2 pt) |

## Interpretation

1. **The candidate-recall ceiling is high.** Only 0.1–1.9% of gaps never get a candidate. Phase 2 is not the
   bottleneck. Lost gaps are mostly *candidate not selected* (scorer/ASP: WP-5, WP-7) and, at the hard level,
   binarization failures.
2. **Junctions and corners are the weak gap types** (65% and 75% recall, against 90% for straight). In 23% of junction
   gaps the right candidate exists but is not chosen. This is what WP-7's context-aware scorer is meant to address.
3. **The hard level breaks global Otsu.** Where stains and the light gradient overlap, the background becomes as dark as
   the faded ink. The mask fills with noise, giving thousands of endpoints and wrong strokes (e.g. `shape_set/hard_1`:
   12,941 endpoints). The drawings stay legible to the eye, so this is a real weakness. It is the case WP-3's local
   thresholds are for: re-run `scripts/compare_preprocessing.py` on this split (hard level).
4. **Training on hard variants destroys the scorer.** Their thousands of noise candidates swamp the real ones.
   Training on easy + medium instead **improves candidate separation (96.8 → 98.4%) and cuts the off-drawing line by
   two thirds**, at 2.8 points of recall. The model becomes more cautious at the unchanged acceptance threshold (2:1
   odds). A threshold re-tuned on the val split might recover the recall. That belongs to WP-5, which needs the same
   machinery (`--extra-data`, val split).

## Recommendation to Mahmoud

Keep the shipped model for now (the keep rule says no). Take "train with synthetic easy + medium" into WP-5 as the
main candidate, together with a threshold chosen on the synthetic val split, and judge it again with `check_changes.py`.
