# WP-5 — Scorer ablation

Branch `wp5-scorer-ablation` (from `wp4-synthetic-data`). **Defaults unchanged.** The shipped model is still the
scorecard at threshold 2/3, the snapshot is identical, and the tests pass (17).

## What was added

- **Gradient-boosted trees** as an alternative scorer: `train_scoring.py --model gbt`, scikit-learn
  `HistGradientBoostingClassifier`, one per bridge shape, on the same scorecard-scale measurements.
  - Monotonic constraints from `FEATURE_DIRECTIONS`, seeded, no early stopping, no class weights (probabilities stay calibrated).
  - Model file: JSON with `"kind": "gbt"` + `.joblib`. `load_model` / `score_candidates` handle both kinds, so the
    threshold → log-odds → ASP utility path is unchanged.
  - Trees give no per-measurement log-odds, so their reports name no measurement for or against a bridge
    (`scoring_model` lists the trees' measurements instead of weights).
- **Threshold tuned on the synthetic val split** (`--tune-threshold`, with `--extra-data`). The chosen threshold is
  the one that repairs most gaps on the val split's non-benchmark drawings, while drawing no more line off the drawing
  than the plain scorecard at 2/3 does there. It is chosen per CV fold in `check_changes.py`.
- `check_changes.py`: `--model`, `--tune-threshold`, `--dataset-cache` (reuses labelled candidates between runs), and
  it writes `cv/folds.json`.
- `scripts/evaluate_synthetic.py --fallback-model`.
- `scikit-learn` added to `requirements.txt`.
- Test `test_boosted_model_round_trips_and_respects_directions`: round trip, monotonicity, and a full `restore()`
  with a boosted model.

A design note: the plan's `restore(scorer=...)` argument was not needed. The model file states its kind, so
`restore(..., model_path=...)` is enough.

## Ablation on the 32 real pairs (held out by drawing; `check_changes.py`)

| scorer | training data | threshold | candidate separation | recall % | off-drawing (lw) | wrong strokes | strokes on undamaged | PSNR | verdict |
|---|---|---|---|---|---|---|---|---|---|
| scorecard (shipped) | benchmark | 2/3 | 96.8% | **87.2** | 89.4 | 5 | 23 | 26.06 | — |
| GBT | benchmark | 2/3 | 97.1% | 86.5 | 58.4 | 4 | 24 | 26.15 | keep (−35% / −0.8 pt) |
| scorecard | + synthetic easy, medium | 2/3 | 98.4% | 84.4 | 30.4 | 2 | 18 | 26.11 | not kept (−2.8 pt) |
| **scorecard** | **+ synthetic easy, medium** | **tuned (0.55–0.575)** | **98.4%** | **86.3** | **39.3** | **3** | 26 | 26.10 | **keep (−56% / −0.9 pt)** |
| GBT | + synthetic easy, medium | 2/3 | 98.6% | 82.0 | 36.4 | 2 | 17 | 26.03 | not kept (−5.2 pt) |
| GBT | + synthetic easy, medium | tuned (0.30–0.375) | 98.6% | 86.0 | 50.9 | 2 | 32 | 26.04 | keep (−43% / −1.3 pt) |

The tuned GBT thresholds hit the bottom of the search grid (0.30) in 3 of 5 folds, so that row is not fully tuned.
A wider grid would be needed to judge GBT + synthetic fairly.

## The best two on the synthetic test split (534 variants, 89 drawings never trained on)

| config | easy recall % / off-drawing (lw) | medium recall % / off-drawing | hard recall % | junction recall % | undamaged off-drawing (lw) |
|---|---|---|---|---|---|
| scorecard (shipped) | 91.8 / 146.3 | 88.1 / 149.8 | 69.8 | 64.6 | 70.5 |
| **scorecard + synthetic + tuned** | **91.9 / 122.6** | 87.4 / **138.9** | 70.6 | 63.0 | **50.1** |
| GBT (benchmark only) | 91.0 / 221.3 | **89.3** / 222.4 | 70.6 | 63.6 | 89.0 |

The hard level is left out of the comparison of off-drawing line: there it is dominated by the binarization failure
found in WP-4 (tens of thousands of noise strokes for every scorer).

## Interpretation

- **Recommended: scorecard + synthetic easy/medium training + val-tuned threshold.** On the real benchmark it more than
  halves the line drawn off the drawing (89.4 → 39.3 line widths) and wrong strokes (5 → 3), for 0.9 points of recall.
  It is the first configuration in this project that passes the keep rule. On unseen synthetic drawings it is as good
  or better (less off-drawing line at easy and medium, and on undamaged drawings). It stays a scorecard: interpretable,
  with per-measurement explanations in every report.
- **GBT** helps on the real pairs (−35% off-drawing) but does not generalize: on unseen synthetic drawings it draws
  about 50% more off-drawing line than the scorecard. Trees fit the 32 drawings' peculiarities, while the
  sign-constrained scorecard does not. It also loses the per-measurement explanations.
- **Synthetic data mainly sharpens the ranking** (separation 96.8 → 98.4%). It also makes the probabilities more
  cautious, which is why the threshold needs re-tuning (from 0.667 to about 0.55).
- **Unchanged weak spot:** junction gaps (63–65% recall, about 24% "candidate not selected" under every scorer). Per-pair
  features cannot tell which of several strokes at a junction continue each other. That is the motivation for WP-7
  (context-aware GNN).
- One cost of the recommended model: strokes drawn on undamaged originals go 23 → 26, while their off-drawing length
  falls.

## To adopt the recommendation (needs Mahmoud's approval)

```bash
python train_scoring.py --extra-data data/synthetic --extra-difficulties easy,medium --tune-threshold   # writes restoration/scoring_model.json
python check_changes.py --save-baseline   # new baseline
python regression_snapshot.py --rebless   # new exact snapshot
```

`data/synthetic` must exist first (`scripts/fetch_quickdraw.py`, `scripts/make_synthetic.py`). `check_changes.py`
judges future changes with the same training data only if it is given the same options; the defaults would need
updating to match.
