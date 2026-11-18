# T-joins: extending a stroke end to the stroke ahead of it

Branch `wp-tjoins` (from `wp-hard-binarization`). **Implemented, not in the shipped model.** The default output is
unchanged: the exact snapshot is identical, and the shipped model has no T-join scorecard.

## Why

End-to-end bridges cannot repair a gap between a stroke end and the side of another stroke, nor the stem of an erased
T-junction (17% of junction failures, see [junctions.md](junctions.md)). The README listed this as a known limitation.

## Potential (oracle, before building)

After a normal restoration, each still-free end was extended straight until it met any stroke or drawn bridge
(within 12 line widths), and the gaps this would repair were counted.

| | open gaps after the pipeline | repaired by all extensions | extensions | mostly off the drawing |
|---|---|---|---|---|
| 32 real pairs (784 gaps) | 93 | 38 (+4.8 pt recall) | 156 | 11 |
| synthetic val, easy + medium (2,066 gaps) | 153 | 57 (+2.8 pt) | 1,073 | 214 |

## What was built

- **`restoration/tjoins.py` (Phase 5b, after merging).**
  - Candidates: per free end, the nearest point of any final path (bridges included) within 12 line widths and a
    25° cone.
  - Seven measurements per candidate, a sign-constrained scorecard, and a straight-arriving cubic for the drawn T-join.
- **Model file:** an optional `"tjoin"` section (scorecard plus an optional `threshold`). Without it, Phase 5b does nothing.
- **`train_scoring.py`:** `--tjoins` and `--tjoin-threshold`. `labelled_tjoins` and `build_tjoin_dataset` label
  T-join candidates from the original drawing on the same training images as the bridges (`dataset_jobs`), after a
  restoration with the shipped model. `fit_tjoin` shares the logistic fit with the bridge scorecards (`_fit_logistic`).
- **`check_changes.py --tjoins`:** fits the T-join scorecard per fold, held out by drawing.
- **Report schema 4.2:** `analysis.tjoins`, `"T-Join"` events with explanations, and stroke ends taken by a T-join are
  no longer listed as open. The renderer and evaluation count `tjoin` segments as restoration strokes.
- **Test** `test_tjoin_extends_a_stroke_end_to_the_stroke_ahead`.

## Results (held out by drawing; shipped bridges + T-joins)

The weights are stable across the five folds: aim −0.57 to −0.76, incidence −0.41 to −0.55, crossed ink −0.31 to
−0.34, landing on a bridge +0.87 to +0.96, stub length +0.15 to +0.23, length −0.06 to −0.19. Of the T-join
candidates on the 32 provided pairs, 26% are correct.

| T-join threshold | real recall % | real off-drawing (lw) | wrong strokes | strokes on undamaged | synthetic test recall % |
|---|---|---|---|---|---|
| none (shipped) | 86.3 | 39.3 | 3 | 26 | 86.8 |
| 0.55 (bridges') | **87.9** | 49.9 | 4 | 28 | **88.0** |
| 0.7 | 86.9 | 44.0 | 4 | 27 | — |
| 0.8 | 86.6 | 40.1 | 3 | 26 | — |
| 0.9 | 86.4 | **39.3** | 3 | 26 | — |

At 0.55, the synthetic test split by level: easy 91.9 → 93.0%, medium 87.5 → 88.8%, hard 81.2 → 82.3%. Junction gaps
go 68.7 → 70.8%, but the off-drawing line rises by 20–30% at every level.

The 0.7–0.9 rows were read off the benchmark itself (the held-out fold models with only the T-join threshold
changed), so they show the trade-off; they are not an honest choice of threshold.

## Interpretation

- **T-joins work as intended.** They add 1.5 points of recall on the real pairs and 1.2 on synthetic drawings,
  including junction gaps (+2 points), and they land on freshly drawn bridges as designed (the strongest weight).
- **They cost more line off the drawing than the keep rule allows.** Only about a quarter of T-join candidates are
  right, and the seven measurements cannot separate them well enough. Thresholds that remove the extra off-drawing
  line also remove nearly all the gain (+0.1 points at 0.9).
- The oracle's 4.8 points show the room is there. The limit is distinguishing a missing stroke from a stroke that
  genuinely ends near another one. That needs evidence the current measurements lack, for example ink fragments in
  the gap, or whether the target stroke shows a junction bump (a widening of the stroke where another one meets it).

## How to use them anyway

```bash
python train_scoring.py --tjoins --output models/with_tjoins.json                       # T-joins at the bridges' threshold
python train_scoring.py --tjoins --tjoin-threshold 0.8 --output models/with_tjoins.json # stricter
python -c "from restoration.pipeline import restore; restore('damaged.png', model_path='models/with_tjoins.json')"
```
