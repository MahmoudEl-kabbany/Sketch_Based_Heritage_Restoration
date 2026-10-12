# Phase 3 — Scoring

> **Source file:** [`restoration/scoring.py`](../restoration/scoring.py)  
> **Fitted model:** [`restoration/scoring_model.json`](../restoration/scoring_model.json), written by [`train_scoring.py`](../train_scoring.py)

---

## 1. Purpose

Phase 3 answers one question for every candidate bridge: **how likely is it that this bridge redraws a stroke that is missing?**

Each candidate is described by a handful of geometric measurements. A **scorecard** turns them into a probability: a logistic regression with one weight per measurement. Nothing is hidden: the probability of a bridge is a sum of terms, one per measurement, and the report lists them.

No weight is set by hand. They are fitted from drawings whose undamaged originals are known, and every result quoted here comes from drawings the fitted model had not seen.

---

## 2. Pipeline Overview

```
List[ConnectionCandidate]  +  ExtractionResult  +  scoring_model.json
    │
    ├─► Measure every candidate (natural units: stroke widths, degrees, ...)
    ├─► Put each measurement on its scorecard scale (logarithm for lengths and ratios)
    │
    ├─► For each candidate, on the scorecard of its shape (smooth / corner):
    │       log-odds  = intercept + Σ weight × (value − mean) / scale
    │       probability = 1 / (1 + e^(−log-odds))
    │       score     = log-odds − log-odds of the acceptance threshold
    │
    └─► Sort candidates by score (best first)
```

A candidate with a positive score is more likely right than the threshold demands and may be drawn; Phase 4 decides which.

---

## 3. Measurements

All measurements are free of scale: they are in stroke widths, gap lengths, degrees or pure ratios, so they mean the same on a drawing with 3 px lines and one with 30 px lines. "Stroke width" is the mean width of the two strokes being joined.

### 3.1 Any bridge

| Name | Meaning | Enters as |
|---|---|---|
| `gap_length` | gap length in stroke widths | log2 |
| `width_mismatch` | width of the wider stroke relative to the narrower | log2 |
| `stub_shortness` | gap length relative to the length of the shorter stroke | log2 |
| `mutual_nearest` | share of the two ends for which the other is the nearest free end (0, ½, 1) | as is |
| `nearest_ratio` | gap length relative to the distance from each end to its nearest free end | log2 |
| `same_stroke` | the two ends belong to one stroke: the bridge closes it | as is |
| `crossed_ink` | existing stroke the bridge runs over between its ends, in stroke widths | log2(1 + x) |

`crossed_ink` is read off the image: the bridge is walked pixel by pixel, and stroke pixels that are not part of the two strokes it joins are counted. It replaces the veto that used to delete bridges after the solver had chosen them.

### 3.2 Smooth bridges

| Name | Meaning | Enters as |
|---|---|---|
| `aim_worst` | degrees by which the worse-aimed end points away from the other end | as is |
| `aim_best` | the same for the better-aimed end | as is |
| `turn` | change of direction between the two strokes, in degrees | as is |
| `sideways_offset` | stroke widths by which one stroke, continued straight, passes beside the other end | log2(1 + x) |
| `arc_miss` | gap lengths by which one stroke, continued with its own curvature, passes beside the other end | as is |
| `mdtw` | MDTW of the bridge with the stroke on either side (below) | ln(1 + x) |
| `mjerk` | MJerk of the bridge with the stroke on either side (below) | log10(1 + 10¹⁰ x) |

### 3.3 Corner bridges

| Name | Meaning | Enters as |
|---|---|---|
| `corner_turn` | change of direction at the corner, in degrees | as is |
| `corner_detour` | extra length of the corner bridge over the straight gap, in gap lengths | as is |
| `corner_balance` | how unequal the two legs are (0 = equal, 1 = one leg only) | as is |
| `corner_bend` | how curved the two strokes are where the bridge meets them (0 = straight) | as is |

### 3.4 MDTW and MJerk

Both are computed on the bridge **together with its context**: the stretch of stroke one gap length long on either side. This curve is resampled to 64 points at equal arc-length steps and scaled to a length of 100, so that neither the resolution nor the sampling of the path enters.

- **MDTW** — the dynamic-time-warping distance (`dtaidistance`) between that curve and the straight line joining its ends, computed for x and y and combined as `√(dtw_x² + dtw_y²)`. It is small when the bridge continues its strokes directly.
- **MJerk** — the mean squared third derivative of the curve with respect to arc length. The curve is first replaced by points every unit of length along a cubic spline through it; differentiating raw samples would measure the sampling, not the shape. It is small when curvature changes gently along the bridge.

---

## 4. The Scorecards

Smooth and corner bridges are judged by different evidence, so each shape has its own scorecard. A probability means the same on both.

```
log-odds = intercept + Σ weight × (transform(value) − mean) / scale
```

`mean` and `scale` are the average and spread of the measurement over the training candidates of that shape. Each term is therefore the log-odds a measurement adds **relative to an average candidate**; these terms are what the report calls a bridge's reasons.

The weights below are those in `scoring_model.json` (fitted on all 32 drawings). A weight of −1 means: one standard deviation more of this measurement divides the odds by e ≈ 2.7.

**Smooth bridges** (intercept −5.58)

| Measurement | Weight |
|---|---|
| `crossed_ink` | −2.79 |
| `aim_best` | −1.70 |
| `gap_length` | −0.79 |
| `sideways_offset` | −0.78 |
| `aim_worst` | −0.40 |
| `nearest_ratio` | −0.25 |
| `mdtw` | −0.20 |
| `mutual_nearest` | +0.18 |
| `mjerk` | −0.13 |
| `same_stroke` | −0.12 |

**Corner bridges** (intercept −2.28)

| Measurement | Weight |
|---|---|
| `crossed_ink` | −2.63 |
| `gap_length` | −1.35 |
| `corner_turn` | −1.13 |
| `nearest_ratio` | −0.78 |
| `corner_bend` | −0.40 |
| `width_mismatch` | −0.33 |
| `same_stroke` | −0.16 |
| `corner_detour` | −0.12 |
| `corner_balance` | −0.12 |

### 4.1 Weights keep to the direction of their meaning

`FEATURE_DIRECTIONS` states, for most measurements, which way they may count: a longer gap, a worse aim or more crossed stroke can only count against a bridge; being each other's nearest end can only count for it. The fit keeps every weight within its direction. Without this, correlated measurements can take opposite signs that cancel out, and a weight would read the wrong way in an explanation ("more abrupt, therefore more plausible"). Holding the directions costs nothing measurable: measured before the scorecards were split by shape, the chance that a correct candidate outscores a wrong one was 96.2% either way.

A measurement that would have counted the wrong way gets no weight and is left off the scorecard. In the fitted model these are `width_mismatch`, `stub_shortness`, `turn` and `arc_miss` for smooth bridges and `stub_shortness` and `mutual_nearest` for corner bridges: what they say is already said by the others.

### 4.2 The acceptance threshold

A bridge is drawn when it is at least **twice as likely to be right as wrong**: probability above 2/3 (`ACCEPTANCE_ODDS = 2` in `train_scoring.py`). Drawing a wrong stroke into a heritage drawing is worse than leaving a gap open, hence more than even odds. The threshold is stored in the model file; it is a stated preference, not a fitted number, and it can be relied on because the probabilities are calibrated (section 5.3).

---

## 5. How the Scorecards Are Fitted

```bash
python train_scoring.py                    # fit, report, write restoration/scoring_model.json
python train_scoring.py --cv-dir models/   # also write every drawing's held-out model
```

### 5.1 Labelled candidates

For every drawing with an undamaged original, the true gaps are known: stroke pixels present in the original and missing in the damaged image. Candidates are generated on the damaged image exactly as the pipeline does, and each is labelled:

> **correct** — both of its ends lie next to one true gap, and at least 80% of what it adds (the part of it not on surviving strokes) lies on that gap's erased stroke.

Every other candidate is not correct. Each drawing contributes five images: the provided damaged version, three more made with `add_gaps.py` (gaps 2 to 8 stroke widths across, as many as the provided version has), and the undamaged original, on which no candidate is correct.

### 5.2 Cross-validation by drawing

The 32 drawings are split into 5 folds. A drawing's candidates are scored by a model fitted on the other four folds, so it has never been seen in any damaged form. Everything reported, here and in the README, is measured that way. The model that ships is then fitted on all drawings.

The fit is a logistic regression on standardized measurements, minimizing the log loss plus half the squared length of the weights, with each weight within its direction (`scipy.optimize`, L-BFGS-B).

### 5.3 Results on held-out drawings

| | Scorecards | Gap length alone |
|---|---|---|
| Chance that a correct candidate outscores a wrong one | **96.8%** | 86.4% |
| Stroke ends whose best-scored candidate is a correct one | **92.7%** | 66.4% |

Do the probabilities mean what they say?

| Predicted probability | Candidates | Share that are correct |
|---|---|---|
| 0.00 – 0.10 | 24,823 | 1.0% |
| 0.10 – 0.33 | 3,403 | 20.5% |
| 0.33 – 0.50 | 1,402 | 41.8% |
| 0.50 – 0.67 | 1,372 | 59.6% |
| 0.67 – 0.80 | 1,022 | 70.9% |
| 0.80 – 0.90 | 1,224 | 88.5% |
| 0.90 – 1.00 | 2,684 | 95.1% |

### 5.4 Measured and left out

- **A cue for cut stroke ends.** How square a stroke end is separates the synthetic cuts of this dataset from natural stroke ends. Added as two measurements it changed nothing (96.8% either way, same gap recall), and it would not transfer to other kinds of damage. It is not in the model.
- **One scorecard for both shapes** repaired fewer gaps (80.7% against 83.0% at the same threshold in the candidate-level test): the evidence for a corner is not the evidence for a smooth continuation.
- **Fitting on the provided damage only**, without the synthetic versions, lost 1.6 points of gap recall and drew 77% more line off the drawing.

---

## 6. Entry Points

```python
def score_candidates(candidates, result, model=None) -> List[ConnectionCandidate]
    # fills features, contributions, probability and score; returns best first

def measure_candidates(candidates, result) -> np.ndarray     # measurements in natural units
def candidate_features(candidates, result) -> np.ndarray     # the same on their scorecard scale
def load_model(path=MODEL_PATH) -> ScoringModel
```

`compute_mdtw(curve)`, `compute_mjerk(curve)` and `context_curve(candidate, dense_samples, tail_len)` are also used by `evaluate_bridge_vs_original.py`.

---

## 7. Dependencies

| Library | Usage |
|---|---|
| `dtaidistance` | DTW distance (compiled version when available) |
| `scipy` | cubic spline for MJerk, KD-tree for nearest free ends; the fit in `train_scoring.py` |
| `numpy` | everything else |
